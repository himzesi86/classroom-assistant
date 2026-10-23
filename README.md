# Classroom Assistant API

Minimal cloud and edge demo for direction C: classroom Q&A / knowledge explanation with text and speech input.

## Current Scope

- `GET /health`
- `POST /api/chat`
- `POST /api/stt`
- Edge-side text/audio collector in `edge/`
- Central prompt file in `app/prompts/`
- Isolated LLM client in `app/integrations/llm_client.py`
- Isolated STT client in `app/integrations/stt_client.py`
- Bounded in-memory conversation context

## Context Behavior

The chat API supports recent-turn multi-turn context by `session_id`.

Current limits:

- Store up to 20 messages per session in memory.
- Send the latest 10 messages to the LLM.
- Context is reset when the API process restarts.

This is not unlimited long-context storage.

## Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install optional offline STT dependencies:

```bash
python3 -m pip install -r requirements-offline-stt.txt
```

Start the API server:

```bash
python3 -m uvicorn app.main:app --reload
```

Test the text API:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Explain Newton's first law."}'
```

Test the STT API:

```bash
curl -X POST http://127.0.0.1:8000/api/stt \
  -F "file=@/path/to/audio.wav" \
  -F "language=zh"
```

## Edge Collector

The edge side matches member A's task: run in a VM, collect classroom input, do light preprocessing, and upload to the cloud API.

Send typed input from the edge VM:

```bash
python -m edge.client --text "什么是牛顿第一定律？" --session-id classroom-a
```

Upload a local WAV file. The edge script normalizes WAV audio to the configured sample rate/channels before calling `/api/stt`, then sends the recognized text to `/api/chat`.

```bash
python -m edge.client --audio ./samples/question.wav --session-id classroom-a --language zh
```

Extract a WAV file from a classroom video before uploading it:

```bash
python -m edge.extract_audio ./samples/classroom.mp4 -o ./samples/classroom.wav
```

For a full class recording, split the video into STT-friendly chunks. Eight-minute chunks are about 15 MB at 16 kHz mono PCM, which keeps each upload under the default edge limit.

```bash
python -m edge.extract_audio ./samples/classroom.mp4 --chunk-seconds 480 --output-dir ./tmp/audio_chunks
```

Transcribe the generated WAV chunks locally with `faster-whisper`:

```bash
python -m edge.transcribe_chunks ./tmp/audio_chunks -o ./tmp/classroom_transcript.txt --model base --language zh
```

For a fast edge-side rough transcript preview before the final `base` transcript:

```bash
python -m edge.transcribe_chunks ./tmp/audio_chunks -o ./tmp/classroom_transcript_base.txt --model base --language zh --rough-model tiny
```

Record from a microphone, if `sounddevice` is installed on the VM:

```bash
python -m edge.client --record ./tmp/question.wav --session-id classroom-a --language zh
```

Optional video frame sampling for later multimodal expansion is in `edge/video_capture.py`. It requires `opencv-python` only when used.

Edge environment variables:

```bash
export EDGE_CLOUD_BASE_URL="http://127.0.0.1:8000"
export EDGE_SESSION_ID="classroom-a"
export EDGE_LANGUAGE="zh"
export EDGE_AUDIO_SECONDS="5"
export EDGE_AUDIO_SAMPLE_RATE="16000"
export EDGE_AUDIO_CHANNELS="1"
export EDGE_MAX_AUDIO_BYTES="20971520"
```

## LLM Configuration

By default, the API returns a local demo response when `LLM_API_KEY` is not set.

To call a real OpenAI-compatible LLM API, set:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-reasoner"
export LLM_MOCK="false"
export LLM_TIMEOUT_SECONDS="60"
```

## STT Configuration

The STT API supports three backends:

- `mock`: fake transcript for API testing; it does not recognize speech.
- `local`: offline transcription through `faster-whisper`.
- `api`: OpenAI-compatible transcription provider.

By default, the STT API uses `mock` so the upload route can be tested without installing a model.

For offline STT:

```bash
export STT_BACKEND="local"
export STT_LOCAL_MODEL="base"
export STT_LOCAL_DEVICE="cpu"
export STT_LOCAL_COMPUTE_TYPE="int8"
export STT_TIMEOUT_SECONDS="120"
export STT_MAX_AUDIO_BYTES="26214400"
```

The first local run may download the selected Whisper model. After the model is cached locally, transcription does not call a cloud API.

To call an OpenAI-compatible transcription API, set:

```bash
export STT_API_KEY="your-stt-api-key"
export STT_BASE_URL="https://api.openai.com/v1"
export STT_MODEL="whisper-1"
export STT_BACKEND="api"
export STT_TIMEOUT_SECONDS="60"
export STT_MAX_AUDIO_BYTES="26214400"
```

## Tests

```bash
pytest
```
