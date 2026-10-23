# AGENTS.md

## 1. Project Overview

This project is a classroom assistant system focused on classroom Q&A and knowledge explanation with text and speech input.

Core pipeline:

```text
STT (Speech-to-Text) -> LLM API -> Text Answer
```

The system supports:

- Text-based question answering
- Voice input converted to text through STT
- Multi-turn dialogue with context
- Teaching-style explanations

## 2. System Architecture

The system is logically divided into three layers. Even when deployed locally, these layers must remain separated.

### 2.1 Edge Service

Responsibilities:

- Capture audio input, if enabled, or accept text input
- Send text to the cloud API
- Avoid containing any LLM logic

### 2.2 Cloud Service

The cloud service is the core service.

Responsibilities:

- Provide HTTP APIs
- Handle STT when needed
- Call LLM APIs, such as DeepSeek or other providers
- Manage prompt templates
- Manage multi-turn conversation context
- Return structured responses

### 2.3 Frontend Service

Responsibilities:

- Display answers
- Send user queries
- Show conversation history

## 3. Cloud Backend Responsibilities

The cloud backend must:

1. Provide API endpoints.
2. Handle user input, including text and STT results.
3. Call the LLM API.
4. Manage conversation context.
5. Apply prompt templates.
6. Return clean, readable answers.

## 4. Core Functionality

### 4.1 Classroom Q&A

Input:

- Text
- Speech converted to text through STT

Output:

- Clear explanation
- Optional structured answer

Assistant behavior:

- Acts like a teaching assistant
- Explains concepts clearly
- Avoids unnecessary verbosity
- Supports follow-up questions

## 5. Prompt Engineering Rules

### 5.1 Prompt Storage

All prompts must be stored centrally:

```text
app/prompts/
  classroom_qa_system.txt
```

Do not hardcode prompts across files.

### 5.2 Prompt Design Principles

Prompts must:

- Define the assistant role clearly
- Enforce concise explanation
- Reduce hallucination
- Encourage educational responses

### 5.3 Assistant Persona

The assistant should:

- Explain like a tutor
- Be precise and structured
- Avoid an overly casual or chatty tone

## 6. Context Management

### 6.1 Requirements

- Support multi-turn conversations
- Maintain recent dialogue history
- Identify sessions through `session_id`

### 6.2 Implementation Strategy

Minimum implementation:

- Store messages in an in-memory context store
- Retrieve the last N turns
- Truncate old history

### 6.3 Rules

- Do not send full history blindly.
- Limit context length.
- Prefer the last few turns.

## 7. API Design

### 7.1 Required Endpoints

#### Health Check

```http
GET /health
```

#### Chat API

```http
POST /api/chat
```

Request:

```json
{
  "session_id": "string",
  "message": "string"
}
```

Response:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "answer": "string"
  }
}
```

#### STT API

The STT API is an optional wrapper.

```http
POST /api/stt
```

Request:

```text
audio file
```

Response:

```json
{
  "text": "recognized speech"
}
```

#### History API

```http
GET /api/history?session_id=xxx
```

### 7.2 Response Format

All APIs must follow this response format:

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

## 8. LLM Integration Rules

### 8.1 Isolation

LLM calls must be isolated in:

```text
app/integrations/llm_client.py
```

Do not call the LLM directly inside route handlers.

### 8.2 Configuration

All keys must come from environment variables:

```text
LLM_API_KEY
LLM_BASE_URL
```

Never hardcode keys.

### 8.3 Error Handling

The LLM integration must handle:

- Timeout
- API failure
- Empty response

## 9. Code Structure

Recommended structure:

```text
project/
  AGENTS.md
  README.md

  app/
    main.py

    api/
      routes_chat.py

    services/
      chat_service.py
      context_service.py

    integrations/
      llm_client.py
      stt_client.py

    prompts/
      classroom_qa_system.txt

    models/
      schemas.py

  edge/
    client.py

  frontend/
    index.html
```

## 10. Testing Requirements

Minimum tests:

- Chat API works.
- Context is preserved.
- STT works, if enabled.
- API returns the correct response format.

## 11. Performance Metrics

The system should measure:

- API latency
- LLM response time
- STT latency

## 12. Logging

The system must log:

- Incoming requests
- LLM calls
- Response time
- Errors

## 13. Non-Goals

Do not implement:

- TTS
- Multimodal image analysis
- Complex microservices
- Distributed systems

## 14. Development Priorities

Implement in this order:

1. Chat API
2. Prompt system
3. Context management
4. STT integration
5. History API

## 15. Definition of Done

A feature is complete when:

- The API works locally.
- Context works correctly.
- Prompts are reusable.
- Output is clear and correct.
- The system can be demoed.

## 16. Final Instruction

Keep the system:

- Simple
- Modular
- Easy to demo
- Easy to explain
