"""边缘端麦克风采集工具：输出标准 PCM WAV 文件。"""

import wave
from pathlib import Path


class AudioCaptureError(Exception):
    """表示录音依赖缺失或录音参数非法。"""

    pass


def record_wav(
    output_path: Path,
    *,
    seconds: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """按指定采样率和声道录制麦克风音频到 WAV 文件。"""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioCaptureError(
            "Microphone capture requires sounddevice. Install it on the edge VM if needed."
        ) from exc

    if seconds <= 0:
        raise AudioCaptureError("Recording seconds must be greater than 0.")
    if channels <= 0:
        raise AudioCaptureError("Recording channels must be greater than 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # sounddevice 接收采样点数量；16 位整数与后续 WAV 写入格式一致。
    frames = int(seconds * sample_rate)
    recording = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="int16")
    sd.wait()

    # WAV 头部与 PCM 字节流一起写出，使云端/本地 STT 都可直接读取。
    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(recording.tobytes())

    return output_path
