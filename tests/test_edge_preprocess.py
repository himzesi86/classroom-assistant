"""边缘侧 WAV 构造、规范化和异常输入测试。"""

import io
import wave

import pytest

from edge.preprocess import AudioPreprocessError, inspect_wav, preprocess_wav


def make_wav(
    *,
    sample_rate: int = 8000,
    channels: int = 2,
    sample_width: int = 2,
    frames: int = 800,
) -> bytes:
    """构造指定参数的静音 PCM WAV 字节，避免依赖真实麦克风文件。"""
    payload = b"\x00" * frames * channels * sample_width
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(payload)
    return output.getvalue()


def test_preprocess_wav_converts_to_mono_and_target_rate() -> None:
    """双声道低采样率输入应被转换为单声道目标采样率。"""
    processed = preprocess_wav(
        make_wav(),
        target_sample_rate=16000,
        target_channels=1,
    )

    metadata = inspect_wav(processed)
    assert metadata.sample_rate == 16000
    assert metadata.channels == 1
    assert metadata.duration_seconds > 0


def test_preprocess_wav_rejects_empty_audio() -> None:
    """空字节输入必须在读取 WAV 前被拒绝。"""
    with pytest.raises(AudioPreprocessError, match="empty"):
        preprocess_wav(b"")


def test_preprocess_wav_rejects_oversized_audio() -> None:
    """原始 WAV 超过边缘上传限制时必须抛出可读错误。"""
    with pytest.raises(AudioPreprocessError, match="max edge upload size"):
        preprocess_wav(make_wav(), max_bytes=10)
