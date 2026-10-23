"""WAV 音频的轻量边缘侧检查、单声道转换与重采样。

该实现刻意只支持课程演示所需的 PCM WAV；它使用最近邻抽样而非高保真
重采样，目标是降低上传前的处理依赖和资源消耗。
"""

import io
import wave
from dataclasses import dataclass


class AudioPreprocessError(Exception):
    """表示 WAV 内容、尺寸或受支持格式不符合边缘处理要求。"""

    pass


@dataclass(frozen=True)
class AudioMetadata:
    """从 WAV 头部解析出的音频元数据，不包含原始样本字节。"""
    sample_rate: int
    channels: int
    sample_width: int
    duration_seconds: float
    size_bytes: int


def inspect_wav(content: bytes) -> AudioMetadata:
    """读取 WAV 文件头并返回采样率、声道、位宽、时长和总字节数。"""
    if not content:
        raise AudioPreprocessError("Audio content is empty.")

    try:
        with wave.open(io.BytesIO(content), "rb") as source:
            frames = source.getnframes()
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
    except wave.Error as exc:
        raise AudioPreprocessError("Only WAV audio can be locally preprocessed.") from exc

    # WAV 帧数除以每秒帧数得到时长；零采样率时避免除零。
    duration_seconds = frames / sample_rate if sample_rate else 0.0
    return AudioMetadata(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        duration_seconds=duration_seconds,
        size_bytes=len(content),
    )


def preprocess_wav(
    content: bytes,
    *,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    max_bytes: int = 20 * 1024 * 1024,
) -> bytes:
    """将 WAV 调整到目标采样率/声道并控制上传字节数。"""
    metadata = inspect_wav(content)
    if metadata.size_bytes > max_bytes:
        raise AudioPreprocessError(f"Audio exceeds max edge upload size: {max_bytes} bytes.")

    try:
        with wave.open(io.BytesIO(content), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())
    except wave.Error as exc:
        raise AudioPreprocessError("Failed to read WAV audio.") from exc

    sample_width = params.sampwidth
    channels = params.nchannels
    sample_rate = params.framerate

    if channels != target_channels:
        # 当前仅实现双声道 16-bit PCM 混合为单声道的安全最小范围。
        frames = _convert_channels(
            frames,
            sample_width=sample_width,
            source_channels=channels,
            target_channels=target_channels,
        )
        channels = target_channels

    if sample_rate != target_sample_rate:
        # 最近邻帧映射不改变单个采样点的位宽，仅改变帧序列长度。
        frames = _resample_nearest(
            frames,
            sample_width,
            channels,
            sample_rate,
            target_sample_rate,
        )
        sample_rate = target_sample_rate

    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(frames)

    processed = output.getvalue()
    # 重采样后仍可能超限，因此在重新编码后再次检查。
    if len(processed) > max_bytes:
        raise AudioPreprocessError(
            f"Preprocessed audio exceeds max edge upload size: {max_bytes} bytes."
        )
    return processed


def _convert_channels(
    frames: bytes,
    *,
    sample_width: int,
    source_channels: int,
    target_channels: int,
) -> bytes:
    """把双声道 16-bit PCM 的左右声道平均为一个单声道采样点。"""
    if source_channels == target_channels:
        return frames
    if sample_width != 2:
        raise AudioPreprocessError("Only 16-bit WAV channel conversion is supported.")
    if source_channels != 2 or target_channels != 1:
        raise AudioPreprocessError(
            f"Unsupported channel conversion: {source_channels} -> {target_channels}."
        )

    output = bytearray()
    frame_width = sample_width * source_channels
    for offset in range(0, len(frames), frame_width):
        # 一个 frame 同时包含同一时刻的左右声道采样。
        frame = frames[offset : offset + frame_width]
        if len(frame) < frame_width:
            break
        left = int.from_bytes(frame[0:2], "little", signed=True)
        right = int.from_bytes(frame[2:4], "little", signed=True)
        mono = int((left + right) / 2)
        output.extend(mono.to_bytes(2, "little", signed=True))
    return bytes(output)


def _resample_nearest(
    frames: bytes,
    sample_width: int,
    channels: int,
    source_rate: int,
    target_rate: int,
) -> bytes:
    """使用目标帧到源帧的最近索引映射完成简单重采样。"""
    if source_rate <= 0 or target_rate <= 0:
        raise AudioPreprocessError("Sample rate must be greater than 0.")
    if source_rate == target_rate:
        return frames

    frame_width = sample_width * channels
    if frame_width <= 0:
        raise AudioPreprocessError("Invalid WAV frame width.")

    source_frame_count = len(frames) // frame_width
    target_frame_count = int(source_frame_count * target_rate / source_rate)
    output = bytearray()
    for target_index in range(target_frame_count):
        # 反向映射能保证输出帧数与目标采样率成比例。
        source_index = int(target_index * source_rate / target_rate)
        offset = source_index * frame_width
        output.extend(frames[offset : offset + frame_width])
    return bytes(output)
