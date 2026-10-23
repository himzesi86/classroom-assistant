"""基于 ffmpeg 从课堂视频提取或分段导出标准 WAV 音频。"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path


class AudioExtractionError(Exception):
    """表示视频输入、ffmpeg 可用性或命令执行失败。"""

    pass


def extract_wav_from_video(
    video_path: Path,
    output_path: Path | None = None,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> Path:
    """从一个视频导出单个 16-bit PCM WAV 文件。"""
    if not video_path.exists():
        raise AudioExtractionError(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise AudioExtractionError(f"Video path is not a file: {video_path}")
    if sample_rate <= 0:
        raise AudioExtractionError("Sample rate must be greater than 0.")
    if channels <= 0:
        raise AudioExtractionError("Channels must be greater than 0.")
    if shutil.which("ffmpeg") is None:
        raise AudioExtractionError("ffmpeg is required but was not found in PATH.")

    target_path = output_path or video_path.with_suffix(".wav")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # -vn 丢弃视频流，强制输出便于 STT 处理的 PCM 16-bit 单/多声道音频。
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(target_path),
    ]

    # 不使用 shell，路径中含空格时也能作为独立参数安全传给 ffmpeg。
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or "unknown ffmpeg error"
        raise AudioExtractionError(f"Failed to extract WAV audio: {detail}")

    return target_path


def extract_wav_chunks_from_video(
    video_path: Path,
    output_dir: Path,
    *,
    chunk_seconds: int = 480,
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> list[Path]:
    """从视频提取定长 WAV 分段，并仅返回本次命令新生成的文件。"""
    if not video_path.exists():
        raise AudioExtractionError(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise AudioExtractionError(f"Video path is not a file: {video_path}")
    if chunk_seconds <= 0:
        raise AudioExtractionError("Chunk seconds must be greater than 0.")
    if sample_rate <= 0:
        raise AudioExtractionError("Sample rate must be greater than 0.")
    if channels <= 0:
        raise AudioExtractionError("Channels must be greater than 0.")
    if shutil.which("ffmpeg") is None:
        raise AudioExtractionError("ffmpeg is required but was not found in PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 记录开始时间，过滤输出目录中旧的同名分段文件。
    started_at = time.time()
    output_pattern = output_dir / f"{video_path.stem}_%03d.wav"
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or "unknown ffmpeg error"
        raise AudioExtractionError(f"Failed to extract WAV audio chunks: {detail}")

    chunks = sorted(
        path
        for path in output_dir.glob(f"{video_path.stem}_*.wav")
        if path.stat().st_mtime >= started_at - 1
    )
    if not chunks:
        raise AudioExtractionError("ffmpeg completed but no WAV chunks were created.")
    return chunks


def build_parser() -> argparse.ArgumentParser:
    """定义单文件与分段提取共用的命令行参数。"""
    parser = argparse.ArgumentParser(description="Extract WAV audio from a video file.")
    parser.add_argument("video", help="Input video path, for example classroom.mp4.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output WAV path. Defaults to the input path with .wav extension.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        help="Split output into WAV chunks, for example 480 for 8-minute chunks.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for chunked WAV output. Defaults to tmp/audio_chunks.",
    )
    return parser


def main() -> None:
    """解析 CLI 参数，执行提取，并把可读错误转为退出信息。"""
    args = build_parser().parse_args()
    try:
        if args.chunk_seconds:
            # 指定分段长度时，输出目录而非单个输出路径成为主要结果。
            chunks = extract_wav_chunks_from_video(
                Path(args.video),
                Path(args.output_dir) if args.output_dir else Path("tmp/audio_chunks"),
                chunk_seconds=args.chunk_seconds,
                sample_rate=args.sample_rate,
                channels=args.channels,
                overwrite=not args.no_overwrite,
            )
            for chunk in chunks:
                print(chunk)
            return

        output_path = extract_wav_from_video(
            Path(args.video),
            Path(args.output) if args.output else None,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=not args.no_overwrite,
        )
    except AudioExtractionError as exc:
        raise SystemExit(f"audio extraction error: {exc}") from exc

    print(output_path)


if __name__ == "__main__":
    main()
