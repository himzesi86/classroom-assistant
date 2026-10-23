"""视频音轨提取命令构造和 ffmpeg 错误映射测试。"""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from edge.extract_audio import (
    AudioExtractionError,
    extract_wav_chunks_from_video,
    extract_wav_from_video,
)


TEST_DIR = Path("tmp/test_extract_audio")


def test_extract_wav_from_video_builds_ffmpeg_command(monkeypatch) -> None:
    """单文件导出应生成固定采样率、单声道 PCM 的 ffmpeg 参数。"""
    test_dir = TEST_DIR / "success"
    test_dir.mkdir(parents=True, exist_ok=True)
    video_path = test_dir / "classroom.mp4"
    output_path = test_dir / "classroom.wav"
    video_path.write_bytes(b"fake-video")
    calls = []

    monkeypatch.setattr("edge.extract_audio.shutil.which", lambda name: "ffmpeg")

    def fake_run(command, capture_output, check):
        """记录模拟命令参数，并返回成功的 CompletedProcess。"""
        calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "check": check,
            }
        )
        return CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr("edge.extract_audio.subprocess.run", fake_run)

    result = extract_wav_from_video(video_path, output_path)

    assert result == output_path
    assert calls[0]["command"] == [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]


def test_extract_wav_from_video_rejects_missing_video() -> None:
    """输入视频不存在时不得尝试调用 ffmpeg。"""
    with pytest.raises(AudioExtractionError, match="does not exist"):
        extract_wav_from_video(TEST_DIR / "missing.mp4")


def test_extract_wav_from_video_reports_ffmpeg_failure(monkeypatch) -> None:
    """ffmpeg 非零退出时，应将 stderr 映射为 AudioExtractionError。"""
    test_dir = TEST_DIR / "failure"
    test_dir.mkdir(parents=True, exist_ok=True)
    video_path = test_dir / "classroom.mp4"
    video_path.write_bytes(b"fake-video")
    monkeypatch.setattr("edge.extract_audio.shutil.which", lambda name: "ffmpeg")

    def fake_run(command, capture_output, check):
        """模拟 ffmpeg 返回无音轨错误。"""
        return CompletedProcess(command, 1, b"", b"no audio stream")

    monkeypatch.setattr("edge.extract_audio.subprocess.run", fake_run)

    with pytest.raises(AudioExtractionError, match="no audio stream"):
        extract_wav_from_video(video_path)


def test_extract_wav_chunks_from_video_builds_segment_command(monkeypatch) -> None:
    """分段模式应使用 segment muxer 并按序返回新生成的分段。"""
    test_dir = TEST_DIR / "chunks"
    output_dir = test_dir / "out"
    test_dir.mkdir(parents=True, exist_ok=True)
    video_path = test_dir / "classroom.mp4"
    video_path.write_bytes(b"fake-video")
    calls = []

    monkeypatch.setattr("edge.extract_audio.shutil.which", lambda name: "ffmpeg")

    def fake_run(command, capture_output, check):
        """模拟 ffmpeg 生成两个符合命名模式的 WAV 分段。"""
        calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "check": check,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "classroom_000.wav").write_bytes(b"chunk-0")
        (output_dir / "classroom_001.wav").write_bytes(b"chunk-1")
        return CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr("edge.extract_audio.subprocess.run", fake_run)

    chunks = extract_wav_chunks_from_video(video_path, output_dir, chunk_seconds=480)

    assert chunks == [
        output_dir / "classroom_000.wav",
        output_dir / "classroom_001.wav",
    ]
    assert calls[0]["command"] == [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "segment",
        "-segment_time",
        "480",
        "-reset_timestamps",
        "1",
        str(output_dir / "classroom_%03d.wav"),
    ]
