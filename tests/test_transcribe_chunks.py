"""分段 Whisper 转写的汇总输出、空目录和粗转写测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from edge.transcribe_chunks import (
    ChunkTranscriptionError,
    transcribe_wav_chunks,
)


TEST_DIR = Path("tmp/test_transcribe_chunks")


def write_fake_wav(path: Path) -> None:
    """写入占位 WAV 字节；测试中的假模型不会解析真实音频格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-wav")


def test_transcribe_wav_chunks_writes_text_and_jsonl(monkeypatch) -> None:
    """最终模型应按文件顺序输出合并文本与逐行 JSON 记录。"""
    input_dir = TEST_DIR / "chunks"
    output_path = TEST_DIR / "transcript.txt"
    input_dir.mkdir(parents=True, exist_ok=True)
    write_fake_wav(input_dir / "chunk_001.wav")
    write_fake_wav(input_dir / "chunk_002.wav")

    class FakeModel:
        """替代 WhisperModel 的可预测假实现。"""

        def __init__(self, model_name, device, compute_type):
            """记录构造参数，供假模型转写方法使用。"""
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type

        def transcribe(self, path, language, vad_filter):
            """根据文件名返回确定性文本与固定时长。"""
            stem = Path(path).stem
            segments = [SimpleNamespace(text=f"text for {stem}")]
            info = SimpleNamespace(duration=123.0)
            return segments, info

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeModel),
    )

    results = transcribe_wav_chunks(input_dir, output_path, model_name="tiny")

    assert [result.text for result in results] == [
        "text for chunk_001",
        "text for chunk_002",
    ]
    assert "text for chunk_001" in output_path.read_text(encoding="utf-8")
    assert output_path.with_suffix(".jsonl").exists()


def test_transcribe_wav_chunks_rejects_empty_directory() -> None:
    """没有 WAV 分段的目录应在模型加载前被拒绝。"""
    input_dir = TEST_DIR / "empty"
    input_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ChunkTranscriptionError, match="No WAV files"):
        transcribe_wav_chunks(input_dir, TEST_DIR / "empty.txt")


def test_transcribe_wav_chunks_writes_rough_output(monkeypatch) -> None:
    """开启粗模型时，最终与粗转写应分别写入对应输出。"""
    input_dir = TEST_DIR / "rough"
    output_path = TEST_DIR / "rough_final.txt"
    rough_path = TEST_DIR / "rough_preview.txt"
    write_fake_wav(input_dir / "chunk_001.wav")

    class FakeModel:
        """根据 model_name 返回不同文本的假 Whisper 实现。"""

        def __init__(self, model_name, device, compute_type):
            """只保存模型名，用于区分粗模型与最终模型输出。"""
            self.model_name = model_name

        def transcribe(self, path, language, vad_filter):
            """返回携带模型名的单片段转写结果。"""
            segments = [SimpleNamespace(text=f"{self.model_name} text")]
            info = SimpleNamespace(duration=123.0)
            return segments, info

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeModel),
    )

    results = transcribe_wav_chunks(
        input_dir,
        output_path,
        model_name="base",
        rough_model_name="tiny",
        rough_output_path=rough_path,
    )

    assert results[0].text == "base text"
    assert results[0].rough_text == "tiny text"
    assert "tiny text" in rough_path.read_text(encoding="utf-8")
