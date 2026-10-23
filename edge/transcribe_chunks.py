"""对已分段 WAV 执行本地 Whisper 转写并汇总文本与 JSONL 元数据。"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class ChunkTranscriptionError(Exception):
    """表示模型依赖、输入目录或分段转写流程错误。"""

    pass


@dataclass(frozen=True)
class ChunkTranscript:
    """单个音频分段的最终文本、可选粗转写、时长和耗时。"""
    path: str
    text: str
    rough_text: str | None
    duration_seconds: float
    elapsed_seconds: float


def transcribe_wav_chunks(
    input_dir: Path,
    output_text_path: Path,
    *,
    model_name: str = "base",
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
    rough_model_name: str | None = None,
    rough_output_path: Path | None = None,
) -> list[ChunkTranscript]:
    """依次转写输入目录中的 WAV，并写出合并文本和逐行 JSON 结果。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ChunkTranscriptionError(
            "faster-whisper is required. Install requirements-offline-stt.txt first."
        ) from exc

    if not input_dir.exists():
        raise ChunkTranscriptionError(f"Input directory does not exist: {input_dir}")

    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        raise ChunkTranscriptionError(f"No WAV files found in: {input_dir}")

    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_text_path.with_suffix(".jsonl")
    # 最终模型只加载一次，避免每个音频分段重复初始化权重。
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    rough_model = None
    if rough_model_name:
        rough_model = WhisperModel(rough_model_name, device=device, compute_type=compute_type)
        rough_output_path = rough_output_path or output_text_path.with_name(
            f"{output_text_path.stem}_rough.txt"
        )
        rough_output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[ChunkTranscript] = []
    # 粗转写为可选双模型流程；未指定时不会创建额外文件。
    rough_file = (
        rough_output_path.open("w", encoding="utf-8") if rough_output_path else None
    )
    with output_text_path.open("w", encoding="utf-8") as text_file, jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as jsonl_file:
        for index, wav_path in enumerate(wav_paths, start=1):
            # 文件名排序决定合并文本的时间顺序，要求分段命名可排序。
            rough_text = None
            if rough_model is not None:
                rough_segments, _rough_info = rough_model.transcribe(
                    str(wav_path),
                    language=language,
                    vad_filter=True,
                )
                rough_text = "".join(segment.text for segment in rough_segments).strip()
                if rough_file is not None:
                    rough_file.write(f"[chunk {index:03d}] {wav_path.name}\n")
                    rough_file.write(rough_text)
                    rough_file.write("\n\n")

            # 计时仅覆盖最终模型对本分段的转写，便于分析性能。
            started_at = time.perf_counter()
            segments, info = model.transcribe(
                str(wav_path),
                language=language,
                vad_filter=True,
            )
            text = "".join(segment.text for segment in segments).strip()
            elapsed_seconds = time.perf_counter() - started_at
            transcript = ChunkTranscript(
                path=str(wav_path),
                text=text,
                rough_text=rough_text,
                duration_seconds=float(info.duration),
                elapsed_seconds=elapsed_seconds,
            )
            results.append(transcript)

            text_file.write(f"[chunk {index:03d}] {wav_path.name}\n")
            text_file.write(text)
            text_file.write("\n\n")
            jsonl_file.write(json.dumps(asdict(transcript), ensure_ascii=False) + "\n")

    if rough_file is not None:
        # rough_file 在 with 块外打开，因此在所有分段完成后显式关闭。
        rough_file.close()

    return results


def build_parser() -> argparse.ArgumentParser:
    """定义本地分段转写的模型、设备、语言和输出参数。"""
    parser = argparse.ArgumentParser(description="Transcribe WAV chunks into text.")
    parser.add_argument("input_dir", help="Directory containing WAV chunks.")
    parser.add_argument(
        "-o",
        "--output",
        default="tmp/transcript.txt",
        help="Merged transcript text path.",
    )
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--rough-model",
        help="Run a fast local rough transcription model before the final model.",
    )
    parser.add_argument(
        "--rough-output",
        help="Output path for rough transcript. Defaults to OUTPUT stem + _rough.txt.",
    )
    return parser


def main() -> None:
    """运行 CLI 转写并打印每段的时长和实际耗时。"""
    args = build_parser().parse_args()
    try:
        results = transcribe_wav_chunks(
            Path(args.input_dir),
            Path(args.output),
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            rough_model_name=args.rough_model,
            rough_output_path=Path(args.rough_output) if args.rough_output else None,
        )
    except ChunkTranscriptionError as exc:
        raise SystemExit(f"chunk transcription error: {exc}") from exc

    for result in results:
        print(
            f"{result.path}\t"
            f"duration={result.duration_seconds:.2f}s\t"
            f"elapsed={result.elapsed_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
