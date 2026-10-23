"""课堂视频抽帧工具。

该模块只负责为后续视觉扩展保存压缩帧，当前课堂问答主链路不会上传或
分析这些帧。
"""

from dataclasses import dataclass
from pathlib import Path


class VideoCaptureError(Exception):
    """表示 OpenCV 依赖缺失或视频源无法打开。"""

    pass


@dataclass(frozen=True)
class CapturedFrame:
    """一帧落盘文件及其在原视频中的位置元数据。"""
    path: Path
    index: int
    timestamp_seconds: float


def sample_video_frames(
    source: str,
    output_dir: Path,
    *,
    interval_seconds: float = 2.0,
    jpeg_quality: int = 70,
    max_width: int = 960,
) -> list[CapturedFrame]:
    """按时间间隔抽取视频帧，并在必要时等比例缩小后保存为 JPEG。"""
    try:
        import cv2
    except ImportError as exc:
        raise VideoCaptureError(
            "Video capture requires opencv-python. Install it on the edge VM if needed."
        ) from exc

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise VideoCaptureError(f"Cannot open video source: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 缺失 FPS 时使用常见的 25 FPS，确保仍能得到近似时间间隔。
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(fps * interval_seconds), 1)

    frames: list[CapturedFrame] = []
    index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if index % step == 0:
            # 仅处理采样点，避免把整段视频全部写入磁盘。
            height, width = frame.shape[:2]
            if width > max_width:
                # 用宽度比例缩放，保持原始画面宽高比。
                scale = max_width / width
                frame = cv2.resize(frame, (max_width, int(height * scale)))

            path = output_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frames.append(
                CapturedFrame(
                    path=path,
                    index=index,
                    timestamp_seconds=index / fps,
                )
            )
            saved += 1

        index += 1

    # 显式释放摄像头/文件句柄，避免边缘设备占用视频源。
    capture.release()
    return frames
