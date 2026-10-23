"""边缘端运行配置。

边缘模块只采集、预处理并调用云端 HTTP API；不包含 LLM 推理逻辑。
"""

import os
from dataclasses import dataclass


DEFAULT_CLOUD_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SESSION_ID = "edge-demo"
DEFAULT_LANGUAGE = "zh"
DEFAULT_AUDIO_SECONDS = 5.0
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_MAX_AUDIO_BYTES = 20 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class EdgeSettings:
    """边缘客户端所需的云端地址、音频参数和请求限制。"""
    cloud_base_url: str
    session_id: str
    language: str
    audio_seconds: float
    audio_sample_rate: int
    audio_channels: int
    max_audio_bytes: int
    request_timeout_seconds: float


def load_edge_settings() -> EdgeSettings:
    """从环境变量加载配置；空字符串回退为安全的本地演示默认值。"""
    return EdgeSettings(
        cloud_base_url=os.getenv("EDGE_CLOUD_BASE_URL", DEFAULT_CLOUD_BASE_URL).strip()
        or DEFAULT_CLOUD_BASE_URL,
        session_id=os.getenv("EDGE_SESSION_ID", DEFAULT_SESSION_ID).strip()
        or DEFAULT_SESSION_ID,
        language=os.getenv("EDGE_LANGUAGE", DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE,
        audio_seconds=float(os.getenv("EDGE_AUDIO_SECONDS", str(DEFAULT_AUDIO_SECONDS))),
        audio_sample_rate=int(
            os.getenv("EDGE_AUDIO_SAMPLE_RATE", str(DEFAULT_AUDIO_SAMPLE_RATE))
        ),
        audio_channels=int(os.getenv("EDGE_AUDIO_CHANNELS", str(DEFAULT_AUDIO_CHANNELS))),
        max_audio_bytes=int(os.getenv("EDGE_MAX_AUDIO_BYTES", str(DEFAULT_MAX_AUDIO_BYTES))),
        request_timeout_seconds=float(
            os.getenv(
                "EDGE_REQUEST_TIMEOUT_SECONDS",
                str(DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
        ),
    )
