"""多后端语音转写客户端。

同一接口可切换 mock、云端 API 和本地 faster-whisper；服务层只关心
输入音频字节和最终文本，不依赖具体转写提供商。
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx


# 三种后端决定 transcribe 方法的执行分支。
STTBackend = Literal["mock", "api", "local"]

DEFAULT_STT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_STT_MODEL = "whisper-1"
DEFAULT_STT_TIMEOUT_SECONDS = 60.0
DEFAULT_STT_MAX_AUDIO_BYTES = 25 * 1024 * 1024
DEFAULT_STT_BACKEND: STTBackend = "mock"
DEFAULT_STT_LOCAL_MODEL = "base"
DEFAULT_STT_LOCAL_DEVICE = "cpu"
DEFAULT_STT_LOCAL_COMPUTE_TYPE = "int8"


class STTClientError(Exception):
    """统一表达 STT 配置、上传、网络、解析和本地推理错误。"""

    pass


@dataclass(frozen=True)
class STTSettings:
    """从环境变量读取并冻结的一次 STT 调用配置。"""
    backend: STTBackend
    api_key: str
    base_url: str
    model: str
    local_model: str
    local_device: str
    local_compute_type: str
    timeout_seconds: float
    max_audio_bytes: int


def _read_bool_env(name: str) -> bool | None:
    """解析兼容旧配置 STT_MOCK 的布尔环境变量。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_stt_backend(api_key: str) -> STTBackend:
    """按 STT_BACKEND、STT_MOCK、是否存在密钥的优先级选择后端。"""
    raw_backend = os.getenv("STT_BACKEND", "").strip().lower()
    if raw_backend:
        if raw_backend not in {"mock", "api", "local"}:
            raise STTClientError("STT_BACKEND must be one of: mock, api, local.")
        # 经过集合校验后，字符串可安全视为 STTBackend。
        return raw_backend  # type: ignore[return-value]

    mock_env = _read_bool_env("STT_MOCK")
    if mock_env is not None:
        return "mock" if mock_env else "api"

    return "api" if api_key else DEFAULT_STT_BACKEND


def load_stt_settings() -> STTSettings:
    """读取 API、本地模型、超时和上传大小限制等 STT 配置。"""
    api_key = os.getenv("STT_API_KEY", "").strip()

    return STTSettings(
        backend=_resolve_stt_backend(api_key),
        api_key=api_key,
        base_url=os.getenv("STT_BASE_URL", DEFAULT_STT_BASE_URL).strip()
        or DEFAULT_STT_BASE_URL,
        model=os.getenv("STT_MODEL", DEFAULT_STT_MODEL).strip() or DEFAULT_STT_MODEL,
        local_model=os.getenv("STT_LOCAL_MODEL", DEFAULT_STT_LOCAL_MODEL).strip()
        or DEFAULT_STT_LOCAL_MODEL,
        local_device=os.getenv("STT_LOCAL_DEVICE", DEFAULT_STT_LOCAL_DEVICE).strip()
        or DEFAULT_STT_LOCAL_DEVICE,
        local_compute_type=os.getenv(
            "STT_LOCAL_COMPUTE_TYPE",
            DEFAULT_STT_LOCAL_COMPUTE_TYPE,
        ).strip()
        or DEFAULT_STT_LOCAL_COMPUTE_TYPE,
        timeout_seconds=float(
            os.getenv("STT_TIMEOUT_SECONDS", str(DEFAULT_STT_TIMEOUT_SECONDS))
        ),
        max_audio_bytes=int(
            os.getenv("STT_MAX_AUDIO_BYTES", str(DEFAULT_STT_MAX_AUDIO_BYTES))
        ),
    )


@lru_cache(maxsize=2)
def load_local_whisper_model(model_name: str, device: str, compute_type: str):
    """按模型、设备和精度缓存 local Whisper，避免每次请求重复加载权重。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTClientError(
            "Local STT requires faster-whisper. Install it with "
            "`python3 -m pip install -r requirements-offline-stt.txt`."
        ) from exc

    return WhisperModel(model_name, device=device, compute_type=compute_type)


class STTClient:
    """向 mock、云端 API 或本地 Whisper 转发统一的音频转写请求。"""

    def __init__(self) -> None:
        """创建 STT 专用日志器，不在此处加载模型。"""
        self._logger = logging.getLogger("app.stt")

    async def transcribe(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """检查上传大小后，按当前后端配置分派转写任务。"""
        settings = load_stt_settings()

        if len(content) > settings.max_audio_bytes:
            raise STTClientError(
                f"Audio file exceeds max size of {settings.max_audio_bytes} bytes."
            )

        if settings.backend == "mock":
            # mock 仅回显文件元数据，明确标注它不是实际语音识别。
            return self._mock_transcript(filename, content)
        if settings.backend == "local":
            return self._transcribe_local(
                settings=settings,
                filename=filename,
                content=content,
                language=language,
                prompt=prompt,
            )

        return await self._transcribe_api(
            settings=settings,
            filename=filename,
            content=content,
            content_type=content_type,
            language=language,
            prompt=prompt,
        )

    async def _transcribe_api(
        self,
        *,
        settings: STTSettings,
        filename: str,
        content: bytes,
        content_type: str,
        language: str | None,
        prompt: str | None,
    ) -> str:
        """以 multipart/form-data 调用兼容的 /audio/transcriptions 接口。"""
        if not settings.api_key:
            raise STTClientError("STT_API_KEY is required when STT_BACKEND=api.")

        url = f"{settings.base_url.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.api_key}"}
        # language 与 prompt 均是可选提示，不传时由提供商自动处理。
        data = {"model": settings.model}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        files = {
            "file": (
                filename,
                content,
                content_type or "application/octet-stream",
            )
        }

        self._logger.info(
            "calling stt model=%s base_url=%s filename=%s size=%s",
            settings.model,
            settings.base_url,
            filename,
            len(content),
        )

        # 文件内容直接作为 multipart 字节发送，不写入服务端临时文件。
        try:
            async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
                response = await client.post(
                    url,
                    data=data,
                    files=files,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise STTClientError("STT request timed out.") from exc
        except httpx.HTTPError as exc:
            raise STTClientError("STT request failed.") from exc

        if response.status_code >= 400:
            raise STTClientError(f"STT API returned HTTP {response.status_code}.")

        try:
            body = response.json()
            text = body["text"]
        except (KeyError, TypeError, ValueError) as exc:
            raise STTClientError("STT API returned an invalid response.") from exc

        if not isinstance(text, str) or not text.strip():
            raise STTClientError("STT API returned an empty transcript.")

        return text.strip()

    def _transcribe_local(
        self,
        *,
        settings: STTSettings,
        filename: str,
        content: bytes,
        language: str | None,
        prompt: str | None,
    ) -> str:
        """将上传字节暂存为文件，交给 faster-whisper 后立即清理。"""
        self._logger.info(
            "calling local stt model=%s device=%s filename=%s size=%s",
            settings.local_model,
            settings.local_device,
            filename,
            len(content),
        )

        suffix = Path(filename).suffix or ".audio"
        temp_path = ""
        try:
            # Whisper 接口接收路径，因此使用 delete=False 保证 Windows 可再次打开。
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as audio_file:
                audio_file.write(content)
                temp_path = audio_file.name

            model = load_local_whisper_model(
                settings.local_model,
                settings.local_device,
                settings.local_compute_type,
            )
            # segments 是惰性迭代器；join 时才完成全部片段的解码。
            segments, _info = model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
            )
            text = "".join(segment.text for segment in segments).strip()
        except STTClientError:
            raise
        except Exception as exc:
            raise STTClientError("Local STT transcription failed.") from exc
        finally:
            # 无论模型或解码失败，都不得遗留包含用户音频的临时文件。
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

        if not text:
            raise STTClientError("Local STT returned an empty transcript.")

        return text

    def _mock_transcript(self, filename: str, content: bytes) -> str:
        """返回带文件名和大小的明确 mock 文本，供接口联调与测试使用。"""
        return (
            f"Mock STT transcript for {filename} ({len(content)} bytes). "
            "This is not real speech recognition. Set STT_BACKEND=local "
            "for offline transcription or STT_BACKEND=api for a provider API."
        )
