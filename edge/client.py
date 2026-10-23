"""边缘侧 HTTP 客户端。

该模块把文字或本地音频发送给云端 API；边缘端只负责采集和预处理，
所有 LLM 与 STT 推理仍由云端服务配置决定。
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from edge.config import EdgeSettings, load_edge_settings
from edge.audio_capture import AudioCaptureError, record_wav
from edge.preprocess import AudioPreprocessError, preprocess_wav


class EdgeClientError(Exception):
    """表示本地输入、云端网络、响应格式或业务错误。"""

    pass


class EdgeClient:
    """封装课堂助手云端 API 的文字提问、STT 上传和响应校验。"""

    def __init__(self, settings: EdgeSettings) -> None:
        """保存不可变边缘配置，并创建专属日志器。"""
        self._settings = settings
        self._logger = logging.getLogger("edge.client")

    async def ask_with_text(self, message: str, session_id: str | None = None) -> str:
        """调用云端聊天接口并返回 data.answer 字段。"""
        payload = {
            "session_id": session_id or self._settings.session_id,
            "message": message,
        }
        body = await self._post_json("/api/chat", payload)
        return _read_data_field(body, "answer")

    async def ask_with_audio_file(
        self,
        audio_path: Path,
        session_id: str | None = None,
        language: str | None = None,
    ) -> dict[str, str]:
        """读取音频文件；WAV 先本地规范化，再执行 STT 后接续文本问答。"""
        audio_bytes = audio_path.read_bytes()
        if audio_path.suffix.lower() == ".wav":
            # 只对 WAV 做内置处理；其他格式原样交由云端后端判断。
            audio_bytes = preprocess_wav(
                audio_bytes,
                target_sample_rate=self._settings.audio_sample_rate,
                target_channels=self._settings.audio_channels,
                max_bytes=self._settings.max_audio_bytes,
            )

        transcript = await self.transcribe_audio(
            filename=audio_path.name,
            content=audio_bytes,
            language=language or self._settings.language,
        )
        answer = await self.ask_with_text(transcript, session_id=session_id)
        return {"text": transcript, "answer": answer}

    async def transcribe_audio(
        self,
        *,
        filename: str,
        content: bytes,
        language: str,
    ) -> str:
        """以 multipart 请求上传音频，并从统一响应读取转写文本。"""
        if not content:
            raise EdgeClientError("Audio file is empty.")
        if len(content) > self._settings.max_audio_bytes:
            raise EdgeClientError(
                f"Audio exceeds max edge upload size: {self._settings.max_audio_bytes} bytes."
            )

        url = _join_url(self._settings.cloud_base_url, "/api/stt")
        # 当前边缘客户端声明 WAV MIME；传输协议本身仍由文件字节承载。
        files = {"file": (filename, content, "audio/wav")}
        data = {"language": language}
        self._logger.info("uploading audio filename=%s size=%s", filename, len(content))

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.request_timeout_seconds
            ) as client:
                response = await client.post(url, files=files, data=data)
        except httpx.TimeoutException as exc:
            raise EdgeClientError("Cloud STT request timed out.") from exc
        except httpx.HTTPError as exc:
            raise EdgeClientError("Cloud STT request failed.") from exc

        body = _parse_api_response(response)
        return _read_data_field(body, "text")

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向云端发送 JSON 请求，并复用标准响应校验。"""
        url = _join_url(self._settings.cloud_base_url, path)
        self._logger.info("posting json path=%s", path)

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.request_timeout_seconds
            ) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise EdgeClientError("Cloud request timed out.") from exc
        except httpx.HTTPError as exc:
            raise EdgeClientError("Cloud request failed.") from exc

        return _parse_api_response(response)


def _join_url(base_url: str, path: str) -> str:
    """拼接 URL 时消除重复或缺失的斜杠。"""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _parse_api_response(response: httpx.Response) -> dict[str, Any]:
    """校验 HTTP 成功、业务 code 和 data 对象三层响应契约。"""
    try:
        body = response.json()
    except ValueError as exc:
        raise EdgeClientError("Cloud API returned invalid JSON.") from exc

    if response.status_code >= 400:
        raise EdgeClientError(f"Cloud API returned HTTP {response.status_code}: {body}")
    if body.get("code") != 0:
        raise EdgeClientError(f"Cloud API returned error response: {body}")
    if not isinstance(body.get("data"), dict):
        raise EdgeClientError("Cloud API returned invalid data.")
    return body


def _read_data_field(body: dict[str, Any], field_name: str) -> str:
    """读取非空字符串数据字段，缺失时抛出明确的边缘客户端异常。"""
    value = body["data"].get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EdgeClientError(f"Cloud API response is missing data.{field_name}.")
    return value.strip()


async def _run(args: argparse.Namespace) -> None:
    """根据 CLI 三种输入模式执行一次边缘端问答流程。"""
    settings = load_edge_settings()
    client = EdgeClient(settings)

    if args.text:
        answer = await client.ask_with_text(args.text, session_id=args.session_id)
        print(answer)
        return

    if args.audio:
        result = await client.ask_with_audio_file(
            Path(args.audio),
            session_id=args.session_id,
            language=args.language,
        )
        print(f"识别文本: {result['text']}")
        print(f"模型回答: {result['answer']}")
        return

    if args.record:
        # 录音完成后复用文件上传流程，保持采集与网络协议解耦。
        audio_path = record_wav(
            Path(args.record),
            seconds=settings.audio_seconds,
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
        )
        result = await client.ask_with_audio_file(
            audio_path,
            session_id=args.session_id,
            language=args.language,
        )
        print(f"录音文件: {audio_path}")
        print(f"识别文本: {result['text']}")
        print(f"模型回答: {result['answer']}")
        return

    raise EdgeClientError("Use --text or --audio to send an edge request.")


def build_parser() -> argparse.ArgumentParser:
    """定义文字、文件音频和麦克风录音三种边缘端输入方式。"""
    parser = argparse.ArgumentParser(
        description="Edge-side collector for direction C classroom Q&A."
    )
    parser.add_argument("--text", help="Send a typed question to the cloud chat API.")
    parser.add_argument("--audio", help="Upload a local WAV audio file through STT, then chat.")
    parser.add_argument("--record", help="Record microphone audio to this WAV path, then upload.")
    parser.add_argument("--session-id", help="Conversation session id.")
    parser.add_argument("--language", help="STT language code, for example zh or en.")
    return parser


def main() -> None:
    """初始化日志、运行异步流程，并将预期错误转为 CLI 退出信息。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        asyncio.run(_run(args))
    except (AudioCaptureError, AudioPreprocessError, EdgeClientError) as exc:
        raise SystemExit(f"edge error: {exc}") from exc


if __name__ == "__main__":
    main()
