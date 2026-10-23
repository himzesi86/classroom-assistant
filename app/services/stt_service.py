"""语音转写业务服务：读取上传内容并可选写入会话历史。"""

from fastapi import UploadFile

from app.integrations.stt_client import STTClient
from app.services.history_service import InMemoryHistoryStore, default_history_store


class STTServiceError(Exception):
    """代表上传内容等业务层可预期错误。"""

    pass


class STTService:
    """在 HTTP 上传对象与底层 STT 客户端之间做最小编排。"""

    def __init__(self, stt_client: STTClient, history_store: InMemoryHistoryStore) -> None:
        """保存可替换的客户端和历史仓库，便于测试注入。"""
        self._stt_client = stt_client
        self._history_store = history_store

    async def transcribe(
        self,
        *,
        file: UploadFile,
        language: str | None = None,
        prompt: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """读取音频字节、调用后端转写，并在提供会话时记录结果。"""
        filename = file.filename or "audio"
        content = await file.read()
        if not content:
            raise STTServiceError("Audio file is empty.")

        text = await self._stt_client.transcribe(
            filename=filename,
            content=content,
            content_type=file.content_type or "application/octet-stream",
            language=language,
            prompt=prompt,
        )
        if session_id:
            # /api/stt 当前未传 session_id；边缘端复用此服务时可开启记录。
            self._history_store.add_record(
                session_id=session_id,
                type="stt",
                summary=f"语音转写：{text[:80]}",
                detail=f"识别文本：{text}",
                behavior="回答",
                handRaiseCount=1,
            )
        return text


# 默认实例复用同一个历史仓库，使语音记录可被历史接口查询。
default_stt_service = STTService(stt_client=STTClient(), history_store=default_history_store)
