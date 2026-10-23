"""HTTP 输入输出的 Pydantic 模型。

这些模型同时承担参数校验、OpenAPI 文档生成和响应序列化三项职责。
"""

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

DataT = TypeVar("DataT")


class APIResponse(BaseModel, Generic[DataT]):
    """所有 API 端点共用的泛型响应信封。"""
    code: int
    message: str
    data: DataT


class ChatRequest(BaseModel):
    """一次文本问答的输入；字段会在校验前自动去除首尾空白。"""
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class HistoryRecord(BaseModel):
    """供前端展示的一条会话历史记录。"""
    id: str
    session_id: str
    type: str
    summary: str
    detail: str
    behavior: str | None = None
    handRaiseCount: int | None = None
    createdAt: datetime


class HistoryResponseData(BaseModel):
    """历史查询接口的数据载荷。"""
    session_id: str
    records: list[HistoryRecord]


class ChatResponseData(BaseModel):
    """文本问答接口的数据载荷。"""
    answer: str


class STTResponseData(BaseModel):
    """语音转写接口的数据载荷。"""
    text: str
