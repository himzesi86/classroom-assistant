"""文本问答接口：将 HTTP 请求委派给 ChatService。"""

from fastapi import APIRouter, Depends

from app.errors import AppError
from app.integrations.llm_client import LLMClientError
from app.models.schemas import APIResponse, ChatRequest, ChatResponseData
from app.services.chat_service import ChatService, default_chat_service

# 标签仅用于 OpenAPI/Swagger 中对接口分组，不改变 URL。
router = APIRouter(tags=["chat"])


def get_chat_service() -> ChatService:
    """为 FastAPI 依赖注入返回默认对话服务实例。"""
    return default_chat_service


@router.post("/api/chat", response_model=APIResponse[ChatResponseData])
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> APIResponse[ChatResponseData]:
    """处理一轮提问，并将 LLM 调用失败映射为网关错误。

    路由不拼接 Prompt、不管理上下文，避免 HTTP 层与模型逻辑耦合。
    """
    try:
        answer = await chat_service.answer(payload.session_id, payload.message)
    except LLMClientError as exc:
        raise AppError(
            code=5020,
            message="llm_error",
            status_code=502,
            data={"detail": str(exc)},
        ) from exc

    # Pydantic 会按 response_model 序列化为项目约定的统一响应格式。
    return APIResponse(
        code=0,
        message="success",
        data=ChatResponseData(answer=answer),
    )
