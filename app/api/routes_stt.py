"""语音转文字接口：验证上传内容并映射下游异常。"""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.errors import AppError
from app.integrations.stt_client import STTClientError
from app.models.schemas import APIResponse, STTResponseData
from app.services.stt_service import STTService, STTServiceError, default_stt_service

router = APIRouter(tags=["stt"])


def get_stt_service() -> STTService:
    """注入默认语音转写服务。"""
    return default_stt_service


@router.post("/api/stt", response_model=APIResponse[STTResponseData])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    stt_service: STTService = Depends(get_stt_service),
) -> APIResponse[STTResponseData]:
    """读取 multipart 音频，调用 STT 服务后返回转写文本。"""
    try:
        text = await stt_service.transcribe(
            file=file,
            language=language,
            prompt=prompt,
            session_id=None,
        )
    except STTServiceError as exc:
        raise AppError(
            code=4000,
            message="stt_error",
            status_code=400,
            data={"detail": str(exc)},
        ) from exc
    except STTClientError as exc:
        raise AppError(
            code=5021,
            message="stt_provider_error",
            status_code=502,
            data={"detail": str(exc)},
        ) from exc

    # 文件为空、格式错误等用户侧问题与提供商失败使用不同错误码。
    return APIResponse(
        code=0,
        message="success",
        data=STTResponseData(text=text),
    )
