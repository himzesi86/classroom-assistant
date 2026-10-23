"""课堂助手云端 API 的应用工厂与全局异常处理。

本模块只装配 FastAPI：注册路由、中间件和统一响应格式；对话与 STT
的具体业务逻辑分别保留在 services 和 integrations 层。
"""

import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes_chat import router as chat_router
from app.api.routes_history import router as history_router
from app.api.routes_stt import router as stt_router
from app.errors import AppError


def api_response(code: int, message: str, data: Any) -> dict[str, Any]:
    """构造所有接口共用的 code/message/data 三段式响应。"""
    return {"code": code, "message": message, "data": data}


def create_app() -> FastAPI:
    """创建可测试、可重复实例化的 FastAPI 应用。

    路由测试调用此函数获得隔离实例；模块末尾的 app 则供 Uvicorn
    作为默认 ASGI 入口使用。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(title="Classroom Assistant API")
    app.add_middleware(
        CORSMiddleware,
        # 演示环境允许跨域访问；生产环境应替换为明确的前端域名列表。
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger = logging.getLogger("app.http")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """记录每个请求的开始、结束状态码和总耗时。"""
        started_at = time.perf_counter()
        logger.info("incoming request method=%s path=%s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "request failed method=%s path=%s elapsed_ms=%.2f",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "request completed method=%s path=%s status=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """把已知业务异常转换为约定的错误响应。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=api_response(exc.code, exc.message, exc.data),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """将 Pydantic 参数校验失败统一包装为 422 响应。"""
        return JSONResponse(
            status_code=422,
            content=api_response(
                422,
                "validation_error",
                {"errors": jsonable_encoder(exc.errors())},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        """统一处理路由不存在等 Starlette HTTP 异常。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=api_response(exc.status_code, str(exc.detail), {}),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        """记录未预期异常，但不向调用方泄露内部堆栈。"""
        logging.getLogger("app").exception("unexpected error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=api_response(500, "internal_server_error", {}),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """提供轻量健康检查，不调用 LLM 或 STT 服务。"""
        return api_response(0, "success", {"status": "ok"})

    app.include_router(chat_router)
    app.include_router(stt_router)
    app.include_router(history_router)
    return app


# 给 ASGI 服务器使用的默认应用对象。
app = create_app()
