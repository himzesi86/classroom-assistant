"""应用内部异常类型。

路由层抛出 AppError 后，应用工厂会把它转换为统一的 JSON 响应，
从而避免业务代码直接依赖 FastAPI 的响应对象。
"""

from typing import Any


class AppError(Exception):
    """携带业务错误码、HTTP 状态码及安全可返回详情的异常。"""

    def __init__(
        self,
        code: int,
        message: str,
        status_code: int = 400,
        data: dict[str, Any] | None = None,
    ) -> None:
        """保存异常信息；data 为空时统一使用空字典。"""
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.data = data or {}
