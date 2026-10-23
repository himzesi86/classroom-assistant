"""兼容 OpenAI 风格 chat/completions 接口的 LLM 客户端。

配置始终来自环境变量；未配置密钥时默认进入 mock 模式，便于在没有
外部网络或密钥的课堂演示环境中完成整条接口链路测试。
"""

import logging
import os
from dataclasses import dataclass
from typing import Literal, TypedDict

import httpx


# 默认值只提供服务商兼容接口信息，真实调用仍需要 LLM_API_KEY。
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-reasoner"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0

Role = Literal["system", "user", "assistant"]


class LLMMessage(TypedDict):
    """发送给聊天模型的一条角色消息。"""
    role: Role
    content: str


class LLMClientError(Exception):
    """封装网络、HTTP 状态、响应结构和空回答等 LLM 调用错误。"""

    pass


@dataclass(frozen=True)
class LLMSettings:
    """一次调用所需的、由环境变量解析得到的不可变配置。"""
    api_key: str
    base_url: str
    model: str
    temperature: float | None
    timeout_seconds: float
    mock_enabled: bool


def _read_bool_env(name: str) -> bool | None:
    """将常见真假字符串转换为布尔值；未设置时保留 None。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_float_env(name: str) -> float | None:
    """读取可选浮点环境变量；空字符串按未设置处理。"""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    return float(raw_value)


def load_llm_settings() -> LLMSettings:
    """解析 LLM 环境变量，并决定使用真实服务还是本地 mock。"""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    mock_env = _read_bool_env("LLM_MOCK")
    # 显式 LLM_MOCK 优先；否则无密钥时自动降级，避免演示请求直接失败。
    mock_enabled = mock_env if mock_env is not None else not bool(api_key)

    return LLMSettings(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip()
        or DEFAULT_LLM_BASE_URL,
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL,
        temperature=_read_float_env("LLM_TEMPERATURE"),
        timeout_seconds=float(
            os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS))
        ),
        mock_enabled=mock_enabled,
    )


class LLMClient:
    """负责向模型服务发送请求，或在 mock 模式产生确定性演示回答。"""

    def __init__(self) -> None:
        """创建日志器；不缓存配置，支持运行中调整环境变量后的测试。"""
        self._logger = logging.getLogger("app.llm")

    async def generate(self, messages: list[LLMMessage]) -> str:
        """根据消息序列获得模型回答，并校验最小响应结构。

        真实模式遵循 OpenAI 兼容的 chat/completions 约定；所有底层异常
        都转换为 LLMClientError，使服务层无需了解 httpx 的错误类型。
        """
        settings = load_llm_settings()

        if settings.mock_enabled:
            # Mock 不访问网络，用最近一条用户消息生成可观察的演示文本。
            return self._mock_response(messages)

        if not settings.api_key:
            raise LLMClientError("LLM_API_KEY is required when LLM_MOCK is false.")

        url = f"{settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }
        # messages 由 ChatService 按 system、历史、当前问题的顺序构造。
        payload = {
            "model": settings.model,
            "messages": messages,
        }
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature

        self._logger.info("calling llm model=%s base_url=%s", settings.model, settings.base_url)

        # AsyncClient 仅在本次请求范围内存在，连接完成后会自动关闭。
        try:
            async with httpx.AsyncClient(timeout=settings.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMClientError("LLM request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError("LLM request failed.") from exc

        if response.status_code >= 400:
            raise LLMClientError(f"LLM API returned HTTP {response.status_code}.")

        # 只读取 OpenAI 兼容响应中 choices[0].message.content。
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMClientError("LLM API returned an invalid response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("LLM API returned an empty response.")

        return content.strip()

    def _mock_response(self, messages: list[LLMMessage]) -> str:
        """生成不伪装为真实模型回答的本地课堂演示文本。"""
        user_messages = [
            message["content"]
            for message in messages
            if message["role"] == "user" and message["content"].strip()
        ]
        current_question = user_messages[-1] if user_messages else ""
        # 当前问题也在 user_messages 中，因此减一得到此前提问数量。
        previous_user_count = max(len(user_messages) - 1, 0)

        return (
            f"这是一个课堂演示回答。针对你的问题「{current_question}」，"
            "可以先抓住核心概念，再结合课堂场景来理解。"
            "例如，牛顿第一定律强调物体在不受外力时会保持原来的运动状态，"
            "这可以用教室里静止的课本或匀速滑动的物体来类比。"
            f"当前会话已收到 {previous_user_count} 条学生问题，"
            "如果需要真实模型回答，可以设置 LLM_API_KEY 并将 LLM_MOCK=false。"
        )
