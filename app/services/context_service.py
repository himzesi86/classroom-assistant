"""有长度上限的进程内多轮上下文仓库。"""

from threading import Lock

from app.integrations.llm_client import LLMMessage, Role


class InMemoryContextStore:
    """按 session_id 保存供 LLM 使用的 role/content 消息序列。"""

    def __init__(self, max_messages_per_session: int = 20) -> None:
        """初始化会话字典和锁；锁保护并发请求下的列表修改。"""
        self._max_messages_per_session = max_messages_per_session
        self._messages: dict[str, list[LLMMessage]] = {}
        self._lock = Lock()

    def add_message(self, session_id: str, role: Role, content: str) -> None:
        """追加一条消息，并从最旧消息开始裁剪到最大窗口。"""
        message: LLMMessage = {"role": role, "content": content}
        with self._lock:
            messages = self._messages.setdefault(session_id, [])
            messages.append(message)
            if len(messages) > self._max_messages_per_session:
                # 删除超出容量的最早消息，保留最近的问答语境。
                del messages[: len(messages) - self._max_messages_per_session]

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[LLMMessage]:
        """返回最近 limit 条浅拷贝，避免调用方修改内部状态。"""
        with self._lock:
            messages = self._messages.get(session_id, [])
            return [message.copy() for message in messages[-limit:]]

    def clear(self) -> None:
        """清空全部会话；主要供测试或应用重置使用。"""
        with self._lock:
            self._messages.clear()


# 默认仓库仅存在于当前进程，服务重启后上下文会丢失。
default_context_store = InMemoryContextStore()
