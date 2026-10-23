"""课堂问答编排服务。

该层负责将系统提示词、受限会话上下文与当前问题组合为 LLM 消息，
并在成功后同步写入上下文和前端可查询的历史记录。
"""

from app.integrations.llm_client import LLMClient, LLMMessage
from app.services.context_service import InMemoryContextStore, default_context_store
from app.services.history_service import InMemoryHistoryStore, default_history_store
from app.services.prompt_service import load_classroom_qa_prompt


class ChatService:
    """组合上下文仓库、历史仓库与 LLM 客户端完成一轮问答。"""

    def __init__(
        self,
        context_store: InMemoryContextStore,
        history_store: InMemoryHistoryStore,
        llm_client: LLMClient,
        context_limit: int = 10,
    ) -> None:
        """保存协作者与上下文窗口大小，不在构造阶段发起外部请求。"""
        self._context_store = context_store
        self._history_store = history_store
        self._llm_client = llm_client
        self._context_limit = context_limit

    async def answer(self, session_id: str, message: str) -> str:
        """获取最近上下文，调用 LLM，并在成功后持久化本轮消息。

        先读取旧消息再追加本轮输入，避免同一用户问题在请求消息中重复；
        只有 LLM 成功返回后才写入历史，以免记录失败的半轮对话。
        """
        history = self._context_store.get_recent_messages(
            session_id=session_id,
            limit=self._context_limit,
        )
        # system 指令始终位于首位，history 的长度已由 context_limit 限制。
        messages: list[LLMMessage] = [
            {"role": "system", "content": load_classroom_qa_prompt()},
            *history,
            {"role": "user", "content": message},
        ]

        answer = await self._llm_client.generate(messages)

        # 对话上下文与展示历史分开存放，二者承担不同用途。
        self._context_store.add_message(session_id, "user", message)
        self._context_store.add_message(session_id, "assistant", answer)
        self._history_store.add_record(
            session_id=session_id,
            type="qa",
            summary=f"课堂提问：{message}",
            detail=f"学生提问：{message}\n\n课堂助手回答：{answer}",
            behavior="回答",
            handRaiseCount=max(1, (len(message) + 11) // 12),
        )

        return answer


# 生产演示共用的进程内服务实例；测试可通过依赖覆盖传入替代实例。
default_chat_service = ChatService(
    context_store=default_context_store,
    history_store=default_history_store,
    llm_client=LLMClient(),
)
