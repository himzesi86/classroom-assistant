"""供历史接口展示的进程内记录仓库。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from app.models.schemas import HistoryRecord


@dataclass
class HistoryItem:
    """内部存储格式；字段命名与前端约定保持一致。"""
    id: str
    session_id: str
    type: str
    summary: str
    detail: str
    behavior: str | None = None
    handRaiseCount: int | None = None
    createdAt: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self) -> HistoryRecord:
        """将 dataclass 转换为经过 Pydantic 校验的 API 模型。"""
        payload = asdict(self)
        return HistoryRecord(**payload)


class InMemoryHistoryStore:
    """按会话保存最新优先、数量受限的历史记录。"""

    def __init__(self, max_records_per_session: int = 50) -> None:
        """初始化历史字典；并发读写由 Lock 串行化。"""
        self._max_records_per_session = max_records_per_session
        self._records: dict[str, list[HistoryItem]] = {}
        self._lock = Lock()

    def add_record(
        self,
        *,
        session_id: str,
        type: str,
        summary: str,
        detail: str,
        behavior: str | None = None,
        handRaiseCount: int | None = None,
    ) -> HistoryRecord:
        """创建 UUID 记录、插入队首，并裁剪过旧记录。"""
        item = HistoryItem(
            id=str(uuid4()),
            session_id=session_id,
            type=type,
            summary=summary,
            detail=detail,
            behavior=behavior,
            handRaiseCount=handRaiseCount,
            createdAt=datetime.now(timezone.utc),
        )
        with self._lock:
            records = self._records.setdefault(session_id, [])
            # 新记录放在索引 0，使 API 默认按时间倒序返回。
            records.insert(0, item)
            del records[self._max_records_per_session :]
        return item.to_record()

    def get_recent_records(self, session_id: str, limit: int = 10) -> list[HistoryRecord]:
        """返回会话中最近的 limit 条记录副本。"""
        with self._lock:
            records = self._records.get(session_id, [])[:limit]
            return [record.to_record() for record in records]

    def clear(self, session_id: str | None = None) -> None:
        """清空指定会话；未指定时清空全部历史。"""
        with self._lock:
            if session_id is None:
                self._records.clear()
            else:
                self._records.pop(session_id, None)


# 与上下文仓库相同，该默认实例不提供跨进程持久化。
default_history_store = InMemoryHistoryStore()
