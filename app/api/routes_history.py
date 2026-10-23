"""会话历史查询接口。"""

from fastapi import APIRouter, Depends, Query

from app.models.schemas import APIResponse, HistoryResponseData
from app.services.history_service import InMemoryHistoryStore, default_history_store

router = APIRouter(tags=["history"])


def get_history_service() -> InMemoryHistoryStore:
    """注入进程内历史记录仓库。"""
    return default_history_store


@router.get("/api/history", response_model=APIResponse[HistoryResponseData])
async def get_history(
    session_id: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    history_store: InMemoryHistoryStore = Depends(get_history_service),
) -> APIResponse[HistoryResponseData]:
    """按最新优先的顺序返回指定会话的有限条历史记录。"""
    records = history_store.get_recent_records(session_id=session_id, limit=limit)
    return APIResponse(
        code=0,
        message="success",
        data=HistoryResponseData(session_id=session_id, records=records),
    )
