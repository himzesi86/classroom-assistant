"""聊天 API 的响应契约、上下文存储与参数校验测试。"""

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.context_service import default_context_store


def make_client() -> TestClient:
    """清空全局上下文后创建测试应用，隔离各测试会话。"""
    default_context_store.clear()
    return TestClient(create_app())


def test_health_returns_standard_response() -> None:
    """健康检查应返回项目约定的统一成功响应。"""
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "code": 0,
        "message": "success",
        "data": {"status": "ok"},
    }


def test_chat_api_returns_answer() -> None:
    """mock LLM 模式下，聊天接口也应返回非空 answer。"""
    with make_client() as client:
        response = client.post(
            "/api/chat",
            json={"session_id": "demo", "message": "Explain inertia."},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["answer"]


def test_chat_api_preserves_context_in_memory() -> None:
    """同一 session 的第二次提问应能看到此前用户消息数量。"""
    with make_client() as client:
        first_response = client.post(
            "/api/chat",
            json={"session_id": "demo", "message": "First question"},
        )
        second_response = client.post(
            "/api/chat",
            json={"session_id": "demo", "message": "Follow-up question"},
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert "1 previous user message" in second_response.json()["data"]["answer"]


def test_chat_api_validation_uses_standard_response() -> None:
    """空 session_id/message 的 Pydantic 错误必须使用统一错误信封。"""
    with make_client() as client:
        response = client.post(
            "/api/chat",
            json={"session_id": "", "message": ""},
        )

    body = response.json()
    assert response.status_code == 422
    assert body["code"] == 422
    assert body["message"] == "validation_error"
    assert "errors" in body["data"]
