"""STT 上传接口成功、空文件和缺失文件的响应契约测试。"""

from fastapi.testclient import TestClient

from app.main import create_app


def make_client() -> TestClient:
    """创建独立的 FastAPI 测试客户端。"""
    return TestClient(create_app())


def test_stt_api_returns_transcript() -> None:
    """默认 mock STT 应返回包含上传文件名的演示转写文本。"""
    with make_client() as client:
        response = client.post(
            "/api/stt",
            files={"file": ("sample.wav", b"fake-audio-bytes", "audio/wav")},
            data={"language": "zh"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 0
    assert body["message"] == "success"
    assert "sample.wav" in body["data"]["text"]


def test_stt_api_rejects_empty_audio() -> None:
    """空音频应被服务层映射为 4000 业务错误。"""
    with make_client() as client:
        response = client.post(
            "/api/stt",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )

    body = response.json()
    assert response.status_code == 400
    assert body["code"] == 4000
    assert body["message"] == "stt_error"
    assert body["data"]["detail"] == "Audio file is empty."


def test_stt_api_missing_file_uses_standard_response() -> None:
    """缺少 multipart 文件时应走统一的 422 校验错误响应。"""
    with make_client() as client:
        response = client.post("/api/stt")

    body = response.json()
    assert response.status_code == 422
    assert body["code"] == 422
    assert body["message"] == "validation_error"
    assert "errors" in body["data"]
