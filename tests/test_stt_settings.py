"""STT 后端选择与环境变量解析测试。"""

from app.integrations.stt_client import (
    DEFAULT_STT_BACKEND,
    DEFAULT_STT_BASE_URL,
    DEFAULT_STT_LOCAL_COMPUTE_TYPE,
    DEFAULT_STT_LOCAL_DEVICE,
    DEFAULT_STT_LOCAL_MODEL,
    DEFAULT_STT_MAX_AUDIO_BYTES,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_TIMEOUT_SECONDS,
    load_stt_settings,
)


def test_default_stt_settings_are_mocked(monkeypatch) -> None:
    """无环境变量、无密钥时默认使用 mock 后端。"""
    monkeypatch.delenv("STT_BACKEND", raising=False)
    monkeypatch.delenv("STT_API_KEY", raising=False)
    monkeypatch.delenv("STT_BASE_URL", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.delenv("STT_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("STT_LOCAL_DEVICE", raising=False)
    monkeypatch.delenv("STT_LOCAL_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("STT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("STT_MAX_AUDIO_BYTES", raising=False)
    monkeypatch.delenv("STT_MOCK", raising=False)

    settings = load_stt_settings()

    assert settings.backend == DEFAULT_STT_BACKEND
    assert settings.base_url == DEFAULT_STT_BASE_URL
    assert settings.model == DEFAULT_STT_MODEL
    assert settings.local_model == DEFAULT_STT_LOCAL_MODEL
    assert settings.local_device == DEFAULT_STT_LOCAL_DEVICE
    assert settings.local_compute_type == DEFAULT_STT_LOCAL_COMPUTE_TYPE
    assert settings.timeout_seconds == DEFAULT_STT_TIMEOUT_SECONDS
    assert settings.max_audio_bytes == DEFAULT_STT_MAX_AUDIO_BYTES


def test_stt_backend_can_use_local(monkeypatch) -> None:
    """显式 local 配置应保留本地模型、设备和计算精度。"""
    monkeypatch.setenv("STT_BACKEND", "local")
    monkeypatch.setenv("STT_LOCAL_MODEL", "tiny")
    monkeypatch.setenv("STT_LOCAL_DEVICE", "cpu")
    monkeypatch.setenv("STT_LOCAL_COMPUTE_TYPE", "int8")

    settings = load_stt_settings()

    assert settings.backend == "local"
    assert settings.local_model == "tiny"
    assert settings.local_device == "cpu"
    assert settings.local_compute_type == "int8"


def test_legacy_stt_mock_false_selects_api(monkeypatch) -> None:
    """兼容旧 STT_MOCK=false 配置，令其选择 API 后端。"""
    monkeypatch.delenv("STT_BACKEND", raising=False)
    monkeypatch.setenv("STT_MOCK", "false")

    settings = load_stt_settings()

    assert settings.backend == "api"
