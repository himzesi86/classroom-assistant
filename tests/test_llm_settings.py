"""LLM 环境变量缺省时的安全演示配置测试。"""

from app.integrations.llm_client import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    load_llm_settings,
)


def test_default_llm_settings_use_deepseek_reasoner(monkeypatch) -> None:
    """无密钥、无覆盖配置时应使用默认模型并自动启用 mock。"""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_MOCK", raising=False)

    settings = load_llm_settings()

    assert settings.base_url == DEFAULT_LLM_BASE_URL
    assert settings.model == DEFAULT_LLM_MODEL
    assert settings.temperature is None
    assert settings.timeout_seconds == DEFAULT_LLM_TIMEOUT_SECONDS
    assert settings.mock_enabled is True
