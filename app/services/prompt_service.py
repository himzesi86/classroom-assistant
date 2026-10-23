"""集中加载课堂助手系统提示词，避免在路由或服务中硬编码。"""

from functools import lru_cache
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
CLASSROOM_QA_PROMPT = PROMPT_DIR / "classroom_qa_system.txt"


@lru_cache(maxsize=1)
def load_classroom_qa_prompt() -> str:
    """读取并缓存系统提示词；进程生命周期内最多读取磁盘一次。"""
    return CLASSROOM_QA_PROMPT.read_text(encoding="utf-8").strip()
