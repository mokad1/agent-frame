"""统一配置入口。

通过 .env 文件和 Pydantic 模型管理全部配置参数，
支持切换模型 Provider、调整记忆参数、注册自定义工具路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()

ProviderKind = Literal["openai", "deepseek", "qwen"]


class Config(BaseModel):
    """全局配置单例。"""

    # ── LLM Provider ─────────────────────────────────
    llm_provider: ProviderKind = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek")
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )

    # ── OpenAI ────────────────────────────────────────
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    # ── DeepSeek ──────────────────────────────────────
    deepseek_api_key: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )

    # ── 通义千问 ──────────────────────────────────────
    qwen_api_key: str = Field(
        default_factory=lambda: os.getenv("QWEN_API_KEY", "")
    )
    qwen_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )

    # ── Agent 参数 ────────────────────────────────────
    agent_max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    )
    agent_temperature: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    )

    # ── 记忆参数 ──────────────────────────────────────
    memory_type: Literal["sliding_window", "vector", "none"] = Field(
        default_factory=lambda: os.getenv("MEMORY_TYPE", "sliding_window")
    )
    memory_window_size: int = Field(
        default_factory=lambda: int(os.getenv("MEMORY_WINDOW_SIZE", "6"))
    )

    # ── Dispatcher 参数 ───────────────────────────────
    dispatcher_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("DISPATCHER_MAX_RETRIES", "2"))
    )

    # ── 日志 ──────────────────────────────────────────
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


config = Config()


def get_config() -> Config:
    """获取全局配置单例。"""
    return config
