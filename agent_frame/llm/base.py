"""LLM Provider 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    """LLM Provider 抽象。

    所有 Provider 必须实现 generate 和 generate_with_tools 方法。
    """

    def __init__(self, model: str, api_key: str, base_url: str, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """纯文本生成。"""
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """带工具调用的生成，返回 {"content": str, "tool_calls": list | None}。

        接收完整的 chat messages 列表（含 system / user / assistant / tool 角色），
        以保留多轮工具调用的完整上下文。
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider 名称。"""
        ...
