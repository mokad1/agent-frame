"""三款 LLM Provider 实现。

所有 Provider 统一 Function Calling 格式：
- 内部使用标准 OpenAI tool_choice/tools 格式
- DeepSeek / Qwen 的格式差异在 Provider 层内部转换
- 上层 Agent 代码无需关心调用的是哪家模型
"""

from __future__ import annotations

from typing import Any

import httpx

from agent_frame.config import config
from agent_frame.llm.base import BaseProvider
from agent_frame.utils.logger import get_logger

logger = get_logger("llm.providers")


class OpenAICompatibleProvider(BaseProvider):
    """通用 OpenAI-compatible Provider。

    自动处理 Function Calling 的工具调用响应格式。
    """

    def __init__(self, model: str, api_key: str, base_url: str, name: str = "openai") -> None:
        super().__init__(model, api_key, base_url)
        self._name = name
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return self._name

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        client = await self._get_client()
        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """带工具调用的生成。

        接收完整的 chat messages 列表，保留多轮工具调用上下文。
        适配三家不同的 Function Calling 格式：
        - OpenAI: tools + tool_choice
        - DeepSeek: tools (OpenAI 兼容)
        - Qwen: tools（通过 compatible-mode 兼容）
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        client = await self._get_client()
        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})

        result: dict[str, Any] = {
            "content": message.get("content", ""),
            "tool_calls": None,
            "finish_reason": choice.get("finish_reason", "stop"),
        }

        # 统一解析 tool_calls
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            parsed: list[dict[str, Any]] = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                parsed.append({
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                })
            result["tool_calls"] = parsed

        return result

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self, model: str = "", api_key: str = "", base_url: str = "") -> None:
        super().__init__(
            model=model or config.llm_model or "gpt-4o",
            api_key=api_key or config.openai_api_key,
            base_url=base_url or config.openai_base_url,
            name="openai",
        )


class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self, model: str = "", api_key: str = "", base_url: str = "") -> None:
        super().__init__(
            model=model or config.llm_model or "deepseek-chat",
            api_key=api_key or config.deepseek_api_key,
            base_url=base_url or config.deepseek_base_url,
            name="deepseek",
        )


class QwenProvider(OpenAICompatibleProvider):
    def __init__(self, model: str = "", api_key: str = "", base_url: str = "") -> None:
        super().__init__(
            model=model or config.llm_model or "qwen-turbo",
            api_key=api_key or config.qwen_api_key,
            base_url=base_url or config.qwen_base_url,
            name="qwen",
        )


_PROVIDER_MAP: dict[str, type[OpenAICompatibleProvider]] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
}


def get_provider(provider_name: str | None = None, model: str | None = None) -> BaseProvider:
    """工厂函数：根据配置创建 Provider。"""
    name = provider_name or config.llm_provider
    if name not in _PROVIDER_MAP:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_PROVIDER_MAP)}")
    cls = _PROVIDER_MAP[name]
    kwargs: dict[str, str] = {}
    if model:
        kwargs["model"] = model
    return cls(**kwargs)
