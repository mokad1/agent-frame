"""Agent 基类单元测试。"""

import pytest
from agent_frame.core.agent import BaseAgent
from agent_frame.memory.base import NoMemory
from agent_frame.memory.sliding_window import SlidingWindowMemory
from agent_frame.tools.registry import ToolRegistry


class MockProvider:
    """Mock LLM Provider 用于测试。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Mock answer"]
        self.call_count = 0
        self.tool_call_mode = False

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3, max_tokens: int = 4096) -> str:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

    async def generate_with_tools(self, messages: list | None = None, tools: list | None = None, temperature: float = 0.3, max_tokens: int = 4096) -> dict:
        if self.tool_call_mode and self.call_count == 0:
            self.call_count += 1
            return {
                "content": None,
                "tool_calls": [{"id": "call_1", "name": "echo", "arguments": {"msg": "test"}}],
            }
        self.call_count += 1
        return {"content": "Final answer", "tool_calls": None}


class TestBaseAgent:
    """BaseAgent 核心功能测试。"""

    def test_agent_creation(self) -> None:
        agent = BaseAgent(MockProvider(), name="test")
        assert agent.name == "test"
        assert agent.tool_count == 0
        assert agent.max_iterations == 10

    def test_mount_memory(self) -> None:
        agent = BaseAgent(MockProvider())
        mem = SlidingWindowMemory(window_size=4)
        agent.mount_memory(mem)
        assert isinstance(agent._memory, SlidingWindowMemory)

    def test_mount_tools(self) -> None:
        agent = BaseAgent(MockProvider())
        registry = ToolRegistry()
        def echo(msg: str) -> str:
            return f"Echo: {msg}"
        registry.register(echo)
        agent.mount_tools(registry)
        assert agent.tool_count == 1

    def test_no_memory_default(self) -> None:
        agent = BaseAgent(MockProvider())
        assert isinstance(agent._memory, NoMemory)

    @pytest.mark.asyncio
    async def test_run_without_tools(self) -> None:
        agent = BaseAgent(MockProvider(["Simple answer"]))
        result = await agent.run("Test task")
        assert result == "Simple answer"
        assert len(agent.iteration_log) == 1

    @pytest.mark.asyncio
    async def test_run_with_tools(self) -> None:
        provider = MockProvider()
        provider.tool_call_mode = True
        agent = BaseAgent(provider, max_iterations=5)

        registry = ToolRegistry()
        def echo(msg: str) -> str:
            return f"Echo: {msg}"
        registry.register(echo)
        agent.mount_tools(registry)

        result = await agent.run("Echo test")
        assert result == "Final answer"
        assert len(agent.iteration_log) == 2

    def test_system_prompt_override(self) -> None:
        class CustomAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "Custom prompt"

        agent = CustomAgent(MockProvider())
        assert agent._get_system_prompt() == "Custom prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
