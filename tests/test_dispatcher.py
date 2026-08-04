"""Dispatcher 调度器单元测试。"""

import pytest
from agent_frame.core.agent import BaseAgent
from agent_frame.core.dispatcher import Dispatcher


class MockProvider:
    """Mock Provider，每次调用返回预设响应。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["[]"]
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.3, max_tokens: int = 4096) -> str:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

    async def generate_with_tools(self, prompt: str, system_prompt: str = "", tools: list | None = None, temperature: float = 0.3, max_tokens: int = 4096) -> dict:
        return {"content": self.responses[self.call_count % len(self.responses)], "tool_calls": None}


class TestDispatcher:
    """Dispatcher 核心功能测试。"""

    def test_register_worker(self) -> None:
        disp = Dispatcher(MockProvider())
        agent = BaseAgent(MockProvider(["ok"]), name="worker1")
        disp.register_worker("default", agent)
        assert "default" in disp.worker_types

    def test_register_multiple_workers(self) -> None:
        disp = Dispatcher(MockProvider())
        disp.register_worker("default", BaseAgent(MockProvider(["ok"]), name="w1"))
        disp.register_worker("researcher", BaseAgent(MockProvider(["ok"]), name="w2"))
        assert len(disp.worker_types) == 2

    @pytest.mark.asyncio
    async def test_run_simple(self) -> None:
        # 规划失败 → 降级为单任务 fallback
        provider = MockProvider(["invalid json!!!"])  # 无法解析 → fallback
        disp = Dispatcher(provider)
        worker = BaseAgent(MockProvider(["Simple result"]), name="worker")
        disp.register_worker("default", worker)

        result = await disp.run("简单任务")
        assert "final_report" in result
        assert len(result["plan"]) == 1  # fallback 生成单个子任务
        assert result["plan"][0]["agent_type"] == "default"

    @pytest.mark.asyncio
    async def test_run_with_subtasks(self) -> None:
        # 规划返回 2 个子任务
        plan_json = '{"subtasks": [{"id": 1, "title": "Research", "description": "Find info", "agent_type": "default"}, {"id": 2, "title": "Summarize", "description": "Summarize findings", "agent_type": "default"}]}'
        merge_response = "Merged final report"

        provider = MockProvider([plan_json, merge_response])
        disp = Dispatcher(provider)
        worker = BaseAgent(MockProvider(["Subtask result"]), name="worker")
        disp.register_worker("default", worker)

        result = await disp.run("复杂调研任务")
        assert len(result["plan"]) == 2
        assert len(result["results"]) == 2
        assert result["final_report"] == "Merged final report"

    @pytest.mark.asyncio
    async def test_planning_fallback(self) -> None:
        # 规划失败 → 降级为单任务
        provider = MockProvider(["invalid json!!!"])
        disp = Dispatcher(provider)
        worker = BaseAgent(MockProvider(["Fallback result"]), name="worker")
        disp.register_worker("default", worker)

        result = await disp.run("任务")
        assert result["plan"][0]["agent_type"] == "default"  # fallback 使用 default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
