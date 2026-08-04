"""示例 1：单 Agent 工具调用（计算器 + 文件读写）。

演示：
- 创建 Agent
- 注册内置工具
- 挂载工具和记忆
- 执行多步任务（需要工具调用才能完成）

运行：
    python examples/single_agent_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent_frame.config import config
from agent_frame.core.agent import BaseAgent
from agent_frame.llm.providers import get_provider
from agent_frame.memory.sliding_window import SlidingWindowMemory
from agent_frame.tools.builtin import register_builtin_tools
from agent_frame.tools.registry import ToolRegistry
from agent_frame.utils.logger import setup_logging


class MathAssistant(BaseAgent):
    """数学助手 Agent — 仅覆盖系统提示词。"""

    def _get_system_prompt(self) -> str:
        return (
            "你是一个数学助手，可以使用计算器工具完成数学运算。\n"
            "对于用户的数学问题，使用 calculator 工具计算，然后基于结果给出答案。\n"
            "重要：工具返回的结果是精确的数值，直接引用即可。"
        )


async def main() -> None:
    setup_logging("INFO")

    # 1. 创建 Provider
    provider = get_provider()

    # 2. 创建工具注册中心并注册内置工具
    registry = ToolRegistry()
    register_builtin_tools(registry)
    print(f"Tools available: {registry.tool_names}")

    # 3. 创建 Agent 并挂载工具 + 记忆
    agent = MathAssistant(provider, name="math_agent", max_iterations=8)
    agent.mount_tools(registry)
    agent.mount_memory(SlidingWindowMemory(window_size=6))

    # 4. 执行任务
    tasks = [
        "计算 (3 + 5) * 12 / 4 的结果",
        "计算 2 的 10 次方",
        "计算根号 256 加上 100 的一半",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"User: {task}")
        print("-" * 60)

        result = await agent.run(task)
        print(f"Agent: {result}")

        # 打印迭代日志
        print(f"\n(Iterations: {len(agent.iteration_log)})")
        for log in agent.iteration_log:
            tc_info = ""
            if log.get("tool_calls"):
                tc = log["tool_calls"][0]
                tc_info = f" → Tool: {tc['name']}"
            print(f"  Step {log['iteration']}: {log['content'][:80] or '(tool call)'}{tc_info}")

    await provider.close()
    print("\n✅ Demo completed.")


if __name__ == "__main__":
    asyncio.run(main())
