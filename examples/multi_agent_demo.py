"""示例 2：多 Agent 协作 — 基于 Dispatcher 的调研助手。

演示：
- 创建 Dispatcher
- 注册不同类型的 Worker Agent
- 将复杂调研任务自动拆分为子任务
- 汇总生成最终报告

运行：
    python examples/multi_agent_demo.py
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
from agent_frame.core.dispatcher import Dispatcher
from agent_frame.llm.providers import get_provider
from agent_frame.tools.builtin import register_builtin_tools
from agent_frame.tools.registry import ToolRegistry
from agent_frame.utils.logger import setup_logging


# ── 专用 Agent 子类 ──────────────────────────────────────────

class ResearcherAgent(BaseAgent):
    """调研 Agent — 善于搜索和整理信息。"""

    def _get_system_prompt(self) -> str:
        return (
            "你是一个专业的研究员，善于收集和整理信息。\n"
            "对于给定的研究主题，提供全面、结构化的调研结果。\n"
            "如果涉及数据计算，使用 calculator 工具。\n"
            "如果需要读取文件内容，使用 read_file 工具。"
        )


class WriterAgent(BaseAgent):
    """写作 Agent — 善于撰写结构化报告。"""

    def _get_system_prompt(self) -> str:
        return (
            "你是一个专业的技术写作者，善于撰写结构化、清晰的技术报告。\n"
            "基于提供的信息，撰写一篇格式规范、逻辑清晰的文档。"
        )


# ── 主流程 ───────────────────────────────────────────────────

async def main() -> None:
    setup_logging("INFO")

    provider = get_provider()
    print(f"Provider: {provider.provider_name} | Model: {provider.model}")

    # 工具注册
    registry = ToolRegistry()
    register_builtin_tools(registry)

    # 创建不同角色的 Agent
    researcher = ResearcherAgent(provider, name="researcher", max_iterations=6)
    researcher.mount_tools(registry)

    writer = WriterAgent(provider, name="writer", max_iterations=4)
    writer.mount_tools(registry)

    # 创建 Dispatcher 并注册 Worker
    dispatcher = Dispatcher(provider, max_retries=2, max_subtasks=5)
    dispatcher.register_worker("researcher", researcher)
    dispatcher.register_worker("writer", writer)
    dispatcher.register_worker("default", researcher)  # 默认使用 researcher

    print(f"Workers: {dispatcher.worker_types}")

    # 执行复杂调研任务
    task = "调研 Python asyncio 的核心概念（事件循环、协程、Future/Task），并给出最佳实践建议"

    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")

    result = await dispatcher.run(task)

    print(f"\n📋 Plan ({len(result['plan'])} subtasks):")
    for s in result["plan"]:
        print(f"  [{s.get('id')}] {s.get('title')} → {s.get('agent_type')}")

    print(f"\n📊 Subtask Results:")
    for r in result["results"]:
        status_icon = "✅" if r.get("status") == "success" else "❌"
        print(f"  {status_icon} {r.get('title')} ({r.get('attempts', 1)} attempts)")
        print(f"     {r.get('result', '')[:120]}...")

    print(f"\n📝 Final Report:")
    print(result["final_report"])

    await provider.close()
    print("\n✅ Multi-agent demo completed.")


if __name__ == "__main__":
    asyncio.run(main())
