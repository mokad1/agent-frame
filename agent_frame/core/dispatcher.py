"""多 Agent 调度器（Dispatcher）。

功能：
- 任务拆分：LLM 将复杂任务分解为子任务列表
- 子任务分发：将子任务分配给指定的 Worker Agent 执行
- 结果汇总：收集所有子任务结果，生成最终输出
- 失败重试：子任务失败自动重试（最多 N 次）

设计思路：
- Dispatcher 本身不执行具体工作，只负责协调
- Worker Agent 可以是同一个 Agent 实例的不同调用，也可以是不同类型的 Agent
- 适合场景：调研报告生成、多维度分析、多步骤工作流
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_frame.config import config
from agent_frame.core.agent import BaseAgent
from agent_frame.llm.base import BaseProvider
from agent_frame.utils.logger import get_logger

logger = get_logger("core.dispatcher")

PLANNING_PROMPT = """你是一个任务规划器。将以下复杂任务拆分为 {max_subtasks} 个以内的独立子任务。

拆分原则：
1. 每个子任务应该是独立的、可单独执行的
2. 子任务之间可以有依赖关系，但要最小化
3. 每个子任务描述应清晰、具体、可操作
4. 按执行顺序排列

## 原始任务
{task}

请严格按以下 JSON 格式输出（不要输出其他内容）：
{{"subtasks": [{{"id": 1, "title": "子任务标题", "description": "详细描述", "agent_type": "default"}}]}}
"""

MERGE_PROMPT = """你是一个结果整合器。根据以下子任务结果，生成一份完整的最终报告。

## 原始任务
{task}

## 子任务结果
{subtask_results}

请整合所有结果，输出一份结构清晰、内容完整的最终报告。
"""


class Dispatcher:
    """多 Agent 调度器。

    用法：
        dispatcher = Dispatcher(provider)
        dispatcher.register_worker("default", agent)
        result = await dispatcher.run("调研Python异步编程的最新进展")
    """

    def __init__(
        self,
        provider: BaseProvider,
        max_retries: int | None = None,
        max_subtasks: int = 5,
    ) -> None:
        """初始化调度器。

        Args:
            provider: LLM Provider（用于任务规划和结果合并）。
            max_retries: 子任务失败最大重试次数。
            max_subtasks: 最大子任务拆分数量。
        """
        self.provider = provider
        self.max_retries = max_retries or config.dispatcher_max_retries
        self.max_subtasks = max_subtasks
        self._workers: dict[str, BaseAgent] = {}

    def register_worker(self, agent_type: str, agent: BaseAgent) -> None:
        """注册 Worker Agent。

        Args:
            agent_type: Agent 类型标识（如 "default", "researcher", "coder"）。
            agent: BaseAgent 实例。
        """
        self._workers[agent_type] = agent
        logger.info("Worker registered: %s → %s", agent_type, agent.name)

    async def run(self, task: str) -> dict[str, Any]:
        """执行调度流程：规划 → 分发 → 汇总。

        Args:
            task: 用户原始任务。

        Returns:
            {"plan": [...], "results": [...], "final_report": str}
        """
        # ── Step 1: 任务规划 ─────────────────────────────
        plan = await self._plan(task)
        logger.info("Task planned: %d subtasks", len(plan))

        if not plan:
            # 无法拆解，直接交给 default agent
            worker = self._workers.get("default") or list(self._workers.values())[0]
            answer = await worker.run(task)
            return {"plan": [], "results": [{"subtask": task, "result": answer}], "final_report": answer}

        # ── Step 2: 子任务分发执行 ───────────────────────
        results = await self._dispatch(plan)

        # ── Step 3: 结果汇总 ─────────────────────────────
        final_report = await self._merge(task, results)

        return {
            "plan": plan,
            "results": results,
            "final_report": final_report,
        }

    async def _plan(self, task: str) -> list[dict[str, Any]]:
        """LLM 拆解任务为子任务列表。"""
        import json as _json

        prompt = PLANNING_PROMPT.format(max_subtasks=self.max_subtasks, task=task)
        try:
            raw = await self.provider.generate(prompt, temperature=0.3, max_tokens=2048)
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = _json.loads(raw.strip())
            return data.get("subtasks", [])
        except Exception as e:
            logger.warning("Task planning failed: %s, using single-task fallback", e)
            return [{"id": 1, "title": task, "description": task, "agent_type": "default"}]

    async def _dispatch(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """分发子任务到 Worker Agent 并收集结果。

        支持失败重试：子任务失败后最多重试 max_retries 次。
        """
        results: list[dict[str, Any]] = []

        for subtask in plan:
            agent_type = subtask.get("agent_type", "default")
            worker = self._workers.get(agent_type)
            if worker is None:
                worker = self._workers.get("default") or list(self._workers.values())[0]

            description = subtask.get("description", subtask.get("title", ""))
            success = False

            for attempt in range(self.max_retries + 1):
                try:
                    answer = await worker.run(description)
                    results.append({
                        "subtask_id": subtask.get("id"),
                        "title": subtask.get("title", ""),
                        "result": answer,
                        "status": "success",
                        "attempts": attempt + 1,
                    })
                    success = True
                    logger.info("Subtask %s completed (attempt %d)", subtask.get("id"), attempt + 1)
                    break
                except Exception as e:
                    logger.warning(
                        "Subtask %s attempt %d/%d failed: %s",
                        subtask.get("id"), attempt + 1, self.max_retries + 1, e,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)

            if not success:
                results.append({
                    "subtask_id": subtask.get("id"),
                    "title": subtask.get("title", ""),
                    "result": "Failed after all retries",
                    "status": "failed",
                    "attempts": self.max_retries + 1,
                })

        return results

    async def _merge(self, task: str, results: list[dict[str, Any]]) -> str:
        """汇总子任务结果生成最终报告。"""
        lines: list[str] = []
        for r in results:
            sid = r.get("subtask_id", "?")
            title = r.get("title", f"Subtask {sid}")
            status = r.get("status", "?")
            result_text = r.get("result", "")
            lines.append(f"### {title} [{status}]\n{result_text}")
        result_text = "\n\n".join(lines)

        prompt = MERGE_PROMPT.format(task=task, subtask_results=result_text)
        try:
            return await self.provider.generate(prompt, temperature=0.5, max_tokens=4096)
        except Exception as e:
            logger.error("Result merging failed: %s", e)
            return result_text

    @property
    def worker_types(self) -> list[str]:
        return list(self._workers.keys())
