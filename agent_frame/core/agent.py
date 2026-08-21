"""通用 Agent 基类。

标准接口：run(task) → think → act 循环，支持：
- 工具挂载：self.tools → 自动生成 Function Calling schema
- 记忆注入：self.memory → get_context() 注入 prompt
- 子类扩展：覆盖 _get_system_prompt() 即可定制行为

设计思路：
1. Think 阶段：LLM 分析任务 + 上下文，决定是调用工具还是输出最终答案
2. Act 阶段：执行工具调用，收集结果
3. 循环直到 LLM 输出最终答案或达到 max_iterations
"""

from __future__ import annotations

import json
from typing import Any

from agent_frame.config import config
from agent_frame.llm.base import BaseProvider
from agent_frame.memory.base import BaseMemory, NoMemory
from agent_frame.tools.registry import ToolRegistry
from agent_frame.utils.logger import get_logger

logger = get_logger("core.agent")

# ── 默认系统提示 ─────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """你是一个智能助手，可以使用工具完成任务。

## 工作方式
1. 分析用户的任务需求
2. 如果需要使用工具，调用合适的工具函数获取信息
3. 基于工具返回的结果，给出最终答案
4. 如果不需要工具，直接给出答案

## 注意事项
- 每次只调用一个工具
- 工具返回结果后，根据结果决定下一步
- 如果工具调用失败，尝试其他方法或请求用户澄清
- 最终答案应清晰、完整、准确
"""


class BaseAgent:
    """通用 Agent 基类。

    子类化仅需覆盖 _get_system_prompt()：
        class MyAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "你是一个专业的Python开发者..."

    用法：
        agent = BaseAgent(provider)
        agent.mount_tools(registry)
        agent.mount_memory(memory)
        result = await agent.run("帮我计算 (3+5)*2")
    """

    def __init__(
        self,
        provider: BaseProvider,
        name: str = "agent",
        max_iterations: int | None = None,
    ) -> None:
        """初始化 Agent。

        Args:
            provider: LLM Provider 实例。
            name: Agent 名称标识。
            max_iterations: think/act 最大循环次数。
        """
        self.provider = provider
        self.name = name
        self.max_iterations = max_iterations or config.agent_max_iterations
        self._tools: ToolRegistry | None = None
        self._memory: BaseMemory = NoMemory()
        self._iteration_log: list[dict[str, Any]] = []

    # ── 工具/记忆挂载 ─────────────────────────────────────

    def mount_tools(self, registry: ToolRegistry) -> None:
        """挂载工具注册中心。

        一个 Agent 可以挂载整个 ToolRegistry，
        Agent 会自动获取所有已注册工具的 Schema。
        """
        self._tools = registry
        logger.info("[%s] Tools mounted: %d tools", self.name, registry.count)

    def mount_memory(self, memory: BaseMemory) -> None:
        """挂载记忆模块。"""
        self._memory = memory
        logger.info("[%s] Memory mounted: %s", self.name, type(memory).__name__)

    # ── 子类可覆盖 ────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        """获取系统提示词（子类覆盖以定制行为）。"""
        return DEFAULT_SYSTEM_PROMPT

    # ── 核心循环 ──────────────────────────────────────────

    async def run(self, task: str) -> str:
        """执行 Agent 主循环：think → act → think → ...

        Args:
            task: 用户任务描述。

        Returns:
            Agent 的最终文本回答。
        """
        self._iteration_log.clear()

        # 获取记忆上下文
        memory_context = self._memory.get_context(query=task)

        # 构建初始 messages
        system_prompt = self._get_system_prompt()
        if memory_context:
            system_prompt += f"\n\n## 对话历史\n{memory_context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        tools_schema = self._tools.get_schemas() if self._tools else None

        for iteration in range(self.max_iterations):
            logger.debug("[%s] Iteration %d/%d", self.name, iteration + 1, self.max_iterations)

            # ── THINK：LLM 决定下一步 ──────────────────
            if tools_schema:
                response = await self.provider.generate_with_tools(
                    messages=messages,
                    tools=tools_schema,
                    temperature=config.agent_temperature,
                )
            else:
                text = await self.provider.generate(
                    prompt=messages[-1]["content"],
                    system_prompt=system_prompt,
                    temperature=config.agent_temperature,
                )
                response = {"content": text, "tool_calls": None}

            # 记录迭代
            self._iteration_log.append({
                "iteration": iteration + 1,
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls"),
            })

            # ── 无工具调用 → 返回最终答案 ─────────────
            tool_calls = response.get("tool_calls")
            if not tool_calls:
                final_answer = response.get("content", "")
                # 更新记忆
                self._memory.add("user", task)
                self._memory.add("assistant", final_answer)
                logger.info("[%s] Completed in %d iterations", self.name, iteration + 1)
                return final_answer

            # ── ACT：执行工具调用 ──────────────────────
            # 一条 assistant 消息承载所有 tool_calls（标准 OpenAI 协议）
            messages.append({
                "role": "assistant",
                "content": response.get("content") or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"] if isinstance(tc["arguments"], str)
                            else json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # 每个工具调用对应一条 tool 消息，注入执行结果
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_result = self._tools.call(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

                logger.info(
                    "[%s] Tool called: %s(%s) → %s",
                    self.name, tool_name,
                    str(tool_args)[:50], tool_result[:100],
                )

        logger.warning("[%s] Max iterations (%d) reached", self.name, self.max_iterations)
        return "已达到最大迭代次数，但未能完成任务。请尝试简化需求或增加迭代限制。"

    # ── 辅助 ──────────────────────────────────────────────

    @property
    def iteration_log(self) -> list[dict[str, Any]]:
        """获取 think/act 循环的完整日志。"""
        return self._iteration_log

    @property
    def tool_count(self) -> int:
        return self._tools.count if self._tools else 0
