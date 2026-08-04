"""短期滑动窗口记忆。

使用 collections.deque 实现固定大小的消息滑动窗口。
超出窗口的旧消息自动丢弃。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from agent_frame.config import config
from agent_frame.memory.base import BaseMemory
from agent_frame.utils.logger import get_logger

logger = get_logger("memory.sliding_window")


class SlidingWindowMemory(BaseMemory):
    """滑动窗口记忆。

    保留最近 N 条消息（默认 6 条），超出部分自动丢弃。
    适合短期对话上下文维护。
    """

    def __init__(self, window_size: int | None = None) -> None:
        """初始化。

        Args:
            window_size: 窗口大小（消息条数），默认从 config 读取。
        """
        self.window_size = window_size or config.memory_window_size
        self._messages: deque[dict[str, Any]] = deque(maxlen=self.window_size)

    def add(self, role: str, content: str, **meta: Any) -> None:
        """添加消息到窗口。"""
        self._messages.append({
            "role": role,
            "content": content,
            **meta,
        })

    def get_context(self, query: str = "", max_tokens: int = 2000) -> str:
        """获取滑动窗口中的对话上下文字符串。

        按时间顺序拼接，格式为 role: content。
        """
        if not self._messages:
            return ""

        lines: list[str] = []
        char_count = 0
        char_limit = max_tokens * 2  # 粗略估算：2 字符 ≈ 1 token

        for msg in self._messages:
            line = f"{msg['role']}: {msg['content']}"
            if char_count + len(line) > char_limit:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()

    @property
    def count(self) -> int:
        return len(self._messages)
