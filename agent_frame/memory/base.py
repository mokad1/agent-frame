"""记忆抽象接口。

定义统一的记忆操作规范，所有记忆实现必须继承此基类。
Agent 通过此接口操作记忆，无需关心底层实现（滑动窗口 vs 向量检索）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """记忆抽象基类。

    定义 add / get_context / clear / count 四个标准操作。
    """

    @abstractmethod
    def add(self, role: str, content: str, **meta: Any) -> None:
        """添加一条记忆。

        Args:
            role: 消息角色（user / assistant / system）。
            content: 消息内容。
            **meta: 额外元数据（时间戳、来源等）。
        """
        ...

    @abstractmethod
    def get_context(self, query: str = "", max_tokens: int = 2000) -> str:
        """获取当前上下文字符串。

        Args:
            query: 当前查询（用于向量记忆的语义检索）。
            max_tokens: 最大返回 token 数（估算）。

        Returns:
            格式化的上下文字符串。
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆。"""
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """记忆条目数。"""
        ...


class NoMemory(BaseMemory):
    """空记忆实现——无记忆模式。"""

    def add(self, role: str, content: str, **meta: Any) -> None:
        pass

    def get_context(self, query: str = "", max_tokens: int = 2000) -> str:
        return ""

    def clear(self) -> None:
        pass

    @property
    def count(self) -> int:
        return 0
