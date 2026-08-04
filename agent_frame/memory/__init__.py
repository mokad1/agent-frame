"""模块化记忆抽象层。"""

from agent_frame.memory.base import BaseMemory
from agent_frame.memory.sliding_window import SlidingWindowMemory

# VectorMemory 需要 faiss，延迟导入避免未安装时阻塞其他模块
__all__ = ["BaseMemory", "SlidingWindowMemory", "VectorMemory"]


def __getattr__(name: str):
    if name == "VectorMemory":
        from agent_frame.memory.vector_memory import VectorMemory
        return VectorMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
