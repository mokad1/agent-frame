"""记忆模块单元测试。"""

import pytest
from agent_frame.memory.sliding_window import SlidingWindowMemory
from agent_frame.memory.base import NoMemory


class TestSlidingWindowMemory:
    """滑动窗口记忆测试。"""

    def test_add_and_context(self) -> None:
        mem = SlidingWindowMemory(window_size=4)
        mem.add("user", "你好")
        mem.add("assistant", "你好！有什么可以帮助你的？")
        ctx = mem.get_context()
        assert "user" in ctx
        assert "assistant" in ctx
        assert mem.count == 2

    def test_window_overflow(self) -> None:
        mem = SlidingWindowMemory(window_size=2)
        mem.add("user", "msg1")
        mem.add("assistant", "msg2")
        mem.add("user", "msg3")
        assert mem.count == 2
        ctx = mem.get_context()
        assert "msg1" not in ctx  # 被淘汰
        assert "msg3" in ctx

    def test_clear(self) -> None:
        mem = SlidingWindowMemory(window_size=4)
        mem.add("user", "test")
        mem.clear()
        assert mem.count == 0
        assert mem.get_context() == ""

    def test_max_tokens_truncation(self) -> None:
        mem = SlidingWindowMemory(window_size=10)
        long_text = "x" * 5000
        mem.add("user", long_text)
        ctx = mem.get_context(max_tokens=100)
        assert len(ctx) < 5000  # 被截断


class TestNoMemory:
    """空记忆测试。"""

    def test_no_memory(self) -> None:
        mem = NoMemory()
        mem.add("user", "hello")
        assert mem.count == 0
        assert mem.get_context() == ""
        mem.clear()  # 不应报错


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
