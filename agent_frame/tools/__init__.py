"""可插拔工具系统。"""

from agent_frame.tools.registry import ToolRegistry, register_tool
from agent_frame.tools.builtin import register_builtin_tools

__all__ = ["ToolRegistry", "register_builtin_tools", "register_tool"]
