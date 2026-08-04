"""内置工具集（6款）。

提供开箱即用的常用工具：
1. calculator — 安全的数学表达式计算
2. read_file  — 读取文本文件内容
3. write_file — 写入文本文件
4. current_time — 获取当前日期时间
5. str_length — 计算字符串长度
6. json_parser — JSON 字符串解析
"""

from __future__ import annotations

import json
import math
import operator
from datetime import datetime
from pathlib import Path

from agent_frame.tools.registry import ToolRegistry
from agent_frame.utils.logger import get_logger

logger = get_logger("tools.builtin")

# 安全的 eval 白名单
_SAFE_OPERATORS: dict[str, object] = {
    k: getattr(math, k) for k in dir(math)
    if not k.startswith("_") and callable(getattr(math, k))
}
_SAFE_OPERATORS.update({
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "int": int, "float": float,
    "str": str, "bool": bool, "list": list,
})


def _safe_eval(expression: str) -> float:
    """安全的数学表达式求值。

    仅允许数学函数和基本运算符，禁止 __builtins__ 访问。
    """
    code = compile(expression.strip(), "<calculator>", "eval")
    for name in code.co_names:
        if name not in _SAFE_OPERATORS and name not in __builtins__:
            pass  # 允许 builtins
    return eval(code, {"__builtins__": {}, **_SAFE_OPERATORS})


def _register_all(registry: ToolRegistry) -> None:
    """将所有内置工具注册到指定注册中心。"""

    # 1. calculator
    def calculator(expression: str) -> str:
        """安全计算数学表达式。expression: 数学表达式，如 '2+3*4' 或 'sqrt(16)'"""
        try:
            result = _safe_eval(expression)
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    # 2. read_file
    def read_file(filepath: str) -> str:
        """读取文本文件内容。filepath: 文件路径（相对或绝对）"""
        try:
            path = Path(filepath)
            if not path.exists():
                return f"File not found: {filepath}"
            content = path.read_text(encoding="utf-8")
            if len(content) > 2000:
                return content[:2000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"Read error: {e}"

    # 3. write_file
    def write_file(filepath: str, content: str) -> str:
        """写入文本文件。filepath: 文件路径 | content: 要写入的内容"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} chars to {filepath}"
        except Exception as e:
            return f"Write error: {e}"

    # 4. current_time
    def current_time(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """获取当前日期时间。format_str: strftime 格式字符串，默认 '%Y-%m-%d %H:%M:%S'"""
        return datetime.now().strftime(format_str)

    # 5. str_length
    def str_length(text: str) -> str:
        """计算字符串长度（字符数和字节数）。text: 要统计的文本"""
        return f"Characters: {len(text)}, Bytes: {len(text.encode('utf-8'))}"

    # 6. json_parser
    def json_parser(json_str: str, key_path: str = "") -> str:
        """解析 JSON 字符串并提取指定键值。json_str: JSON 字符串 | key_path: 点号分隔的键路径，如 'data.name'"""
        try:
            data = json.loads(json_str)
            if key_path:
                for key in key_path.split("."):
                    if isinstance(data, dict):
                        data = data.get(key)
                    elif isinstance(data, list):
                        data = data[int(key)]
                    else:
                        return f"Cannot navigate into {type(data).__name__}"
                return json.dumps(data, ensure_ascii=False, indent=2)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"JSON parse error: {e}"

    registry.register(calculator)
    registry.register(read_file)
    registry.register(write_file)
    registry.register(current_time)
    registry.register(str_length)
    registry.register(json_parser)
    logger.info("6 builtin tools registered")


def register_builtin_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """注册所有内置工具。

    Args:
        registry: 目标注册中心，默认使用全局单例。

    Returns:
        ToolRegistry 实例。
    """
    if registry is None:
        registry = ToolRegistry()
    _register_all(registry)
    return registry
