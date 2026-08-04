"""工具注册中心 + 装饰器注册 + Function Calling JSON Schema 自动生成。

设计思路：
- 通过 @register_tool 装饰器注册任意 Python 函数为 Agent 工具
- 自动提取函数签名 + 类型注解 + docstring → OpenAI Function Calling JSON Schema
- ToolRegistry 管理所有已注册工具的增删查
- 统一的 Schema 格式在所有 Provider 间通用
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from agent_frame.utils.logger import get_logger

logger = get_logger("tools.registry")

# Python 类型 → JSON Schema type 映射
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """将 Python 类型注解转为 JSON Schema 属性定义。"""
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        return {"type": "string", "description": f"Type: {py_type}"}
    json_type = _TYPE_MAP.get(py_type, "string")
    return {"type": json_type}


def _build_function_schema(func: Callable) -> dict[str, Any]:
    """从函数签名自动生成 Function Calling JSON Schema。

    提取：函数名、docstring 首行作描述、参数名+类型+默认值。

    Args:
        func: 被注册的 Python 函数。

    Returns:
        标准的 OpenAI Function Calling tool schema。
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n")[0] if doc else func.__name__

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        param_schema = {"type": "string"}
        if name in hints:
            param_schema = _python_type_to_json_schema(hints[name])

        # 从 docstring 中提取参数说明
        param_doc_match = None
        if doc:
            import re
            pattern = rf"{name}\s*[:：]\s*(.+?)(?:\n|$)"
            param_doc_match = re.search(pattern, doc)
        if param_doc_match:
            param_schema["description"] = param_doc_match.group(1).strip()
        else:
            param_schema["description"] = f"Parameter: {name}"

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            default = param.default
            if default is not None:
                param_schema["default"] = default

        properties[name] = param_schema

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolRegistry:
    """工具注册中心。

    单例模式，全局管理所有已注册的工具。

    用法：
        registry = ToolRegistry()
        registry.register(my_func)
        schemas = registry.get_schemas()
        result = registry.call("my_func", {"arg": "value"})
    """

    _instance: ToolRegistry | None = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: dict[str, Callable] = {}
            cls._instance._schemas: dict[str, dict[str, Any]] = {}
        return cls._instance

    def register(self, func: Callable, name: str | None = None) -> Callable:
        """注册一个工具函数。

        Args:
            func: Python 可调用对象。
            name: 工具名（默认使用函数名）。

        Returns:
            原函数（装饰器兼容）。
        """
        tool_name = name or func.__name__
        if tool_name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting.", tool_name)

        self._tools[tool_name] = func
        self._schemas[tool_name] = _build_function_schema(func)
        logger.info("Tool registered: %s", tool_name)
        return func

    def unregister(self, name: str) -> None:
        """注销工具。"""
        self._tools.pop(name, None)
        self._schemas.pop(name, None)

    def get_schemas(self) -> list[dict[str, Any]]:
        """获取所有已注册工具的 Function Calling JSON Schema 列表。"""
        return list(self._schemas.values())

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """获取指定工具的 Schema。"""
        return self._schemas.get(name)

    def call(self, name: str, arguments: dict[str, Any] | str) -> str:
        """调用已注册的工具函数。

        Args:
            name: 工具名。
            arguments: 参数字典或 JSON 字符串。

        Returns:
            工具执行结果字符串。
        """
        func = self._tools.get(name)
        if func is None:
            return f"Error: Tool '{name}' not found. Available: {list(self._tools.keys())}"

        if isinstance(arguments, str):
            import json
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments for '{name}': {arguments}"

        try:
            result = func(**arguments)
            return str(result)
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", name, e)
            return f"Error executing '{name}': {e}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        return len(self._tools)


# ── 装饰器 ───────────────────────────────────────────────────

def register_tool(name: str | None = None) -> Callable:
    """工具注册装饰器。

    用法：
        @register_tool()
        def calculator(expression: str) -> str:
            '''计算数学表达式。expression: 数学表达式'''
            return str(eval(expression))

        @register_tool("web_search")
        def search(query: str) -> str:
            '''搜索网页。query: 搜索关键词'''
            return f"Results for: {query}"
    """
    registry = ToolRegistry()

    def decorator(func: Callable) -> Callable:
        registry.register(func, name=name)
        return func

    return decorator
