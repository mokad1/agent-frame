"""工具系统单元测试。"""

import pytest
from agent_frame.tools.registry import ToolRegistry, register_tool
from agent_frame.tools.builtin import register_builtin_tools


class TestToolRegistry:
    """ToolRegistry 核心功能测试。"""

    def test_register_single_tool(self) -> None:
        registry = ToolRegistry()
        before = registry.count
        def add(a: int, b: int) -> str:
            """Add two numbers. a: first | b: second"""
            return str(a + b)

        registry.register(add)
        assert "add" in registry.tool_names
        assert registry.count == before + 1

    def test_register_decorator(self) -> None:
        registry = ToolRegistry.__new__(ToolRegistry)
        registry._tools = {}
        registry._schemas = {}

        @register_tool()
        def greet(name: str) -> str:
            """Greet someone. name: person's name"""
            return f"Hello, {name}!"

        assert "greet" in ToolRegistry().tool_names

    def test_tool_schema_generation(self) -> None:
        registry = ToolRegistry()
        def multiply(x: float, y: float) -> str:
            """Multiply two numbers. x: first factor | y: second factor"""
            return str(x * y)

        registry.register(multiply)
        schema = registry.get_schema("multiply")
        assert schema is not None
        assert schema["function"]["name"] == "multiply"
        assert "x" in schema["function"]["parameters"]["properties"]
        assert "y" in schema["function"]["parameters"]["properties"]

    def test_tool_call(self) -> None:
        registry = ToolRegistry()
        def echo(msg: str) -> str:
            return f"Echo: {msg}"

        registry.register(echo)
        result = registry.call("echo", {"msg": "hello"})
        assert result == "Echo: hello"

    def test_tool_not_found(self) -> None:
        registry = ToolRegistry()
        result = registry.call("nonexistent", {})
        assert "not found" in result

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        def temp() -> str:
            return "temp"
        registry.register(temp)
        assert "temp" in registry.tool_names
        registry.unregister("temp")
        assert "temp" not in registry.tool_names

    def test_get_schemas(self) -> None:
        registry = ToolRegistry()
        before = registry.count
        def f1() -> str:
            return ""
        def f2() -> str:
            return ""
        registry.register(f1)
        registry.register(f2)
        schemas = registry.get_schemas()
        assert len(schemas) == before + 2

    def test_call_with_json_string_args(self) -> None:
        registry = ToolRegistry()
        def add(a: int, b: int) -> str:
            return str(a + b)
        registry.register(add)
        result = registry.call("add", '{"a": 3, "b": 4}')
        assert "7" in result


class TestBuiltinTools:
    """内置工具测试。"""

    def test_calculator(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        result = registry.call("calculator", {"expression": "2+3*4"})
        assert "14" in result

    def test_calculator_sqrt(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        result = registry.call("calculator", {"expression": "sqrt(16)"})
        assert "4.0" in result

    def test_str_length(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        result = registry.call("str_length", {"text": "你好"})
        assert "Characters: 2" in result

    def test_json_parser(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        result = registry.call("json_parser", {"json_str": '{"name": "test", "age": 25}', "key_path": "name"})
        assert "test" in result

    def test_current_time(self) -> None:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        result = registry.call("current_time", {})
        assert len(result) > 10  # 至少是完整日期时间格式


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
