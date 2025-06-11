"""Unit tests for tool registry and implementations."""

import pytest
from core.tools import ToolRegistry


class TestToolRegistry:
    """Test tool registry functionality."""
    
    def test_register_tool(self):
        """Test registering a custom tool."""
        @ToolRegistry.register("test_tool", "A test tool")
        def test_tool(query: str) -> str:
            return f"Test: {query}"
        
        # Should be registered
        tool = ToolRegistry.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool("hello") == "Test: hello"
    
    def test_get_builtin_tool(self):
        """Test getting built-in tools."""
        # Calculator should be registered
        calc = ToolRegistry.get("calculator")
        assert calc is not None
        assert calc.name == "calculator"
        assert "mathematical" in calc.description.lower()
        
        # Test basic calculation
        result = calc("2 + 2")
        assert "4" in result
    
    def test_get_nonexistent_tool_returns_mock(self):
        """Test that nonexistent tools return mock implementation."""
        tool = ToolRegistry.get("nonexistent_tool")
        assert tool is not None
        assert tool.name == "nonexistent_tool"
        assert "not implemented" in tool.description
        
        result = tool("test query")
        assert "not implemented" in result
        assert "test query" in result
    
    def test_calculator_tool(self):
        """Test calculator tool functionality."""
        calc = ToolRegistry.get("calculator")
        
        # Test basic operations
        assert "4" in calc("2 + 2")
        assert "10" in calc("5 * 2")
        assert "3" in calc("9 / 3")
        assert "8" in calc("2 ** 3")
        
        # Test invalid input
        result = calc("2 + x")
        assert "Error" in result
    
    def test_web_search_tool(self):
        """Test web search tool (mock)."""
        search = ToolRegistry.get("web_search")
        result = search("DSPy framework")
        
        assert "Web search results" in result
        assert "DSPy framework" in result
        assert "Mock results" in result
    
    def test_json_parser_tool(self):
        """Test JSON parser tool."""
        parser = ToolRegistry.get("json_parser")
        
        # Valid JSON
        result = parser('{"key": "value", "number": 42}')
        assert "Parsed JSON" in result
        assert '"key": "value"' in result
        
        # Invalid JSON
        result = parser("not json")
        assert "error" in result.lower()
    
    def test_all_builtin_tools_have_required_attributes(self):
        """Test that all built-in tools have required attributes."""
        builtin_tools = [
            "calculator",
            "web_search", 
            "database_query",
            "python_repl",
            "file_reader",
            "json_parser"
        ]
        
        for tool_name in builtin_tools:
            tool = ToolRegistry.get(tool_name)
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert tool.name == tool_name
            assert callable(tool)