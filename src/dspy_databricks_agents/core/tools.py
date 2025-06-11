"""Tool implementations for ReAct agents."""

from typing import Callable, Dict
import json
import math


class ToolRegistry:
    """Registry for ReAct tools."""
    
    _tools: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, description: str = ""):
        """Decorator to register tools."""
        def decorator(func: Callable) -> Callable:
            func.name = name
            func.description = description or f"{name} tool"
            cls._tools[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a registered tool."""
        if name not in cls._tools:
            # Return a mock tool if not found
            return cls._create_mock_tool(name)
        return cls._tools[name]
    
    @classmethod
    def _create_mock_tool(cls, name: str) -> Callable:
        """Create a mock tool as fallback."""
        def mock_tool(query: str) -> str:
            return f"Tool '{name}' not implemented. Query: {query}"
        
        mock_tool.__name__ = name
        mock_tool.name = name
        mock_tool.description = f"Mock {name} tool (not implemented)"
        return mock_tool


# Register built-in tools

@ToolRegistry.register("calculator", "Performs mathematical calculations")
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        # Remove any dangerous operations
        safe_chars = "0123456789+-*/()., "
        if not all(c in safe_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        # Replace common math operations
        expression = expression.replace("^", "**")
        
        # Evaluate using Python's ast for safety
        import ast
        import operator
        
        # Supported operations
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_expr(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](eval_expr(node.operand))
            else:
                raise ValueError(f"Unsupported operation: {node}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return f"Result: {result}"
        
    except Exception as e:
        return f"Error: {str(e)}"


@ToolRegistry.register("web_search", "Searches the web for information")
def web_search(query: str) -> str:
    """Mock web search tool."""
    # In production, this would use an actual search API
    return f"Web search results for '{query}': [Mock results - integrate with real search API]"


@ToolRegistry.register("database_query", "Queries structured databases")
def database_query(query: str) -> str:
    """Mock database query tool."""
    # In production, this would execute actual SQL queries
    return f"Database query '{query}': [Mock results - integrate with real database]"


@ToolRegistry.register("python_repl", "Executes Python code")
def python_repl(code: str) -> str:
    """Execute Python code in a safe environment."""
    # In production, this would use a sandboxed Python environment
    return f"Python execution of '{code}': [Mock results - integrate with safe execution environment]"


@ToolRegistry.register("file_reader", "Reads content from files")
def file_reader(filepath: str) -> str:
    """Read file contents."""
    try:
        # In production, add proper security checks
        with open(filepath, 'r') as f:
            content = f.read(1000)  # Limit to first 1000 chars
            return f"File content: {content}..."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@ToolRegistry.register("json_parser", "Parses and extracts data from JSON")
def json_parser(json_str: str) -> str:
    """Parse JSON and return formatted result."""
    try:
        data = json.loads(json_str)
        return f"Parsed JSON: {json.dumps(data, indent=2)}"
    except json.JSONDecodeError as e:
        return f"JSON parsing error: {str(e)}"