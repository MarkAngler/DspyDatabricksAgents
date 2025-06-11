"""DSPy-Databricks Agents - Create and deploy DSPy-powered agentic workflows via YAML on Databricks."""

from config.schema import AgentConfig
from core.agent import Agent

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
]