"""MLflow model signature utilities for DSPy agents."""

import mlflow
from mlflow.models import infer_signature
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
)
from typing import List, Dict, Any, Optional


def create_chat_model_signature():
    """Create MLflow model signature for ChatAgent interface.
    
    Returns:
        MLflow ModelSignature for chat agents
    """
    # Sample input
    sample_input = {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
    
    # Sample output
    sample_output = {
        "messages": [
            {
                "role": "assistant",
                "content": "I'm doing well, thank you! How can I help you today?"
            }
        ]
    }
    
    # Create signature
    signature = infer_signature(
        model_input=sample_input,
        model_output=sample_output
    )
    
    return signature


def create_chat_agent_signature():
    """Create MLflow model signature for backwards-compatible ChatAgent interface.
    
    Returns:
        MLflow ModelSignature for chat agents
    """
    from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
    
    # Sample input
    sample_input = {
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ]
    }
    
    # Sample output
    sample_output = {
        "messages": [
            {
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence..."
            }
        ]
    }
    
    # Create signature
    signature = infer_signature(
        model_input=sample_input,
        model_output=sample_output
    )
    
    return signature


def create_tool_enabled_signature():
    """Create MLflow model signature for tool-enabled chat models.
    
    Returns:
        MLflow ModelSignature for chat models with tool support
    """
    # Sample input with tools
    sample_input = {
        "messages": [
            {"role": "user", "content": "Calculate 25 * 4"}
        ],
        "custom_inputs": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Performs mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The mathematical expression to evaluate"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        }
    }
    
    # Sample output
    sample_output = {
        "messages": [
            {
                "role": "assistant",
                "content": "I'll calculate 25 * 4 for you.\nAction: calculator\nAction Input: 25 * 4\nObservation: 100\nThe result is 100."
            }
        ]
    }
    
    signature = infer_signature(
        model_input=sample_input,
        model_output=sample_output
    )
    
    return signature


def get_signature_for_config(config):
    """Get appropriate model signature based on agent configuration.
    
    Args:
        config: AgentConfig object
        
    Returns:
        MLflow ModelSignature
    """
    # Check if agent has tool-enabled modules
    has_tools = any(
        module.type.value == "react" and module.tools
        for module in config.modules
    )
    
    if has_tools:
        return create_tool_enabled_signature()
    else:
        # Use standard chat model signature
        return create_chat_model_signature()