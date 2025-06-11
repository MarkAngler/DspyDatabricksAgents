"""MLflow model implementation for DSPy-Databricks agents."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import pandas as pd
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from config.schema import AgentConfig
from core.agent import Agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSPyAgentModel(mlflow.pyfunc.PythonModel):
    """MLflow model for serving DSPy agents."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config path.

        Args:
            config_path: Path to agent configuration JSON file
        """
        self.config_path = config_path
        self.agent = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model artifacts and initialize agent.

        Args:
            context: MLflow context containing artifacts
        """
        try:
            # Load configuration from artifacts
            if "config_path" in context.artifacts:
                config_path = context.artifacts["config_path"]
            else:
                # Fallback to looking in the model directory
                config_path = (
                    Path(context.artifacts["model_path"])
                    / "config"
                    / "agent_config.json"
                )

            logger.info(f"Loading agent configuration from: {config_path}")

            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # Create agent configuration
            config = AgentConfig(**config_dict)

            # Initialize agent
            self.agent = Agent(config)
            logger.info(f"Successfully loaded agent: {config.name} v{config.version}")

        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            raise

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        """Process chat messages and return response.

        Args:
            context: MLflow context
            model_input: Input data containing messages
            params: Optional parameters for prediction

        Returns:
            Dictionary with chat response
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call load_context first.")

        try:
            # Handle different input formats
            if isinstance(model_input, dict):
                messages_data = model_input.get("messages", [])
            elif isinstance(model_input, list):
                messages_data = model_input
            else:
                # Handle pandas DataFrame input
                if hasattr(model_input, "to_dict"):
                    messages_data = model_input.to_dict("records")
                else:
                    raise ValueError(f"Unsupported input type: {type(model_input)}")

            # Convert to ChatAgentMessage objects
            messages = []
            for msg in messages_data:
                if isinstance(msg, dict):
                    # Ensure role and content are not None
                    role = msg.get("role") or "user"  # Default to "user" if role is missing
                    content = msg.get("content")
                    # ChatAgentMessage requires content to not be empty
                    if content is None or content == "":
                        content = " "  # Use a single space for empty content
                    messages.append(
                        ChatAgentMessage(
                            role=role, content=content
                        )
                    )
                elif isinstance(msg, ChatAgentMessage):
                    messages.append(msg)

            # Get optional parameters
            context_data = None
            custom_inputs = None
            if isinstance(model_input, dict):
                context_data = model_input.get("context")
                custom_inputs = model_input.get("custom_inputs")

            # Process through agent
            response = self.agent.predict(messages, context_data, custom_inputs)

            # Format response
            if isinstance(response, ChatAgentResponse):
                return {
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in response.messages
                    ]
                }
            else:
                return {"response": str(response)}

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "error": str(e),
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}",
                    }
                ],
            }


# This is the entry point that MLflow will use
def _load_pyfunc(path):
    """Load function for MLflow.

    This function is called by MLflow when loading the model.

    Args:
        path: Path to model artifacts

    Returns:
        Model instance
    """
    # Look for config in the artifacts directory
    config_path = Path(path) / "config" / "agent_config.json"
    if not config_path.exists():
        # Try alternative location
        config_path = Path(path) / "agent_config.json"

    model = DSPyAgentModel(str(config_path))

    # Create a mock context to load the model
    class MockContext:
        def __init__(self, path):
            self.artifacts = {"config_path": str(config_path)}

    model.load_context(MockContext(path))

    return model