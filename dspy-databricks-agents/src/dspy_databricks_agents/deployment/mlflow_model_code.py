"""MLflow model code for models-from-code deployment pattern."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

# Import the necessary modules
from dspy_databricks_agents.config.schema import AgentConfig
from dspy_databricks_agents.core.agent import Agent
from dspy_databricks_agents.deployment.mlflow_model import DSPyAgentModel


class DSPyAgentModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for DSPy Agent Model that handles MLflow integration."""
    
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model artifacts and initialize agent.
        
        Args:
            context: MLflow context containing artifacts
        """
        # Look for config in the artifacts directory
        if hasattr(context, 'artifacts') and 'config' in context.artifacts:
            config_dir = Path(context.artifacts['config'])
            config_path = config_dir / 'agent_config.json'
        else:
            # Fallback to model path
            config_path = Path(context.model_path) / "config" / "agent_config.json"
            if not config_path.exists():
                config_path = Path(context.model_path) / "agent_config.json"
        
        # Create the actual model
        self.model = DSPyAgentModel(str(config_path))
        
        # Initialize it with context
        self.model.load_context(context)
    
    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process chat messages and return response.
        
        Args:
            context: MLflow context
            model_input: Input data containing messages
            params: Optional parameters for prediction
            
        Returns:
            Dictionary with chat response
        """
        return self.model.predict(context, model_input)


# Create an instance of the wrapper
_model = DSPyAgentModelWrapper()

# Set the model for MLflow (required for models-from-code)
mlflow.models.set_model(_model)