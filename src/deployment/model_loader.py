"""MLflow model loader for code-based deployment."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_pyfunc(data_path):
    """MLflow entry point for loading the model.
    
    This function is called by MLflow when loading the model.
    
    Args:
        data_path: Path to data directory containing model artifacts
        
    Returns:
        Model instance
    """
    # Import here to ensure all dependencies are available
    from deployment.mlflow_model import DSPyAgentModel
    
    # Look for config in the data directory
    config_path = Path(data_path) / "agent_config.json"
    
    logger.info(f"Loading model from config: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Create model instance
    model = DSPyAgentModel(str(config_path))
    
    # Create a mock context to load the model
    class MockContext:
        def __init__(self, path):
            self.artifacts = {"config_path": str(config_path)}
    
    # Load the model configuration
    model.load_context(MockContext(data_path))
    
    return model


class DSPyAgentModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for DSPy agent model with proper MLflow interface."""
    
    def __init__(self):
        """Initialize wrapper."""
        self.model = None
        
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model from context.
        
        Args:
            context: MLflow context containing artifacts
        """
        # Get the model path
        model_path = context.artifacts.get("model_path", ".")
        
        # Load using the _load_pyfunc function
        self.model = _load_pyfunc(model_path)
        logger.info("Model loaded successfully")
        
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
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_context first.")
            
        # Delegate to the actual model
        return self.model.predict(context, model_input)


# For MLflow to find the model
def _load_model(path):
    """Alternative entry point for MLflow.
    
    Args:
        path: Path to model artifacts
        
    Returns:
        Model instance
    """
    return _load_pyfunc(path)