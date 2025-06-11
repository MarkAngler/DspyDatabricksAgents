"""Unit tests for MLflow model loader."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deployment.model_loader import (
    _load_pyfunc,
    _load_model,
    DSPyAgentModelWrapper,
)


class TestModelLoader:
    """Test MLflow model loader functionality."""

    def test_load_pyfunc_with_config_in_data_path(self):
        """Test loading model with config in data path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file in data path
            config_path = Path(temp_dir) / "agent_config.json"
            
            config_data = {
                "name": "test-agent",
                "version": "1.0.0",
                "dspy": {"lm_model": "test-model"},
                "modules": [],
                "workflow": []
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            # Mock DSPyAgentModel at the import location
            with patch("dspy_databricks_agents.deployment.mlflow_model.DSPyAgentModel") as mock_model_class:
                mock_instance = Mock()
                mock_model_class.return_value = mock_instance
                
                # Load model
                model = _load_pyfunc(temp_dir)
                
                # Verify
                assert model == mock_instance
                mock_model_class.assert_called_once_with(str(config_path))
                mock_instance.load_context.assert_called_once()


    def test_load_model_alias(self):
        """Test _load_model function calls _load_pyfunc."""
        with patch("dspy_databricks_agents.deployment.model_loader._load_pyfunc") as mock_load:
            mock_load.return_value = "test_model"
            
            result = _load_model("test_path")
            
            assert result == "test_model"
            mock_load.assert_called_once_with("test_path")

    def test_dspy_agent_model_wrapper_load_context(self):
        """Test DSPyAgentModelWrapper load_context method."""
        wrapper = DSPyAgentModelWrapper()
        
        # Create mock context
        mock_context = Mock()
        mock_context.artifacts = {"model_path": "/test/path"}
        
        # Mock _load_pyfunc
        with patch("dspy_databricks_agents.deployment.model_loader._load_pyfunc") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load context
            wrapper.load_context(mock_context)
            
            # Verify
            assert wrapper.model == mock_model
            mock_load.assert_called_once_with("/test/path")

    def test_dspy_agent_model_wrapper_predict(self):
        """Test DSPyAgentModelWrapper predict method."""
        wrapper = DSPyAgentModelWrapper()
        
        # Set up mock model
        mock_model = Mock()
        mock_model.predict.return_value = {"response": "test response"}
        wrapper.model = mock_model
        
        # Create mock context
        mock_context = Mock()
        
        # Test input
        model_input = {"messages": [{"role": "user", "content": "Hello"}]}
        
        # Predict
        result = wrapper.predict(mock_context, model_input)
        
        # Verify
        assert result == {"response": "test response"}
        mock_model.predict.assert_called_once_with(mock_context, model_input)

    def test_dspy_agent_model_wrapper_predict_without_model(self):
        """Test DSPyAgentModelWrapper predict fails without loaded model."""
        wrapper = DSPyAgentModelWrapper()
        
        # Model not loaded
        wrapper.model = None
        
        # Create mock context
        mock_context = Mock()
        
        # Test input
        model_input = {"messages": [{"role": "user", "content": "Hello"}]}
        
        # Should raise error
        with pytest.raises(RuntimeError, match="Model not loaded"):
            wrapper.predict(mock_context, model_input)

    def test_load_pyfunc_missing_config(self):
        """Test loading model with missing config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # No config file created
            
            # Try to load - should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="Config file not found"):
                _load_pyfunc(temp_dir)

    def test_dspy_agent_model_wrapper_load_context_no_model_path(self):
        """Test DSPyAgentModelWrapper with missing model_path in context."""
        wrapper = DSPyAgentModelWrapper()
        
        # Create mock context without model_path
        mock_context = Mock()
        mock_context.artifacts = {}
        
        # Mock _load_pyfunc
        with patch("dspy_databricks_agents.deployment.model_loader._load_pyfunc") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Load context - should use default path
            wrapper.load_context(mock_context)
            
            # Verify it used "." as default
            mock_load.assert_called_once_with(".")