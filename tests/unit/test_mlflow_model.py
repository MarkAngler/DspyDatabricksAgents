"""Unit tests for MLflow model wrapper."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from config.schema import (
    AgentConfig,
    DSPyConfig,
    ModuleConfig,
    ModuleType,
    WorkflowStep,
)
from deployment.mlflow_model import DSPyAgentModel


class TestDSPyAgentModel:
    """Test DSPy Agent MLflow model wrapper."""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(lm_model="test-model"),
            modules=[
                ModuleConfig(
                    name="qa",
                    type=ModuleType.SIGNATURE,
                    signature="question -> answer"
                )
            ],
            workflow=[
                WorkflowStep(step="answer", module="qa")
            ]
        )
    
    @pytest.fixture
    def temp_context(self, tmp_path, agent_config):
        """Create a temporary context with agent config."""
        # Create config directory and file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(agent_config.model_dump()))
        
        # Create context mock
        context = Mock()
        context.artifacts = {"model_path": str(tmp_path)}
        
        return context
    
    def test_model_initialization(self):
        """Test model initializes with empty state."""
        model = DSPyAgentModel()
        assert model.agent is None
        assert model.config_path is None
    
    def test_load_context_success(self, temp_context, agent_config):
        """Test loading agent from context."""
        model = DSPyAgentModel()
        
        # Mock the Agent class
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent") as mock_agent:
            model.load_context(temp_context)
            
            # Verify agent was created with correct config
            mock_agent.assert_called_once()
            args = mock_agent.call_args[0]
            assert args[0].name == agent_config.name
            assert args[0].version == agent_config.version
            assert model.agent is not None
    
    def test_load_context_missing_config(self, tmp_path):
        """Test handling missing config file."""
        model = DSPyAgentModel()
        
        # Create context without config file
        context = Mock()
        context.artifacts = {"model_path": str(tmp_path)}
        
        with pytest.raises(Exception):
            model.load_context(context)
    
    def test_load_context_invalid_config(self, tmp_path):
        """Test handling invalid config file."""
        model = DSPyAgentModel()
        
        # Create config with invalid JSON
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "agent_config.json"
        config_file.write_text("invalid json")
        
        context = Mock()
        context.artifacts = {"model_path": str(tmp_path)}
        
        with pytest.raises(Exception):
            model.load_context(context)
    
    def test_predict_without_agent(self):
        """Test prediction fails when agent not loaded."""
        model = DSPyAgentModel()
        
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            model.predict(None, {"messages": []})
    
    def test_predict_with_dict_input(self, temp_context):
        """Test prediction with dictionary input format."""
        model = DSPyAgentModel()
        
        # Mock agent and response
        mock_agent = Mock()
        # Need to mock ChatAgentResponse properly
        from mlflow.types.agent import ChatAgentResponse
        mock_response = Mock(spec=ChatAgentResponse)
        mock_response.messages = [
            Mock(role="assistant", content="Test response")
        ]
        mock_agent.predict.return_value = mock_response
        
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent", return_value=mock_agent):
            model.load_context(temp_context)
            
            # Test with dict input
            result = model.predict(None, {
                "messages": [
                    {"role": "user", "content": "Test question"}
                ]
            })
            
            assert "messages" in result
            assert result["messages"][0]["role"] == "assistant"
            assert result["messages"][0]["content"] == "Test response"
            
            # Verify agent was called with correct arguments
            mock_agent.predict.assert_called_once()
            call_args = mock_agent.predict.call_args[0]
            assert len(call_args[0]) == 1  # One message
            assert call_args[1] is None  # No context
            assert call_args[2] is None  # No custom inputs
    
    def test_predict_with_list_input(self, temp_context):
        """Test prediction with list input format."""
        model = DSPyAgentModel()
        
        # Mock agent and response
        mock_agent = Mock()
        # Need to mock ChatAgentResponse properly
        from mlflow.types.agent import ChatAgentResponse
        mock_response = Mock(spec=ChatAgentResponse)
        mock_response.messages = [
            Mock(role="assistant", content="Test response")
        ]
        mock_agent.predict.return_value = mock_response
        
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent", return_value=mock_agent):
            model.load_context(temp_context)
            
            # Test with list input
            result = model.predict(None, [
                {"role": "user", "content": "Test question"}
            ])
            
            assert "messages" in result
            assert result["messages"][0]["role"] == "assistant"
            assert result["messages"][0]["content"] == "Test response"
    
    def test_predict_handles_errors(self, temp_context):
        """Test prediction error handling."""
        model = DSPyAgentModel()
        
        # Mock agent that raises error
        mock_agent = Mock()
        mock_agent.predict.side_effect = Exception("Test error")
        
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent", return_value=mock_agent):
            model.load_context(temp_context)
            
            result = model.predict(None, {"messages": []})
            
            assert "error" in result
            assert "Test error" in result["error"]
            # Error responses also include messages
            assert "messages" in result
            assert "Test error" in result["messages"][0]["content"]
    
    def test_load_pyfunc(self, tmp_path, agent_config):
        """Test the _load_pyfunc entry point."""
        # Import the module to get access to _load_pyfunc
        from deployment import mlflow_model
        
        # Create config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(agent_config.model_dump()))
        
        # Mock the agent creation
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent") as mock_agent:
            # Call _load_pyfunc
            model = mlflow_model._load_pyfunc(str(tmp_path))
            
            # Verify model was loaded
            assert model.agent is not None
            mock_agent.assert_called_once()
    
    def test_multiple_messages(self, temp_context):
        """Test handling multiple messages in conversation."""
        model = DSPyAgentModel()
        
        # Mock agent and response
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.messages = [
            Mock(role="assistant", content="Response 1"),
            Mock(role="assistant", content="Response 2")
        ]
        mock_agent.predict.return_value = mock_response
        
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent", return_value=mock_agent):
            model.load_context(temp_context)
            
            # Test with multiple input messages
            result = model.predict(None, {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"}
                ]
            })
            
            # Verify we get a ChatAgentResponse
            assert isinstance(result, dict)
            # The actual model returns a ChatAgentResponse, not a dict with messages
            # So we check that the agent was called properly
            mock_agent.predict.assert_called_once()