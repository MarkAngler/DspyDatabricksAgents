"""Tests for Databricks LM integration."""

import os
import pytest
from unittest.mock import patch, MagicMock
import dspy

from dspy_databricks_agents.config.schema import AgentConfig, DSPyConfig
from dspy_databricks_agents.core.agent import DSPyDatabricksAgent


class TestDatabricksLMIntegration:
    """Test Databricks LM integration."""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(
                lm_model="databricks/databricks-meta-llama-3-1-70b-instruct",
                temperature=0.7,
                max_tokens=1000
            ),
            modules=[],
            workflow=[]
        )
    
    @pytest.fixture
    def legacy_agent_config(self):
        """Create test agent configuration with legacy format."""
        return AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(
                lm_model="databricks-meta-llama-3-1-70b-instruct",  # Legacy format
                temperature=0.7,
                max_tokens=1000
            ),
            modules=[],
            workflow=[]
        )
    
    def test_databricks_lm_initialization_with_env_vars(self, agent_config, monkeypatch):
        """Test Databricks LM initialization with environment variables."""
        # Set environment variables
        monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")
        monkeypatch.setenv("DSPY_USE_MOCK", "false")
        
        with patch('dspy.LM') as mock_lm:
            mock_lm_instance = MagicMock()
            mock_lm.return_value = mock_lm_instance
            
            # Create agent
            agent = DSPyDatabricksAgent(agent_config)
            
            # Verify DSPy LM was called with correct parameters
            mock_lm.assert_called_once_with(
                "databricks/databricks-meta-llama-3-1-70b-instruct",
                temperature=0.7,
                max_tokens=1000,
                api_key="test-token",
                api_base="https://test.databricks.com"
            )
    
    def test_databricks_lm_initialization_without_env_vars(self, agent_config):
        """Test Databricks LM initialization without environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('dspy.LM') as mock_lm:
                mock_lm_instance = MagicMock()
                mock_lm.return_value = mock_lm_instance
                
                # Create agent
                agent = DSPyDatabricksAgent(agent_config)
                
                # Verify DSPy LM was called without auth params
                mock_lm.assert_called_once_with(
                    "databricks/databricks-meta-llama-3-1-70b-instruct",
                    temperature=0.7,
                    max_tokens=1000
                )
    
    def test_legacy_model_name_conversion(self, legacy_agent_config):
        """Test that legacy model names are converted to new format."""
        with patch('dspy.LM') as mock_lm:
            mock_lm_instance = MagicMock()
            mock_lm.return_value = mock_lm_instance
            
            # Create agent with legacy config
            agent = DSPyDatabricksAgent(legacy_agent_config)
            
            # Verify model name was converted
            mock_lm.assert_called_once()
            call_args = mock_lm.call_args[0]
            assert call_args[0] == "databricks/databricks-meta-llama-3-1-70b-instruct"
    
    def test_databricks_lm_always_real(self, agent_config, monkeypatch):
        """Test that real LM is always used, never mocks."""
        # Even if someone tries to enable mock mode, we should still use real LM
        monkeypatch.setenv("DSPY_USE_MOCK", "true")
        
        with patch('dspy.LM') as mock_lm:
            mock_lm_instance = MagicMock()
            mock_lm.return_value = mock_lm_instance
            
            # Create agent
            agent = DSPyDatabricksAgent(agent_config)
            
            # Verify real LM was called despite DSPY_USE_MOCK being true
            mock_lm.assert_called_once()
    
    def test_error_propagation_on_lm_failure(self, agent_config, monkeypatch):
        """Test that errors are properly propagated when LM initialization fails."""
        
        with patch('dspy.LM') as mock_lm:
            # Make LM initialization fail
            mock_lm.side_effect = Exception("Connection failed")
            
            # Create agent - should raise
            with pytest.raises(Exception, match="Connection failed"):
                agent = DSPyDatabricksAgent(agent_config)
    
    def test_non_databricks_model_uses_dspy_lm(self):
        """Test that non-Databricks models use DSPy's native LM support."""
        config = AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(
                lm_model="openai/gpt-4",
                temperature=0.7,
                max_tokens=1000
            ),
            modules=[],
            workflow=[]
        )
        
        with patch('dspy.LM') as mock_lm:
            mock_lm_instance = MagicMock()
            mock_lm.return_value = mock_lm_instance
            
            # Create agent
            agent = DSPyDatabricksAgent(config)
            
            # Verify DSPy LM was called with the OpenAI model
            mock_lm.assert_called_once_with(
                "openai/gpt-4",
                temperature=0.7,
                max_tokens=1000
            )