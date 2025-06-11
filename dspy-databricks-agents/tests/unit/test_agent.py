"""Unit tests for DSPy-Databricks Agent implementation."""

import os
import uuid
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

from dspy_databricks_agents.config.schema import (
    AgentConfig,
    DSPyConfig,
    ModuleConfig,
    ModuleType,
    WorkflowStep,
)
from dspy_databricks_agents.core.agent import Agent, DSPyDatabricksAgent


class TestAgent:
    """Test the high-level Agent class."""

    def test_agent_from_yaml(self, tmp_path):
        """Test creating agent from YAML file."""
        yaml_content = """
agent:
  name: test-agent
  version: 1.0.0
  
  dspy:
    lm_model: gpt-4
    temperature: 0.7
    
  modules:
    - name: responder
      type: signature
      signature: "query -> response"
      
  workflow:
    - step: respond
      module: responder
      inputs:
        query: "$input.query"
"""
        yaml_file = tmp_path / "test_agent.yaml"
        yaml_file.write_text(yaml_content)
        
        agent = Agent.from_yaml(str(yaml_file))
        
        assert isinstance(agent, Agent)
        assert agent.config.name == "test-agent"
        assert isinstance(agent.chat_agent, DSPyDatabricksAgent)

    def test_agent_from_config(self):
        """Test creating agent from config object."""
        config = AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(lm_model="gpt-4"),
            modules=[
                ModuleConfig(
                    name="test",
                    type=ModuleType.SIGNATURE,
                    signature="a -> b"
                )
            ],
            workflow=[
                WorkflowStep(step="test", module="test")
            ]
        )
        
        agent = Agent(config)
        
        assert agent.config == config
        assert isinstance(agent.chat_agent, DSPyDatabricksAgent)

    def test_agent_predict(self):
        """Test agent predict method."""
        config = AgentConfig(
            name="test",
            version="1.0",
            dspy=DSPyConfig(lm_model="test"),
            modules=[
                ModuleConfig(name="test", type=ModuleType.SIGNATURE, signature="q -> a")
            ],
            workflow=[WorkflowStep(step="test", module="test")]
        )
        
        agent = Agent(config)
        
        # Mock the chat agent
        agent.chat_agent.predict = Mock(
            return_value=ChatAgentResponse(
                messages=[ChatAgentMessage(role="assistant", content="Test response", id=str(uuid.uuid4()))]
            )
        )
        
        messages = [ChatAgentMessage(role="user", content="Test")]
        response = agent.predict(messages)
        
        assert isinstance(response, ChatAgentResponse)
        agent.chat_agent.predict.assert_called_once_with(messages, None, None)

    def test_agent_predict_stream(self):
        """Test agent streaming predict method."""
        config = AgentConfig(
            name="test",
            version="1.0",
            dspy=DSPyConfig(lm_model="test"),
            modules=[
                ModuleConfig(name="test", type=ModuleType.SIGNATURE, signature="q -> a")
            ],
            workflow=[WorkflowStep(step="test", module="test")]
        )
        
        agent = Agent(config)
        
        # Mock streaming
        def mock_stream(*args, **kwargs):
            yield {"delta": {"content": "Test"}}
            yield {"delta": {"content": " response"}}
        
        agent.chat_agent.predict_stream = Mock(side_effect=mock_stream)
        
        messages = [ChatAgentMessage(role="user", content="Test")]
        chunks = list(agent.predict_stream(messages))
        
        assert len(chunks) == 2
        agent.chat_agent.predict_stream.assert_called_once()


class TestDSPyDatabricksAgent:
    """Test the MLflow ChatAgent implementation."""

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(
                lm_model="test-model",
                temperature=0.7,
                max_tokens=100,
            ),
            modules=[
                ModuleConfig(
                    name="classifier",
                    type=ModuleType.SIGNATURE,
                    signature="text -> category, confidence",
                ),
                ModuleConfig(
                    name="generator",
                    type=ModuleType.CHAIN_OF_THOUGHT,
                    signature="query, category -> response",
                ),
            ],
            workflow=[
                WorkflowStep(
                    step="classify",
                    module="classifier",
                    inputs={"text": "$input.query"},
                ),
                WorkflowStep(
                    step="generate",
                    module="generator",
                    inputs={
                        "query": "$input.query",
                        "category": "$classify.category",
                    },
                ),
            ],
        )

    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_agent_initialization(self, mock_factory, mock_dspy, agent_config):
        """Test agent initialization."""
        # Mock module creation
        mock_classifier = Mock()
        mock_generator = Mock()
        mock_factory.return_value.create_module.side_effect = [
            mock_classifier,
            mock_generator,
        ]
        
        agent = DSPyDatabricksAgent(agent_config)
        
        assert agent.config == agent_config
        assert len(agent.modules) == 2
        assert "classifier" in agent.modules
        assert "generator" in agent.modules
        
        # DSPy configuration should be called with a mock LM
        # This ensures DSPy is always configured to avoid "No LM is loaded" errors
        mock_dspy.settings.configure.assert_called_once()
        # Check that it was called with an lm argument
        call_args = mock_dspy.settings.configure.call_args
        assert 'lm' in call_args.kwargs
        assert call_args.kwargs['lm'] is not None

    @patch('mlflow.log_metrics')
    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    @patch('mlflow.start_span')
    def test_predict_simple(self, mock_span, mock_factory, mock_dspy, mock_log_metrics, agent_config):
        """Test simple predict functionality."""
        # Setup mocks
        mock_classifier = Mock(return_value={"category": "technical", "confidence": 0.9})
        mock_generator = Mock(return_value={"response": "Generated response"})
        mock_factory.return_value.create_module.side_effect = [
            mock_classifier,
            mock_generator,
        ]
        
        agent = DSPyDatabricksAgent(agent_config)
        
        # Create test messages
        messages = [
            ChatAgentMessage(role="user", content="How do I fix this error?")
        ]
        
        # Test predict
        response = agent.predict(messages)
        
        # Verify response
        assert isinstance(response, ChatAgentResponse)
        assert len(response.messages) == 1
        assert response.messages[0].role == "assistant"
        assert response.messages[0].content == "Generated response"
        assert response.messages[0].id is not None
        
        # Verify workflow execution
        mock_classifier.assert_called_once()
        mock_generator.assert_called_once()

    @patch('mlflow.log_metrics')
    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_predict_with_context(self, mock_factory, mock_dspy, mock_log_metrics, agent_config):
        """Test predict with conversation context."""
        # Setup mocks
        mock_classifier = Mock(return_value={"category": "general"})
        mock_generator = Mock(return_value={"response": "Context-aware response"})
        mock_factory.return_value.create_module.side_effect = [
            mock_classifier,
            mock_generator,
        ]
        
        agent = DSPyDatabricksAgent(agent_config)
        
        # Create messages with history
        messages = [
            ChatAgentMessage(role="user", content="Hello"),
            ChatAgentMessage(role="assistant", content="Hi there!"),
            ChatAgentMessage(role="user", content="What's the weather?"),
        ]
        
        context = ChatContext(
            conversation_id="test-conv-123",
            user_id="user-456"
        )
        
        # Test predict
        response = agent.predict(messages, context)
        
        # Verify the workflow received correct query
        call_args = mock_classifier.call_args[1]
        assert call_args["text"] == "What's the weather?"  # Last user message
        
        # Generator should receive conversation context
        gen_args = mock_generator.call_args[1]
        assert gen_args["query"] == "What's the weather?"

    @patch('mlflow.log_metrics')
    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_predict_with_custom_inputs(self, mock_factory, mock_dspy, mock_log_metrics, agent_config):
        """Test predict with custom inputs."""
        # Setup mocks
        mock_modules = {
            "classifier": Mock(return_value={"category": "custom"}),
            "generator": Mock(return_value={"response": "Custom response"}),
        }
        
        def create_module(config):
            return mock_modules[config.name]
        
        mock_factory.return_value.create_module.side_effect = create_module
        
        agent = DSPyDatabricksAgent(agent_config)
        
        messages = [ChatAgentMessage(role="user", content="Test")]
        custom_inputs = {
            "temperature": 0.5,
            "max_length": 200,
            "custom_param": "value"
        }
        
        response = agent.predict(messages, custom_inputs=custom_inputs)
        
        # Verify custom inputs were passed to workflow
        assert isinstance(response, ChatAgentResponse)

    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_predict_stream(self, mock_factory, mock_dspy, agent_config):
        """Test streaming predict functionality."""
        # Setup streaming module
        def stream_response(**kwargs):
            for chunk in ["This ", "is ", "streaming ", "response"]:
                yield chunk
        
        mock_classifier = Mock(return_value={"category": "stream"})
        mock_generator = Mock()
        mock_generator.forward_stream = Mock(side_effect=stream_response)
        
        mock_factory.return_value.create_module.side_effect = [
            mock_classifier,
            mock_generator,
        ]
        
        agent = DSPyDatabricksAgent(agent_config)
        
        messages = [ChatAgentMessage(role="user", content="Stream test")]
        
        # Collect streamed chunks
        chunks = list(agent.predict_stream(messages))
        
        assert len(chunks) == 4
        # Verify chunks have correct structure
        for chunk in chunks:
            assert hasattr(chunk, 'delta')
            assert chunk.delta.role == "assistant"
            assert chunk.delta.content in ["This ", "is ", "streaming ", "response"]

    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_error_handling(self, mock_factory, mock_dspy, agent_config):
        """Test error handling in predict."""
        # Make classifier raise an error
        mock_classifier = Mock(side_effect=RuntimeError("Classification failed"))
        mock_generator = Mock()
        
        mock_factory.return_value.create_module.side_effect = [
            mock_classifier,
            mock_generator,
        ]
        
        agent = DSPyDatabricksAgent(agent_config)
        
        messages = [ChatAgentMessage(role="user", content="Error test")]
        
        # Should propagate the error
        with pytest.raises(RuntimeError, match="Classification failed"):
            agent.predict(messages)

    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_prepare_input_formats_correctly(self, mock_factory, mock_dspy, agent_config):
        """Test that input preparation formats data correctly."""
        mock_factory.return_value.create_module.return_value = Mock(
            return_value={"response": "test"}
        )
        
        agent = DSPyDatabricksAgent(agent_config)
        
        messages = [
            ChatAgentMessage(role="system", content="You are helpful"),
            ChatAgentMessage(role="user", content="Question 1"),
            ChatAgentMessage(role="assistant", content="Answer 1"),
            ChatAgentMessage(role="user", content="Question 2"),
        ]
        
        context = ChatContext(conversation_id="123", user_id="456")
        custom_inputs = {"param": "value"}
        
        prepared = agent._prepare_input(messages, context, custom_inputs)
        
        assert prepared["query"] == "Question 2"  # Last user message
        assert len(prepared["messages"]) == 4
        assert prepared["conversation_id"] == "123"
        assert prepared["user_id"] == "456"
        assert prepared["param"] == "value"
        assert "You are helpful" in prepared["conversation_history"]

    @patch('dspy_databricks_agents.core.agent.dspy')
    @patch('dspy_databricks_agents.core.agent.ModuleFactory')
    def test_format_response_finds_output(self, mock_factory, mock_dspy, agent_config):
        """Test response formatting logic."""
        mock_factory.return_value.create_module.return_value = Mock()
        
        agent = DSPyDatabricksAgent(agent_config)
        
        # Test various result formats
        test_cases = [
            {"response": "Direct response field"},
            {"answer": "Answer field"},
            {"output": "Output field"},
            {"result": "Result field"},
            {"step1": {"response": "Nested response"}},
            {"step1": "String output"},
            {"other": "Fallback to string representation"},
        ]
        
        expected_responses = [
            "Direct response field",
            "Answer field",
            "Output field",
            "Result field",
            "Nested response",
            "String output",
            str({"other": "Fallback to string representation"}),
        ]
        
        for result, expected in zip(test_cases, expected_responses):
            response = agent._format_response(result)
            assert isinstance(response, ChatAgentResponse)
            assert response.messages[0].content == expected
            assert response.messages[0].role == "assistant"
            assert response.messages[0].id is not None