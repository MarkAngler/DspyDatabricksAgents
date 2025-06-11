"""Integration tests for MLflow deployment and serving."""

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import pytest
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

from config.schema import (
    AgentConfig,
    DSPyConfig,
    ModuleConfig,
    ModuleType,
    WorkflowStep,
)
from core.agent import Agent, DSPyDatabricksAgent
from deployment.mlflow_model import DSPyAgentModel
from deployment.model_signature import (
    create_chat_model_signature,
    create_tool_enabled_signature,
    get_signature_for_config,
)


class TestMLFlowIntegration:
    """Test MLflow integration functionality."""
    
    @pytest.fixture
    def simple_agent_config(self):
        """Create a simple agent configuration."""
        return AgentConfig(
            name="test-mlflow-agent",
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
                WorkflowStep(
                    step="answer",
                    module="qa",
                    inputs={"question": "$input.query"}
                )
            ]
        )
    
    @pytest.fixture
    def tool_enabled_config(self):
        """Create a tool-enabled agent configuration."""
        return AgentConfig(
            name="tool-agent",
            version="1.0.0",
            dspy=DSPyConfig(lm_model="test-model"),
            modules=[
                ModuleConfig(
                    name="react_agent",
                    type=ModuleType.REACT,
                    signature="question -> answer",
                    tools=["calculator", "web_search"]
                )
            ],
            workflow=[
                WorkflowStep(
                    step="process",
                    module="react_agent",
                    inputs={"question": "$input.query"}
                )
            ]
        )
    
    def test_enhanced_predict_interface(self, simple_agent_config):
        """Test enhanced predict interface implementation."""
        agent = DSPyDatabricksAgent(simple_agent_config)
        
        # Mock the orchestrator execute method
        with patch.object(agent.orchestrator, 'execute') as mock_execute:
            mock_execute.return_value = {"answer": "Test answer"}
            
            # Test predict method
            messages = [ChatAgentMessage(role="user", content="What is AI?")]
            response = agent.predict(messages)
            
            assert isinstance(response, ChatAgentResponse)
            assert response.messages[0].role == "assistant"
            assert response.messages[0].content == "Test answer"
    
    def test_predict_with_tools(self, tool_enabled_config):
        """Test predict with tools functionality."""
        agent = DSPyDatabricksAgent(tool_enabled_config)
        
        # Mock the orchestrator for tool execution
        with patch.object(agent.orchestrator, 'execute') as mock_execute:
            # Mock ReAct output with tool usage
            mock_execute.return_value = {
                "answer": "Let me calculate that.\nAction: calculator\nAction Input: 25 * 4\nObservation: 100\nThe answer is 100."
            }
            
            # Define tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Performs calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            }
                        }
                    }
                }
            ]
            
            # Test predict with tools
            messages = [ChatAgentMessage(role="user", content="What is 25 * 4?")]
            response = agent.predict_with_tools(messages, tools)
            
            assert isinstance(response, ChatAgentResponse)
            assert response.messages[0].role == "assistant"
            assert "Action: calculator" in response.messages[0].content
            
            # Check if tool calls were attached (if ToolCall has tool_calls attribute)
            if hasattr(response.messages[0], 'tool_calls'):
                assert len(response.messages[0].tool_calls) > 0
                tool_call = response.messages[0].tool_calls[0]
                assert tool_call.function.name == "calculator"
    
    def test_get_tools(self, tool_enabled_config):
        """Test tool discovery."""
        agent = DSPyDatabricksAgent(tool_enabled_config)
        
        tools = agent.get_tools()
        
        assert len(tools) == 2
        tool_names = [t["function"]["name"] for t in tools]
        assert "calculator" in tool_names
        assert "web_search" in tool_names
    
    def test_multi_turn_conversation(self, simple_agent_config):
        """Test multi-turn conversation handling."""
        agent = DSPyDatabricksAgent(simple_agent_config)
        
        # Mock the orchestrator
        with patch.object(agent.orchestrator, 'execute') as mock_execute:
            # Mock should see conversation history
            def execute_with_history(workflow, inputs):
                # Check that conversation history is passed
                if "conversation_history" in inputs and inputs["conversation_history"]:
                    return {"answer": "Based on our conversation, the answer is 42"}
                return {"answer": "Default answer"}
            
            mock_execute.side_effect = execute_with_history
            
            # Multi-turn conversation
            messages = [
                ChatAgentMessage(role="user", content="Let's talk about numbers"),
                ChatAgentMessage(role="assistant", content="Sure, what about numbers?"),
                ChatAgentMessage(role="user", content="What's the answer?")
            ]
            
            context = ChatContext(
                conversation_id="test-conv-123",
                user_id="user-456"
            )
            
            response = agent.predict(messages, context)
            
            assert isinstance(response, ChatAgentResponse)
            assert "Based on our conversation" in response.messages[0].content
    
    def test_mlflow_model_save_and_load(self, simple_agent_config, tmp_path):
        """Test saving and loading agent as MLflow model."""
        # Create agent
        agent = Agent(simple_agent_config)
        
        # Save model
        model_path = tmp_path / "model"
        
        # Create config directory
        config_dir = model_path / "config"
        config_dir.mkdir(parents=True)
        
        # Save config
        config_file = config_dir / "agent_config.json"
        config_file.write_text(json.dumps(simple_agent_config.model_dump()))
        
        # Test loading
        model = DSPyAgentModel()
        
        # Create mock context
        context = Mock()
        context.artifacts = {"model_path": str(model_path)}
        
        # Load model
        with patch("dspy_databricks_agents.deployment.mlflow_model.Agent"):
            model.load_context(context)
            assert model.agent is not None
    
    def test_model_signature_creation(self, simple_agent_config, tool_enabled_config):
        """Test model signature creation."""
        # Test simple chat signature
        signature = create_chat_model_signature()
        assert signature is not None
        
        # Test tool-enabled signature
        tool_signature = create_tool_enabled_signature()
        assert tool_signature is not None
        
        # Test signature selection
        simple_sig = get_signature_for_config(simple_agent_config)
        assert simple_sig is not None
        
        tool_sig = get_signature_for_config(tool_enabled_config)
        assert tool_sig is not None
    
    @pytest.mark.skipif(
        not os.environ.get("MLFLOW_TRACKING_URI"),
        reason="MLflow tracking URI not configured"
    )
    def test_end_to_end_mlflow_deployment(self, simple_agent_config, tmp_path):
        """Test end-to-end MLflow deployment (requires MLflow server)."""
        import mlflow.pyfunc
        
        # Set up MLflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("dspy-agent-tests")
        
        with mlflow.start_run():
            # Log agent configuration
            mlflow.log_param("agent_name", simple_agent_config.name)
            mlflow.log_param("agent_version", simple_agent_config.version)
            
            # Create model artifacts
            artifacts_dir = tmp_path / "artifacts"
            config_dir = artifacts_dir / "config"
            config_dir.mkdir(parents=True)
            
            config_file = config_dir / "agent_config.json"
            config_file.write_text(json.dumps(simple_agent_config.model_dump()))
            
            # Get model code path
            model_code_path = Path(__file__).parent.parent.parent / "src" / "dspy_databricks_agents" / "deployment" / "mlflow_model_code.py"
            
            # Log model
            signature = get_signature_for_config(simple_agent_config)
            
            model_info = mlflow.pyfunc.log_model(
                artifact_path="agent",
                python_model=str(model_code_path),
                artifacts={"config_path": str(config_file)},
                signature=signature,
                input_example={"messages": [{"role": "user", "content": "Hello"}]}
            )
            
            # Load and test model
            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            
            # Test prediction
            test_input = {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"}
                ]
            }
            
            # Mock the agent's predict method
            with patch.object(loaded_model._model_impl.python_model.agent, 'predict') as mock_predict:
                mock_predict.return_value = Mock(
                    messages=[Mock(role="assistant", content="ML is a subset of AI")]
                )
                
                result = loaded_model.predict(test_input)
                
                assert "messages" in result
                assert result["messages"][0]["role"] == "assistant"
    
    def test_streaming_support(self, simple_agent_config):
        """Test streaming functionality."""
        agent = DSPyDatabricksAgent(simple_agent_config)
        
        # Mock the module in the modules dict
        mock_qa = Mock()
        mock_qa.forward_stream = Mock(return_value=iter(["This ", "is ", "streaming"]))
        
        # Replace the actual module with our mock
        agent.modules["qa"] = mock_qa
        
        # Also need to mock the orchestrator for non-final steps
        with patch.object(agent.orchestrator, '_should_execute_step', return_value=True):
            with patch.object(agent.orchestrator, '_resolve_input_value', return_value="Stream test"):
                messages = [ChatAgentMessage(role="user", content="Stream test")]
                
                chunks = list(agent.predict_stream(messages))
                
                assert len(chunks) == 3
                contents = [chunk.delta.content for chunk in chunks]
                assert contents == ["This ", "is ", "streaming"]
    
    def test_error_handling_and_recovery(self, simple_agent_config):
        """Test error handling in chat interface."""
        agent = DSPyDatabricksAgent(simple_agent_config)
        
        # Test with invalid messages
        with pytest.raises(ValueError, match="No user message found"):
            agent.predict([ChatAgentMessage(role="system", content="System prompt")])
        
        # Test orchestrator errors
        with patch.object(agent.orchestrator, 'execute') as mock_execute:
            mock_execute.side_effect = RuntimeError("Orchestrator failed")
            
            messages = [ChatAgentMessage(role="user", content="Test")]
            
            with pytest.raises(RuntimeError, match="Orchestrator failed"):
                agent.predict(messages)
    
    def test_conversation_context_persistence(self, simple_agent_config):
        """Test conversation context handling."""
        agent = DSPyDatabricksAgent(simple_agent_config)
        
        # Mock the workflow orchestrator
        with patch.object(agent.orchestrator, 'execute') as mock_execute:
            mock_execute.return_value = {"answer": "Context-aware response"}
            
            messages = [
                ChatAgentMessage(role="user", content="Remember A=1"),
                ChatAgentMessage(role="assistant", content="OK, I'll remember A=1"),
                ChatAgentMessage(role="user", content="What is A?")
            ]
            
            context = ChatContext(
                conversation_id="test-123",
                user_id="user-456"
            )
            
            response = agent.predict(messages, context)
            
            # Verify context was passed to workflow
            call_args = mock_execute.call_args[0][1]
            assert call_args["conversation_id"] == "test-123"
            assert call_args["user_id"] == "user-456"
            assert len(call_args["messages"]) == 3
            assert "User: Remember A=1" in call_args["conversation_history"]