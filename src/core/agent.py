"""DSPy-Databricks Agent implementation."""

import json
import logging
import os
import uuid
from typing import Dict

import dspy
import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ToolCall,
)

from config.parser import YAMLParser
from config.schema import AgentConfig
from core.modules import ModuleFactory
from core.workflow import WorkflowOrchestrator
from core.tools import ToolRegistry

# Configure logging
logger = logging.getLogger(__name__)


class Agent:
    """High-level agent class for easy usage."""

    def __init__(self, config: AgentConfig):
        """Initialize agent from configuration."""
        self.config = config
        self.chat_agent = DSPyDatabricksAgent(config)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Agent":
        """Create agent from YAML file."""
        parser = YAMLParser()
        config = parser.parse_file(yaml_path)
        return cls(config)

    def predict(
        self,
        messages,
        context=None,
        custom_inputs=None,
    ):
        """Process messages and return response."""
        return self.chat_agent.predict(messages, context, custom_inputs)

    def predict_stream(
        self,
        messages,
        context=None,
        custom_inputs=None,
    ):
        """Stream responses."""
        return self.chat_agent.predict_stream(messages, context, custom_inputs)


class DSPyDatabricksAgent(ChatAgent):
    """MLflow ChatAgent implementation for DSPy agents with enhanced capabilities."""

    def __init__(self, config: AgentConfig):
        """Initialize agent from configuration."""
        self.config = config

        # Ensure MLflow experiment is set to avoid default experiment warning
        from deployment.mlflow_utils import ensure_experiment_set

        ensure_experiment_set(agent_name=config.name)

        self.modules = self._build_modules()
        self.orchestrator = WorkflowOrchestrator(self.modules)
        self._initialize_dspy()

    def _initialize_dspy(self):
        """Initialize DSPy with configuration."""
        lm = None

        logger.debug(f"Initializing DSPy with model: {self.config.dspy.lm_model}")

        # Configure language model - ALWAYS use real LMs, never mocks
        if self.config.dspy.lm_model.startswith(
            "databricks/"
        ) or self.config.dspy.lm_model.startswith("databricks-"):
            # Use real Databricks LM
            logger.info("Using real Databricks LM via DSPy")
            lm = self._get_databricks_lm_real()
        else:
            # Other providers - use DSPy's native support
            logger.info(f"Using LM provider for model: {self.config.dspy.lm_model}")
            try:
                # Use DSPy's native LM support for all models
                kwargs = {}
                if (
                    hasattr(self.config.dspy, "temperature")
                    and self.config.dspy.temperature is not None
                ):
                    kwargs["temperature"] = self.config.dspy.temperature
                if (
                    hasattr(self.config.dspy, "max_tokens")
                    and self.config.dspy.max_tokens is not None
                ):
                    kwargs["max_tokens"] = self.config.dspy.max_tokens

                lm = dspy.LM(self.config.dspy.lm_model, **kwargs)
            except Exception as e:
                logger.error(
                    f"Failed to initialize LM {self.config.dspy.lm_model}: {e}"
                )
                raise

        if lm:
            dspy.settings.configure(lm=lm)
        else:
            raise ValueError(
                f"Failed to initialize language model: {self.config.dspy.lm_model}"
            )

    def _get_databricks_lm_real(self):
        """Get real Databricks language model using DSPy native support."""
        try:
            # Get authentication from environment
            api_key = os.getenv("DATABRICKS_TOKEN")
            api_base = os.getenv("DATABRICKS_HOST")

            # Security: Log presence of auth without exposing values
            if api_key:
                logger.debug("DATABRICKS_TOKEN is set")
            else:
                logger.debug("DATABRICKS_TOKEN not found in environment")

            if api_base:
                logger.debug(
                    f"DATABRICKS_HOST is set: {api_base[:20]}..."
                )  # Only log prefix
            else:
                logger.debug("DATABRICKS_HOST not found in environment")

            # Handle model name format
            model_name = self.config.dspy.lm_model

            # Convert legacy format if needed
            if model_name.startswith("databricks-"):
                # Replace only the first "databricks-" with "databricks/databricks-"
                model_name = model_name.replace(
                    "databricks-", "databricks/databricks-", 1
                )

            # Prepare kwargs
            kwargs = {}
            if (
                hasattr(self.config.dspy, "temperature")
                and self.config.dspy.temperature is not None
            ):
                kwargs["temperature"] = self.config.dspy.temperature
            if (
                hasattr(self.config.dspy, "max_tokens")
                and self.config.dspy.max_tokens is not None
            ):
                kwargs["max_tokens"] = self.config.dspy.max_tokens

            # Add authentication if available
            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                # For Databricks, ensure the api_base ends with /serving-endpoints
                if api_base and not api_base.endswith("/serving-endpoints"):
                    # Remove any trailing slashes first
                    api_base = api_base.rstrip("/")
                    # Add the serving-endpoints path
                    api_base = f"{api_base}/serving-endpoints"
                kwargs["api_base"] = api_base

            # Create and return the LM
            import dspy

            lm = dspy.LM(model_name, **kwargs)
            logger.info(f"Initialized Databricks LM: {model_name}")
            return lm

        except Exception as e:
            logger.error(f"Failed to initialize Databricks LM: {e}")
            raise

    def _build_modules(self) -> Dict[str, dspy.Module]:
        """Build all DSPy modules from configuration."""
        factory = ModuleFactory()
        modules = {}

        for module_config in self.config.modules:
            module = factory.create_module(module_config)
            modules[module_config.name] = module

        return modules

    def predict(
        self,
        messages,
        context=None,
        custom_inputs=None,
    ):
        """Process messages and return response."""
        with mlflow.start_span("agent_predict") as span:
            span.set_attributes(
                {
                    "agent_name": self.config.name,
                    "agent_version": self.config.version,
                    "num_messages": len(messages),
                }
            )

            # Prepare workflow input
            workflow_input = self._prepare_input(messages, context, custom_inputs)

            # Execute workflow
            with mlflow.start_span("workflow_execution"):
                result = self.orchestrator.execute(self.config.workflow, workflow_input)

            # Format response
            response = self._format_response(result)

            # Log metrics
            self._log_metrics(messages, response, span)

            return response

    def predict_stream(
        self,
        messages,
        context=None,
        custom_inputs=None,
    ):
        """Stream responses."""
        # Prepare input
        workflow_input = self._prepare_input(messages, context, custom_inputs)

        # For streaming, we need to identify which module generates the final response
        final_step = self.config.workflow[-1]

        # Initialize context object
        from core.workflow import WorkflowContext

        context_obj = WorkflowContext(workflow_input)

        # Execute workflow up to final step
        if len(self.config.workflow) > 1:
            for step in self.config.workflow[:-1]:
                if self.orchestrator._should_execute_step(step, context_obj):
                    output = self.orchestrator._execute_step(step, context_obj)
                    context_obj.set_output(step.step, output)

            # Update workflow input with intermediate results
            workflow_input.update(context_obj.step_outputs)

        # Stream final step
        module = self.modules[final_step.module]

        # Prepare final step inputs
        final_inputs = {}
        for key, value in final_step.inputs.items():
            final_inputs[key] = self.orchestrator._resolve_input_value(
                value, context_obj
            )

        # Check if module supports streaming
        if hasattr(module, "forward_stream"):
            stream_id = str(uuid.uuid4())

            for chunk in module.forward_stream(**final_inputs):
                yield ChatAgentChunk(
                    delta=ChatAgentMessage(
                        role="assistant", content=chunk, id=stream_id
                    )
                )
        else:
            # Fallback to non-streaming
            result = module(**final_inputs)
            response_text = self._extract_response_text(result)

            # Simulate streaming by yielding the whole response
            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    role="assistant", content=response_text, id=str(uuid.uuid4())
                )
            )

    def _prepare_input(self, messages, context, custom_inputs):
        """Prepare workflow input from chat messages."""
        # Extract last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg
                break

        if not user_message:
            raise ValueError("No user message found")

        # Build input
        workflow_input = {
            "query": user_message.content,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
            "conversation_history": self._format_history(messages[:-1]),
        }

        # Add context if available
        if context:
            workflow_input["conversation_id"] = context.conversation_id
            workflow_input["user_id"] = context.user_id

        # Add custom inputs
        if custom_inputs:
            workflow_input.update(custom_inputs)

        return workflow_input

    def _format_response(self, result):
        """Format workflow result as ChatAgentResponse."""
        # Find the response in the result
        response_content = None

        # Check common response fields
        for field in ["response", "answer", "output", "result"]:
            if field in result:
                response_content = result[field]
                break

        # Check step outputs (only if they look like step outputs)
        if response_content is None and isinstance(result, dict):
            # Check all values, not just specific step names
            for key, step_output in result.items():
                # Skip metadata and other non-step keys
                if key in ["metadata", "_meta", "config"]:
                    continue

                # Handle DSPy Prediction objects
                if (
                    hasattr(step_output, "__class__")
                    and step_output.__class__.__name__ == "Prediction"
                ):
                    # Extract fields from Prediction object
                    for field in ["response", "answer", "output", "result"]:
                        if hasattr(step_output, field):
                            response_content = getattr(step_output, field)
                            break
                elif isinstance(step_output, dict):
                    for field in ["response", "answer", "output"]:
                        if field in step_output:
                            response_content = step_output[field]
                            break
                elif isinstance(step_output, str):
                    response_content = step_output

                if response_content:
                    break

        if response_content is None:
            response_content = str(result)

        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant", content=response_content, id=str(uuid.uuid4())
                )
            ]
        )

    def _extract_response_text(self, result):
        """Extract response text from module result."""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            for field in ["response", "answer", "output", "text"]:
                if field in result:
                    return str(result[field])
        elif hasattr(result, "response"):
            return str(result.response)
        elif hasattr(result, "answer"):
            return str(result.answer)

        return str(result)

    def _format_history(self, messages):
        """Format conversation history as string."""
        history = []

        for msg in messages:
            role = msg.role.capitalize()
            content = msg.content
            history.append(f"{role}: {content}")

        return "\n".join(history)

    def _log_metrics(self, messages, response, span):
        """Log metrics for monitoring."""
        # Response metrics
        response_length = len(response.messages[0].content)

        # Log to MLflow
        mlflow.log_metrics(
            {
                "response_length": response_length,
                "input_messages": len(messages),
                "conversation_turns": len([m for m in messages if m.role == "user"]),
            }
        )

        # Add to span
        span.set_attributes({"response_length": response_length, "has_error": False})

    # Enhanced ChatAgent methods for tool support
    def predict_with_tools(
        self,
        messages,
        tools,
        context=None,
        custom_inputs=None,
    ):
        """Enhanced predict method with tool support.

        Args:
            messages: List of ChatAgentMessage objects
            tools: List of tool definitions
            context: Optional conversation context
            custom_inputs: Optional custom inputs

        Returns:
            ChatAgentResponse with model output and optional tool calls
        """
        # Check if we have a ReAct module that can use tools
        react_module = None
        for module_name, module in self.modules.items():
            if hasattr(module, "use_tools") or "react" in module_name.lower():
                react_module = module
                break

        if react_module and tools:
            # Add tools to custom inputs
            if custom_inputs is None:
                custom_inputs = {}
            custom_inputs["tools"] = tools

            # Process with tool support
            response = self.predict(messages, context, custom_inputs)

            # Check if response contains tool calls and enhance response
            content = response.messages[0].content
            if "Action:" in content:
                # Extract tool call from ReAct format
                lines = content.split("\n")
                tool_calls = []

                for i, line in enumerate(lines):
                    if line.strip().startswith("Action:"):
                        tool_name = line.split("Action:")[1].strip()
                        if i + 1 < len(lines) and lines[i + 1].strip().startswith(
                            "Action Input:"
                        ):
                            tool_input = lines[i + 1].split("Action Input:")[1].strip()

                            from mlflow.types.chat import Function

                            tool_call = ToolCall(
                                id=str(uuid.uuid4()),
                                function=Function(
                                    name=tool_name,
                                    arguments=json.dumps({"query": tool_input}),
                                ),
                            )
                            tool_calls.append(tool_call)

                # Create enhanced response with tool calls
                if tool_calls:
                    # Add tool calls to the response metadata
                    response.messages[0].tool_calls = tool_calls

            return response
        else:
            # Fallback to regular predict
            return self.predict(messages, context, custom_inputs)

    def get_tools(self):
        """Get available tools for this agent.

        Returns:
            List of tool definitions in OpenAI function format
        """
        tools = []

        # Check if any module supports tools (e.g., ReAct)
        for module_config in self.config.modules:
            if module_config.type.value == "react" and module_config.tools:
                # Add configured tools
                for tool_name in module_config.tools:
                    tool_func = ToolRegistry.get(tool_name)
                    if tool_func:
                        tools.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "description": getattr(
                                        tool_func, "description", f"{tool_name} tool"
                                    ),
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "The input query or expression for the tool",
                                            }
                                        },
                                        "required": ["query"],
                                    },
                                },
                            }
                        )

        return tools
