"""Unit tests for configuration schema."""

import pytest
from pydantic import ValidationError

from dspy_databricks_agents.config.schema import (
    AgentConfig,
    DSPyConfig,
    ModuleConfig,
    ModuleType,
    OptimizerConfig,
    OptimizerType,
    VectorStoreConfig,
    WorkflowStep,
    DeploymentConfig,
)


class TestDSPyConfig:
    """Test DSPy configuration schema."""

    def test_valid_dspy_config(self):
        """Test valid DSPy configuration."""
        config = DSPyConfig(
            lm_model="databricks-dbrx-instruct",
            temperature=0.7,
            max_tokens=1000,
        )
        
        assert config.lm_model == "databricks-dbrx-instruct"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.optimizer is None

    def test_dspy_config_with_optimizer(self):
        """Test DSPy config with optimizer."""
        config = DSPyConfig(
            lm_model="gpt-4",
            optimizer=OptimizerConfig(
                type=OptimizerType.BOOTSTRAP_FEW_SHOT,
                metric="accuracy",
                num_candidates=10,
            ),
        )
        
        assert config.optimizer.type == OptimizerType.BOOTSTRAP_FEW_SHOT
        assert config.optimizer.metric == "accuracy"
        assert config.optimizer.num_candidates == 10

    def test_temperature_validation(self):
        """Test temperature bounds validation."""
        # Valid temperatures
        DSPyConfig(lm_model="test", temperature=0.0)
        DSPyConfig(lm_model="test", temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            DSPyConfig(lm_model="test", temperature=-0.1)
        
        with pytest.raises(ValidationError):
            DSPyConfig(lm_model="test", temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max tokens validation."""
        # Valid max_tokens
        DSPyConfig(lm_model="test", max_tokens=1)
        DSPyConfig(lm_model="test", max_tokens=8192)
        
        # Invalid max_tokens
        with pytest.raises(ValidationError):
            DSPyConfig(lm_model="test", max_tokens=0)
        
        with pytest.raises(ValidationError):
            DSPyConfig(lm_model="test", max_tokens=8193)


class TestModuleConfig:
    """Test module configuration schema."""

    def test_signature_module_config(self):
        """Test signature module configuration."""
        config = ModuleConfig(
            name="classifier",
            type=ModuleType.SIGNATURE,
            signature="text -> category",
        )
        
        assert config.name == "classifier"
        assert config.type == ModuleType.SIGNATURE
        assert config.signature == "text -> category"

    def test_retriever_module_config(self):
        """Test retriever module configuration."""
        config = ModuleConfig(
            name="retriever",
            type=ModuleType.RETRIEVER,
            vector_store=VectorStoreConfig(
                catalog="ml",
                schema_name="vectors",
                index="knowledge_base",
                k=10,
            ),
        )
        
        assert config.name == "retriever"
        assert config.type == ModuleType.RETRIEVER
        assert config.vector_store.catalog == "ml"
        assert config.vector_store.k == 10

    def test_custom_module_config(self):
        """Test custom module configuration."""
        config = ModuleConfig(
            name="custom",
            type=ModuleType.CUSTOM,
            custom_class="MyCustomModule",
            config={"param1": "value1"},
        )
        
        assert config.custom_class == "MyCustomModule"
        assert config.config["param1"] == "value1"


class TestWorkflowStep:
    """Test workflow step configuration."""

    def test_simple_workflow_step(self):
        """Test simple workflow step."""
        step = WorkflowStep(
            step="classify",
            module="classifier",
        )
        
        assert step.step == "classify"
        assert step.module == "classifier"
        assert step.condition is None
        assert step.inputs == {}

    def test_workflow_step_with_condition(self):
        """Test workflow step with condition."""
        step = WorkflowStep(
            step="retrieve",
            module="retriever",
            condition="$classify.confidence > 0.8",
            inputs={"query": "$input.text"},
        )
        
        assert step.condition == "$classify.confidence > 0.8"
        assert step.inputs["query"] == "$input.text"

    def test_workflow_step_with_error_handler(self):
        """Test workflow step with error handler."""
        step = WorkflowStep(
            step="process",
            module="processor",
            error_handler="fallback_processor",
        )
        
        assert step.error_handler == "fallback_processor"


class TestAgentConfig:
    """Test complete agent configuration."""

    def test_minimal_agent_config(self):
        """Test minimal valid agent configuration."""
        config = AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(lm_model="gpt-4"),
            modules=[
                ModuleConfig(
                    name="responder",
                    type=ModuleType.SIGNATURE,
                    signature="query -> response",
                )
            ],
            workflow=[
                WorkflowStep(step="respond", module="responder")
            ],
        )
        
        assert config.name == "test-agent"
        assert config.version == "1.0.0"
        assert len(config.modules) == 1
        assert len(config.workflow) == 1

    def test_agent_config_with_deployment(self):
        """Test agent config with deployment settings."""
        config = AgentConfig(
            name="prod-agent",
            version="2.0.0",
            dspy=DSPyConfig(lm_model="databricks-dbrx"),
            modules=[
                ModuleConfig(
                    name="module1",
                    type=ModuleType.CHAIN_OF_THOUGHT,
                    signature="input -> output",
                )
            ],
            workflow=[
                WorkflowStep(step="step1", module="module1")
            ],
            deployment=DeploymentConfig(
                catalog="ml",
                schema_name="agents",
                model_name="prod_agent_v2",
                serving_endpoint="prod-endpoint",
                compute_size="Medium",
            ),
        )
        
        assert config.deployment.catalog == "ml"
        assert config.deployment.compute_size == "Medium"

    def test_duplicate_module_names_validation(self):
        """Test that duplicate module names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(
                name="test",
                version="1.0",
                dspy=DSPyConfig(lm_model="test"),
                modules=[
                    ModuleConfig(name="dup", type=ModuleType.SIGNATURE, signature="a->b"),
                    ModuleConfig(name="dup", type=ModuleType.SIGNATURE, signature="c->d"),
                ],
                workflow=[WorkflowStep(step="s1", module="dup")],
            )
        
        assert "Module names must be unique" in str(exc_info.value)

    def test_complete_agent_config(self):
        """Test complete agent configuration with all features."""
        config = AgentConfig(
            name="full-featured-agent",
            version="3.0.0",
            description="A fully featured agent for testing",
            dspy=DSPyConfig(
                lm_model="databricks-dbrx-instruct",
                temperature=0.5,
                max_tokens=2000,
                optimizer=OptimizerConfig(
                    type=OptimizerType.MIPRO_V2,
                    metric="f1_score",
                    num_candidates=20,
                    additional_params={"init_temperature": 0.8},
                ),
            ),
            modules=[
                ModuleConfig(
                    name="classifier",
                    type=ModuleType.SIGNATURE,
                    signature="text -> category, confidence",
                ),
                ModuleConfig(
                    name="retriever",
                    type=ModuleType.RETRIEVER,
                    vector_store=VectorStoreConfig(
                        catalog="ml",
                        schema_name="vectors",
                        index="docs",
                        k=5,
                    ),
                ),
                ModuleConfig(
                    name="generator",
                    type=ModuleType.CHAIN_OF_THOUGHT,
                    signature="query, context -> response",
                ),
            ],
            workflow=[
                WorkflowStep(
                    step="classify",
                    module="classifier",
                    inputs={"text": "$input.query"},
                ),
                WorkflowStep(
                    step="retrieve",
                    module="retriever",
                    condition="$classify.confidence > 0.7",
                    inputs={"query": "$input.query"},
                ),
                WorkflowStep(
                    step="generate",
                    module="generator",
                    inputs={
                        "query": "$input.query",
                        "context": "$retrieve.passages",
                    },
                    error_handler="fallback_response",
                ),
            ],
            deployment=DeploymentConfig(
                catalog="production",
                schema_name="agents",
                model_name="full_featured_agent",
                serving_endpoint="prod-full-featured",
                compute_size="Large",
                auto_capture_config={"enabled": True, "catalog": "logs"},
            ),
            metadata={
                "author": "test-team",
                "tags": ["production", "rag"],
            },
        )
        
        assert config.name == "full-featured-agent"
        assert config.description == "A fully featured agent for testing"
        assert len(config.modules) == 3
        assert len(config.workflow) == 3
    
    def test_backward_compatibility_schema_field(self):
        """Test that 'schema' field is converted to 'schema_name' for backward compatibility."""
        # Test VectorStoreConfig with 'schema' field
        vector_config = VectorStoreConfig(
            catalog="ml",
            schema="vectors",  # Using old field name
            index="knowledge_base",
            k=10,
        )
        assert vector_config.schema_name == "vectors"
        
        # Test DeploymentConfig with 'schema' field
        deploy_config = DeploymentConfig(
            catalog="ml",
            schema="agents",  # Using old field name
            model_name="test_model",
        )
        assert deploy_config.schema_name == "agents"