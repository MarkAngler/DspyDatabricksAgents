"""Unit tests for DSPy module factory."""

import pytest
from unittest.mock import Mock, patch
import dspy

from dspy_databricks_agents.core.modules import (
    ModuleFactory,
    ModuleRegistry,
    BaseModule,
    SignatureModule,
    ChainOfThoughtModule,
    ReActModule,
    DatabricksRetriever,
)
from dspy_databricks_agents.core.tools import ToolRegistry
from dspy_databricks_agents.config.schema import (
    ModuleConfig,
    ModuleType,
    VectorStoreConfig,
)


class TestModuleRegistry:
    """Test module registry functionality."""

    def test_register_module(self):
        """Test registering a custom module."""
        
        @ModuleRegistry.register("test_module")
        class TestModule(BaseModule):
            def build(self, config: ModuleConfig) -> dspy.Module:
                return Mock()
        
        # Should be registered
        assert "test_module" in ModuleRegistry._modules
        assert ModuleRegistry._modules["test_module"] == TestModule

    def test_get_registered_module(self):
        """Test retrieving a registered module."""
        
        @ModuleRegistry.register("another_test")
        class AnotherTestModule(BaseModule):
            def build(self, config: ModuleConfig) -> dspy.Module:
                return Mock()
        
        # Should retrieve the registered module
        module_class = ModuleRegistry.get("another_test")
        assert module_class == AnotherTestModule

    def test_get_unregistered_module_raises_error(self):
        """Test that getting an unregistered module raises error."""
        with pytest.raises(ValueError, match="Module 'nonexistent' not registered"):
            ModuleRegistry.get("nonexistent")


class TestSignatureModule:
    """Test signature module creation."""

    def test_build_signature_module(self):
        """Test building a signature module."""
        config = ModuleConfig(
            name="test_sig",
            type=ModuleType.SIGNATURE,
            signature="question -> answer",
        )
        
        module = SignatureModule().build(config)
        
        # Should return a dspy.Predict instance
        assert isinstance(module, dspy.Predict)
        # Check that signature has expected fields
        assert "question" in module.signature.input_fields
        assert "answer" in module.signature.output_fields

    def test_signature_module_without_signature_raises_error(self):
        """Test that signature module without signature field raises error."""
        config = ModuleConfig(
            name="invalid",
            type=ModuleType.SIGNATURE,
        )
        
        with pytest.raises(ValueError, match="Signature module requires 'signature' field"):
            SignatureModule().build(config)


class TestChainOfThoughtModule:
    """Test chain of thought module creation."""

    def test_build_chain_of_thought_module(self):
        """Test building a chain of thought module."""
        config = ModuleConfig(
            name="test_cot",
            type=ModuleType.CHAIN_OF_THOUGHT,
            signature="context, question -> reasoning, answer",
        )
        
        module = ChainOfThoughtModule().build(config)
        
        # Should return a dspy.ChainOfThought instance
        assert isinstance(module, dspy.ChainOfThought)
        # ChainOfThought wraps a Predict module
        assert hasattr(module, 'predict')

    def test_chain_of_thought_without_signature_raises_error(self):
        """Test that CoT module without signature field raises error."""
        config = ModuleConfig(
            name="invalid_cot",
            type=ModuleType.CHAIN_OF_THOUGHT,
        )
        
        with pytest.raises(ValueError, match="ChainOfThought module requires 'signature' field"):
            ChainOfThoughtModule().build(config)


class TestReActModule:
    """Test ReAct module creation."""

    def test_build_react_module(self):
        """Test building a ReAct module."""
        config = ModuleConfig(
            name="test_react",
            type=ModuleType.REACT,
            signature="task -> action, result",
            tools=["calculator", "web_search"]
        )
        
        module = ReActModule().build(config)
        
        # Should return a dspy.ReAct instance
        assert isinstance(module, dspy.ReAct)
        # Should have tools
        assert hasattr(module, 'tools')
        # ReAct automatically adds a 'finish' tool, so we have 3 total
        assert len(module.tools) == 3
        assert 'calculator' in module.tools
        assert 'web_search' in module.tools
        assert 'finish' in module.tools

    def test_react_without_signature_raises_error(self):
        """Test that ReAct module without signature field raises error."""
        config = ModuleConfig(
            name="invalid_react",
            type=ModuleType.REACT,
        )
        
        with pytest.raises(ValueError, match="ReAct module requires 'signature' field"):
            ReActModule().build(config)


class TestDatabricksRetriever:
    """Test Databricks retriever module."""

    @patch('dspy_databricks_agents.core.modules.WorkspaceClient')
    def test_databricks_retriever_initialization(self, mock_client):
        """Test Databricks retriever initialization."""
        vector_config = VectorStoreConfig(
            catalog="ml",
            schema="vectors",
            index="knowledge_base",
            k=10,
        )
        
        retriever = DatabricksRetriever(vector_config)
        
        assert retriever.vector_store == vector_config
        assert retriever.k == 10
        mock_client.assert_called_once()

    @patch('dspy_databricks_agents.core.modules.WorkspaceClient')
    def test_databricks_retriever_forward(self, mock_client):
        """Test Databricks retriever forward method."""
        # Setup mock
        mock_vector_client = Mock()
        mock_client.return_value.vector_search_indexes = mock_vector_client
        
        # Mock search results
        mock_result = Mock()
        mock_result.text = "Retrieved document"
        mock_response = Mock()
        mock_response.results = [mock_result]
        mock_vector_client.query.return_value = mock_response
        
        # Create retriever
        vector_config = VectorStoreConfig(
            catalog="ml",
            schema="vectors",
            index="docs",
            k=5,
        )
        retriever = DatabricksRetriever(vector_config)
        
        # Test forward
        result = retriever.forward("test query", k=3)
        
        # Verify call
        mock_vector_client.query.assert_called_once_with(
            index_name="ml.vectors.docs",
            query_text="test query",
            num_results=3,
        )
        
        # Check result
        assert isinstance(result, dspy.Prediction)
        assert result.passages == ["Retrieved document"]


class TestModuleFactory:
    """Test module factory functionality."""

    @pytest.fixture
    def factory(self):
        """Create a module factory instance."""
        return ModuleFactory()

    def test_factory_registers_builtin_modules(self, factory):
        """Test that factory registers built-in modules on init."""
        # Should have registered signature, chain_of_thought, and react
        assert "signature" in ModuleRegistry._modules
        assert "chain_of_thought" in ModuleRegistry._modules
        assert "react" in ModuleRegistry._modules

    def test_create_signature_module(self, factory):
        """Test creating a signature module through factory."""
        config = ModuleConfig(
            name="classifier",
            type=ModuleType.SIGNATURE,
            signature="text -> category",
        )
        
        module = factory.create_module(config)
        
        assert isinstance(module, dspy.Predict)

    def test_create_chain_of_thought_module(self, factory):
        """Test creating a CoT module through factory."""
        config = ModuleConfig(
            name="reasoner",
            type=ModuleType.CHAIN_OF_THOUGHT,
            signature="problem -> solution",
        )
        
        module = factory.create_module(config)
        
        assert isinstance(module, dspy.ChainOfThought)

    @patch('dspy_databricks_agents.core.modules.WorkspaceClient')
    def test_create_retriever_module(self, mock_client, factory):
        """Test creating a retriever module through factory."""
        config = ModuleConfig(
            name="retriever",
            type=ModuleType.RETRIEVER,
            vector_store=VectorStoreConfig(
                catalog="ml",
                schema="vectors",
                index="docs",
                k=5,
            ),
        )
        
        module = factory.create_module(config)
        
        assert isinstance(module, DatabricksRetriever)
        assert module.k == 5

    def test_create_custom_module(self, factory):
        """Test creating a custom module through factory."""
        # Register a custom module
        @ModuleRegistry.register("custom")
        class CustomTestModule(BaseModule):
            def build(self, config: ModuleConfig) -> dspy.Module:
                return dspy.Predict("input -> output")
        
        config = ModuleConfig(
            name="custom",
            type=ModuleType.CUSTOM,
            custom_class="custom",
        )
        
        module = factory.create_module(config)
        
        assert isinstance(module, dspy.Predict)

    def test_create_unregistered_module_raises_error(self, factory):
        """Test that creating an unregistered module type raises error."""
        config = ModuleConfig(
            name="unknown",
            type=ModuleType.CUSTOM,
            custom_class="unregistered_module",
        )
        
        with pytest.raises(ValueError, match="Module 'unregistered_module' not registered"):
            factory.create_module(config)

    def test_create_react_module(self, factory):
        """Test creating a ReAct module through factory."""
        config = ModuleConfig(
            name="react_agent",
            type=ModuleType.REACT,
            signature="task -> result",
            tools=["calculator", "web_search"]
        )
        
        module = factory.create_module(config)
        
        assert isinstance(module, dspy.ReAct)
    
    def test_react_module_without_signature_raises_error(self, factory):
        """Test that ReAct module without signature raises error."""
        config = ModuleConfig(
            name="invalid_react",
            type=ModuleType.REACT,
        )
        
        with pytest.raises(ValueError, match="ReAct module requires 'signature' field"):
            factory.create_module(config)