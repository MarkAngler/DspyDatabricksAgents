"""DSPy module factory and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import dspy
from databricks.sdk import WorkspaceClient

from config.schema import ModuleConfig, ModuleType, VectorStoreConfig
from core.tools import ToolRegistry


class ModuleRegistry:
    """Registry for DSPy modules."""
    
    _modules: Dict[str, Type['BaseModule']] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register custom modules."""
        def decorator(module_class: Type['BaseModule']) -> Type['BaseModule']:
            cls._modules[name] = module_class
            return module_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type['BaseModule']:
        """Get a registered module class."""
        if name not in cls._modules:
            raise ValueError(f"Module '{name}' not registered")
        return cls._modules[name]


class BaseModule(ABC):
    """Base class for all DSPy modules."""
    
    @abstractmethod
    def build(self, config: ModuleConfig) -> dspy.Module:
        """Build the DSPy module from configuration."""
        pass


class SignatureModule(BaseModule):
    """Basic signature module."""
    
    def build(self, config: ModuleConfig) -> dspy.Predict:
        """Build a dspy.Predict module."""
        if not config.signature:
            raise ValueError("Signature module requires 'signature' field")
        return dspy.Predict(config.signature)


class ChainOfThoughtModule(BaseModule):
    """Chain of thought reasoning module."""
    
    def build(self, config: ModuleConfig) -> dspy.ChainOfThought:
        """Build a dspy.ChainOfThought module."""
        if not config.signature:
            raise ValueError("ChainOfThought module requires 'signature' field")
        return dspy.ChainOfThought(config.signature)


class ReActModule(BaseModule):
    """ReAct (Reasoning and Acting) module."""
    
    def build(self, config: ModuleConfig) -> dspy.ReAct:
        """Build a dspy.ReAct module."""
        if not config.signature:
            raise ValueError("ReAct module requires 'signature' field")
        
        # Build tools if specified
        tools = []
        if config.tools:
            # Get tools from registry
            for tool_name in config.tools:
                tool = ToolRegistry.get(tool_name)
                tools.append(tool)
        
        # dspy.ReAct requires tools parameter
        return dspy.ReAct(config.signature, tools=tools)


class DatabricksRetriever(dspy.Retrieve):
    """Custom retriever for Databricks Vector Search."""
    
    def __init__(self, vector_store: VectorStoreConfig):
        """Initialize Databricks retriever."""
        self.vector_store = vector_store
        self.client = self._init_vector_client()
        super().__init__(k=vector_store.k)
    
    def _init_vector_client(self):
        """Initialize Databricks Vector Search client."""
        # For now, always return None since we're using mock data
        # In production, this would initialize the actual client
        return None
    
    def forward(self, query: str, k: Optional[int] = None, **kwargs) -> dspy.Prediction:
        """Retrieve relevant documents.
        
        Args:
            query: The main query string
            k: Number of results to retrieve
            **kwargs: Additional arguments (e.g., sub_queries) that may be passed but not used
        """
        k = k or self.k
        
        # If client is not initialized (deployment/testing), return mock results
        if not self.client:
            # Return mock documents as a Prediction with 'documents' field
            mock_docs = [f"Mock passage {i+1} for query: {query}" for i in range(k)]
            return dspy.Prediction(documents=mock_docs, passages=mock_docs)
        
        # Query vector search
        results = self.client.query(
            index_name=f"{self.vector_store.catalog}.{self.vector_store.schema_name}.{self.vector_store.index}",
            query_text=query,
            num_results=k,
        )
        
        # Format results - return both 'documents' and 'passages' for compatibility
        passages = [r.text for r in results.results]
        return dspy.Prediction(documents=passages, passages=passages)


class ModuleFactory:
    """Factory for creating DSPy modules."""
    
    def __init__(self):
        """Initialize and register built-in modules."""
        # Register built-in modules
        ModuleRegistry.register("signature")(SignatureModule)
        ModuleRegistry.register("chain_of_thought")(ChainOfThoughtModule)
        ModuleRegistry.register("react")(ReActModule)
    
    def create_module(self, config: ModuleConfig) -> dspy.Module:
        """Create a DSPy module from configuration."""
        # Handle retriever separately (not using registry)
        if config.type == ModuleType.RETRIEVER:
            if not config.vector_store:
                raise ValueError("Retriever module requires 'vector_store' configuration")
            return DatabricksRetriever(config.vector_store)
        
        # Handle custom modules
        if config.type == ModuleType.CUSTOM:
            if not config.custom_class:
                raise ValueError("Custom module requires 'custom_class' field")
            module_name = config.custom_class
        else:
            # Get module name from enum
            module_name = config.type.value if isinstance(config.type, ModuleType) else config.type
        
        # Get module class from registry
        module_class = ModuleRegistry.get(module_name)
        module_builder = module_class()
        return module_builder.build(config)