"""Configuration schema for DSPy-Databricks Agents."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OptimizerType(str, Enum):
    """Supported DSPy optimizer types."""
    
    BOOTSTRAP_FEW_SHOT = "BootstrapFewShot"
    BOOTSTRAP_FEW_SHOT_WITH_RANDOM_SEARCH = "BootstrapFewShotWithRandomSearch"
    MIPRO = "MIPRO"
    MIPRO_V2 = "MIPROv2"
    COPRO = "COPRO"


class ModuleType(str, Enum):
    """Supported DSPy module types."""
    
    SIGNATURE = "signature"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    RETRIEVER = "retriever"
    CUSTOM = "custom"


class OptimizerConfig(BaseModel):
    """Configuration for DSPy optimizers."""
    
    type: OptimizerType
    metric: str
    num_candidates: int = Field(default=10, ge=1)
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class DSPyConfig(BaseModel):
    """DSPy-specific configuration."""
    
    lm_model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=8192)
    optimizer: Optional[OptimizerConfig] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector store retrieval."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    catalog: str
    schema_name: str
    index: str
    k: int = Field(default=5, ge=1)
    
    @model_validator(mode='before')
    @classmethod
    def handle_schema_field(cls, values):
        """Handle both 'schema' and 'schema_name' fields for backward compatibility."""
        if isinstance(values, dict):
            if 'schema' in values and 'schema_name' not in values:
                values['schema_name'] = values.pop('schema')
        return values


class ModuleConfig(BaseModel):
    """Configuration for a DSPy module."""
    
    name: str
    type: ModuleType
    signature: Optional[str] = None
    vector_store: Optional[VectorStoreConfig] = None
    tools: Optional[List[str]] = None
    custom_class: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStep(BaseModel):
    """Configuration for a workflow step."""
    
    step: str
    module: str
    condition: Optional[str] = None
    inputs: Dict[str, Union[str, Any]] = Field(default_factory=dict)
    error_handler: Optional[str] = None


class DeploymentConfig(BaseModel):
    """Configuration for Databricks deployment."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    catalog: str
    schema_name: str
    model_name: str
    serving_endpoint: Optional[str] = None
    compute_size: str = "Small"
    auto_capture_config: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, Any]] = None
    environment_vars: Optional[Dict[str, str]] = None
    
    @model_validator(mode='before')
    @classmethod
    def handle_schema_field(cls, values):
        """Handle both 'schema' and 'schema_name' fields for backward compatibility."""
        if isinstance(values, dict):
            if 'schema' in values and 'schema_name' not in values:
                values['schema_name'] = values.pop('schema')
        return values


class AgentConfig(BaseModel):
    """Complete agent configuration."""
    
    name: str
    version: str
    description: Optional[str] = None
    dspy: DSPyConfig
    modules: List[ModuleConfig]
    workflow: List[WorkflowStep]
    deployment: Optional[DeploymentConfig] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("modules")
    @classmethod
    def validate_unique_module_names(cls, modules: List[ModuleConfig]) -> List[ModuleConfig]:
        """Ensure all module names are unique."""
        names = [m.name for m in modules]
        if len(names) != len(set(names)):
            raise ValueError("Module names must be unique")
        return modules