"""Unit tests for YAML parser."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from dspy_databricks_agents.config.parser import YAMLParser
from dspy_databricks_agents.config.schema import AgentConfig, ModuleType


class TestYAMLParser:
    """Test YAML parser functionality."""

    @pytest.fixture
    def parser(self):
        """Create a YAMLParser instance."""
        return YAMLParser()

    @pytest.fixture
    def sample_yaml(self):
        """Sample YAML configuration."""
        return """
agent:
  name: test-agent
  version: 1.0.0
  description: Test agent for unit tests
  
  dspy:
    lm_model: gpt-4
    temperature: 0.7
    max_tokens: 1000
    
  modules:
    - name: classifier
      type: signature
      signature: "text -> category"
      
  workflow:
    - step: classify
      module: classifier
      inputs:
        text: "$input.query"
"""

    def test_parse_valid_yaml(self, parser, sample_yaml, tmp_path):
        """Test parsing valid YAML file."""
        # Write YAML to temp file
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(sample_yaml)
        
        # Parse file
        config = parser.parse_file(str(yaml_file))
        
        # Verify parsed config
        assert isinstance(config, AgentConfig)
        assert config.name == "test-agent"
        assert config.version == "1.0.0"
        assert config.description == "Test agent for unit tests"
        assert config.dspy.lm_model == "gpt-4"
        assert len(config.modules) == 1
        assert config.modules[0].name == "classifier"

    def test_environment_variable_substitution(self, parser, tmp_path):
        """Test environment variable substitution in YAML."""
        # Set environment variables
        os.environ["TEST_MODEL"] = "databricks-dbrx"
        os.environ["TEST_TEMP"] = "0.5"
        
        yaml_content = """
agent:
  name: env-test
  version: 1.0.0
  
  dspy:
    lm_model: ${TEST_MODEL}
    temperature: ${TEST_TEMP}
    
  modules:
    - name: test
      type: signature
      signature: "a -> b"
      
  workflow:
    - step: test
      module: test
"""
        
        yaml_file = tmp_path / "env_agent.yaml"
        yaml_file.write_text(yaml_content)
        
        config = parser.parse_file(str(yaml_file))
        
        assert config.dspy.lm_model == "databricks-dbrx"
        assert config.dspy.temperature == 0.5
        
        # Cleanup
        del os.environ["TEST_MODEL"]
        del os.environ["TEST_TEMP"]

    def test_environment_variable_with_default(self, parser, tmp_path):
        """Test environment variable substitution with default values."""
        yaml_content = """
agent:
  name: default-test
  version: ${VERSION:-2.0.0}
  
  dspy:
    lm_model: ${MODEL:-gpt-3.5-turbo}
    temperature: ${TEMP:-0.7}
    
  modules:
    - name: test
      type: signature
      signature: "a -> b"
      
  workflow:
    - step: test
      module: test
"""
        
        yaml_file = tmp_path / "default_agent.yaml"
        yaml_file.write_text(yaml_content)
        
        config = parser.parse_file(str(yaml_file))
        
        # Should use default values
        assert config.version == "2.0.0"
        assert config.dspy.lm_model == "gpt-3.5-turbo"
        assert config.dspy.temperature == 0.7

    def test_yaml_imports(self, parser, tmp_path):
        """Test YAML import functionality."""
        # Create base config
        base_config = """
dspy:
  lm_model: gpt-4
  temperature: 0.7
  
modules:
  - name: base_module
    type: signature
    signature: "input -> output"
"""
        base_file = tmp_path / "base.yaml"
        base_file.write_text(base_config)
        
        # Create main config with import
        main_config = f"""
imports:
  - base.yaml

agent:
  name: import-test
  version: 1.0.0
  
  dspy:
    temperature: 0.9  # Override base temperature
    
  modules:
    - name: main_module
      type: chain_of_thought
      signature: "query -> answer"
      
  workflow:
    - step: process
      module: main_module
"""
        main_file = tmp_path / "main.yaml"
        main_file.write_text(main_config)
        
        config = parser.parse_file(str(main_file))
        
        # Should have both modules
        assert len(config.modules) == 2
        assert any(m.name == "base_module" for m in config.modules)
        assert any(m.name == "main_module" for m in config.modules)
        
        # Temperature should be overridden
        assert config.dspy.temperature == 0.9
        assert config.dspy.lm_model == "gpt-4"  # From base

    def test_nested_imports(self, parser, tmp_path):
        """Test nested YAML imports."""
        # Create configs directory
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        
        # Base modules config
        modules_config = """
modules:
  - name: retriever
    type: retriever
    vector_store:
      catalog: ml
      schema: vectors
      index: docs
      k: 10
"""
        (configs_dir / "modules.yaml").write_text(modules_config)
        
        # Workflow config that imports modules
        workflow_config = """
imports:
  - modules.yaml

workflow:
  - step: retrieve
    module: retriever
    inputs:
      query: "$input.query"
"""
        (configs_dir / "workflow.yaml").write_text(workflow_config)
        
        # Main config that imports workflow
        main_config = """
imports:
  - configs/workflow.yaml

agent:
  name: nested-import-test
  version: 1.0.0
  
  dspy:
    lm_model: gpt-4
"""
        main_file = tmp_path / "main.yaml"
        main_file.write_text(main_config)
        
        config = parser.parse_file(str(main_file))
        
        # Should have module from nested import
        assert len(config.modules) == 1
        assert config.modules[0].name == "retriever"
        assert config.modules[0].type == ModuleType.RETRIEVER
        
        # Should have workflow from import
        assert len(config.workflow) == 1
        assert config.workflow[0].step == "retrieve"

    def test_circular_import_detection(self, parser, tmp_path):
        """Test that circular imports are detected."""
        # Create two files that import each other
        config1 = """
imports:
  - config2.yaml

agent:
  name: config1
  version: 1.0.0
"""
        
        config2 = """
imports:
  - config1.yaml
  
dspy:
  lm_model: gpt-4
"""
        
        (tmp_path / "config1.yaml").write_text(config1)
        (tmp_path / "config2.yaml").write_text(config2)
        
        # Should detect circular import
        with pytest.raises(ValueError, match="Circular import detected"):
            parser.parse_file(str(tmp_path / "config1.yaml"))

    def test_missing_required_fields(self, parser, tmp_path):
        """Test validation of missing required fields."""
        yaml_content = """
agent:
  name: incomplete-agent
  # Missing version and other required fields
"""
        
        yaml_file = tmp_path / "incomplete.yaml"
        yaml_file.write_text(yaml_content)
        
        with pytest.raises(ValueError):
            parser.parse_file(str(yaml_file))

    def test_invalid_yaml_syntax(self, parser, tmp_path):
        """Test handling of invalid YAML syntax."""
        yaml_content = """
agent:
  name: invalid
  version: 1.0.0
  modules:  # Invalid indentation below
   - name: test
  type: signature
"""
        
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(yaml_content)
        
        with pytest.raises((yaml.YAMLError, ValueError)):
            parser.parse_file(str(yaml_file))

    def test_parse_string(self, parser, sample_yaml):
        """Test parsing YAML from string."""
        config = parser.parse_string(sample_yaml)
        
        assert isinstance(config, AgentConfig)
        assert config.name == "test-agent"
        assert config.version == "1.0.0"

    def test_cache_imports(self, parser, tmp_path):
        """Test that imports are cached for performance."""
        # Create a base config
        base_config = """
modules:
  - name: cached_module
    type: signature
    signature: "a -> b"
"""
        base_file = tmp_path / "cached.yaml"
        base_file.write_text(base_config)
        
        # Create two configs that import the same base
        for i in range(2):
            config = f"""
imports:
  - cached.yaml

agent:
  name: test-{i}
  version: 1.0.0
  
  dspy:
    lm_model: gpt-4
    
  workflow:
    - step: test
      module: cached_module
"""
            (tmp_path / f"config{i}.yaml").write_text(config)
        
        # Parse both configs
        config1 = parser.parse_file(str(tmp_path / "config0.yaml"))
        config2 = parser.parse_file(str(tmp_path / "config1.yaml"))
        
        # Both should have the cached module
        assert config1.modules[0].name == "cached_module"
        assert config2.modules[0].name == "cached_module"
        
        # Cache should have the import (using full path as key)
        cached_path = str((tmp_path / "cached.yaml").resolve())
        assert cached_path in parser.import_cache