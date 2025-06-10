"""YAML parser for agent configurations."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Set

import yaml
from pydantic import ValidationError

from dspy_databricks_agents.config.schema import AgentConfig


class YAMLParser:
    """Enhanced YAML parser with environment variables and imports."""
    
    def __init__(self):
        self.env_pattern = re.compile(r'\$\{([^}]+)\}')
        self.import_cache: Dict[str, Any] = {}
        self._import_stack: Set[str] = set()  # For circular import detection
    
    def parse_file(self, file_path: str) -> AgentConfig:
        """Parse YAML file with all enhancements."""
        path = Path(file_path).resolve()
        
        # Load raw YAML
        with open(path, 'r') as f:
            raw_content = f.read()
        
        # Substitute environment variables
        content = self._substitute_env_vars(raw_content)
        
        # Parse YAML
        data = yaml.safe_load(content)
        
        # Handle imports with circular detection
        if 'imports' in data:
            data = self._process_imports(data, path.parent, str(path))
        
        # Validate and convert to Pydantic
        try:
            return AgentConfig.model_validate(data['agent'])
        except KeyError:
            raise ValueError("YAML must contain 'agent' top-level key")
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def parse_string(self, yaml_content: str) -> AgentConfig:
        """Parse YAML from string."""
        # Substitute environment variables
        content = self._substitute_env_vars(yaml_content)
        
        # Parse YAML
        data = yaml.safe_load(content)
        
        # Validate and convert to Pydantic
        try:
            return AgentConfig.model_validate(data['agent'])
        except KeyError:
            raise ValueError("YAML must contain 'agent' top-level key")
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR} with environment variable values."""
        def replacer(match):
            var_name = match.group(1)
            # Support default values: ${VAR:-default}
            if ':-' in var_name:
                var_name, default = var_name.split(':-', 1)
                value = os.environ.get(var_name, default)
            else:
                value = os.environ.get(var_name, match.group(0))
            
            # Try to preserve type for common cases
            if value.lower() in ('true', 'false'):
                return value.lower()
            elif value.replace('.', '').isdigit():
                return value
            else:
                # Keep as string
                return value
        
        return self.env_pattern.sub(replacer, content)
    
    def _process_imports(self, data: Dict, base_path: Path, current_file: str) -> Dict:
        """Process import statements with circular import detection."""
        # Check for circular import
        if current_file in self._import_stack:
            raise ValueError(f"Circular import detected: {current_file}")
        
        self._import_stack.add(current_file)
        
        try:
            imports = data.pop('imports', [])
            
            for import_path in imports:
                # Resolve relative to base path
                full_path = (base_path / import_path).resolve()
                cache_key = str(full_path)
                
                if cache_key in self.import_cache:
                    imported_data = self.import_cache[cache_key]
                else:
                    # Load and parse imported file
                    with open(full_path, 'r') as f:
                        raw_content = f.read()
                    
                    content = self._substitute_env_vars(raw_content)
                    imported_data = yaml.safe_load(content)
                    
                    # Process nested imports
                    if 'imports' in imported_data:
                        imported_data = self._process_imports(
                            imported_data,
                            full_path.parent,
                            str(full_path)
                        )
                    
                    self.import_cache[cache_key] = imported_data
                
                # Merge imported data
                data = self._deep_merge(imported_data, data)
            
            return data
        finally:
            # Remove from stack when done
            self._import_stack.remove(current_file)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result:
                if key == 'agent' and isinstance(result[key], dict) and isinstance(value, dict):
                    # Special handling for agent section - merge nested fields
                    if 'modules' in result and 'modules' in value['agent']:
                        # Move base modules into agent section
                        if 'modules' not in result['agent']:
                            result['agent']['modules'] = []
                        result['agent']['modules'].extend(result.pop('modules', []))
                    
                    if 'dspy' in result and 'dspy' in value['agent']:
                        # Merge dspy configs
                        if 'dspy' not in result['agent']:
                            result['agent']['dspy'] = {}
                        result['agent']['dspy'] = self._deep_merge(
                            result.pop('dspy', {}),
                            result['agent']['dspy']
                        )
                    
                    # Merge agent sections
                    result[key] = self._deep_merge(result[key], value)
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge dictionaries
                    result[key] = self._deep_merge(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # Extend lists (modules, workflow steps, etc.)
                    result[key] = result[key] + value
                else:
                    # Override value
                    result[key] = value
            else:
                result[key] = value
        
        # Move top-level configs into agent section if needed
        if 'agent' in result:
            for field in ['modules', 'workflow', 'dspy']:
                if field in result and field not in result['agent']:
                    result['agent'][field] = result.pop(field)
                elif field in result and field in result['agent']:
                    if isinstance(result[field], list):
                        result['agent'][field] = result.pop(field) + result['agent'][field]
                    elif isinstance(result[field], dict):
                        result['agent'][field] = self._deep_merge(result.pop(field), result['agent'][field])
        
        return result