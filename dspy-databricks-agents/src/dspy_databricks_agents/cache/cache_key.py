"""Cache key generation utilities."""

import hashlib
import json
from typing import Any, Dict, List, Union


def normalize_value(value: Any) -> Any:
    """Normalize values for consistent hashing."""
    if isinstance(value, dict):
        # Sort dict keys for consistency
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    elif isinstance(value, set):
        return sorted(list(value))
    else:
        return value


def generate_cache_key(
    module_name: str,
    inputs: Dict[str, Any],
    model_name: str = None,
    temperature: float = None,
    additional_params: Dict[str, Any] = None
) -> str:
    """
    Generate a cache key for LLM calls.
    
    Args:
        module_name: Name of the DSPy module
        inputs: Input parameters to the module
        model_name: Optional model name
        temperature: Optional temperature setting
        additional_params: Any additional parameters that affect output
    
    Returns:
        SHA256 hash as cache key
    """
    # Build cache key components
    key_data = {
        "module": module_name,
        "inputs": normalize_value(inputs),
    }
    
    if model_name:
        key_data["model"] = model_name
    
    if temperature is not None:
        key_data["temperature"] = temperature
    
    if additional_params:
        key_data["params"] = normalize_value(additional_params)
    
    # Create deterministic JSON string
    json_str = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
    
    # Generate SHA256 hash
    return hashlib.sha256(json_str.encode()).hexdigest()