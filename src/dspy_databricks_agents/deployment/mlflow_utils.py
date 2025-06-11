"""MLflow utilities for experiment management and tracking."""

import os
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(
    experiment_name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[dict] = None
) -> str:
    """Get or create an MLflow experiment.
    
    This function ensures that we never use the default experiment (ID 0) by
    creating a named experiment if it doesn't exist, or retrieving the existing one.
    
    Args:
        experiment_name: Name of the experiment
        artifact_location: Optional artifact storage location
        tags: Optional tags to attach to the experiment
        
    Returns:
        Experiment ID
    """
    client = MlflowClient()
    
    # Try to get existing experiment
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # Create new experiment
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=tags or {}
        )
    else:
        experiment_id = experiment.experiment_id
        
        # Update tags if provided and experiment exists
        if tags:
            for key, value in tags.items():
                client.set_experiment_tag(experiment_id, key, value)
    
    return experiment_id


def set_experiment_with_environment(
    base_name: str,
    environment: str = "dev",
    project_prefix: Optional[str] = None
) -> str:
    """Set MLflow experiment with environment-aware naming.
    
    Creates a structured experiment name that includes environment and optional
    project prefix, following Databricks naming conventions.
    
    Args:
        base_name: Base name for the experiment (e.g., agent name)
        environment: Deployment environment (dev, staging, prod)
        project_prefix: Optional project prefix for organization
        
    Returns:
        Experiment ID
    """
    # Build experiment name components
    name_parts = []
    
    if project_prefix:
        name_parts.append(project_prefix.lower())
    
    name_parts.extend([
        "dspy_agents",
        environment.lower(),
        base_name.lower().replace(" ", "_")
    ])
    
    experiment_name = "/".join(name_parts)
    
    # Add standard tags
    tags = {
        "environment": environment,
        "framework": "dspy",
        "agent_type": "databricks_agent",
        "base_name": base_name
    }
    
    # Get or create the experiment
    experiment_id = get_or_create_experiment(experiment_name, tags=tags)
    
    # Set as active experiment
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def ensure_experiment_set(agent_name: Optional[str] = None) -> str:
    """Ensure an experiment is set, creating a default if needed.
    
    This function checks if an experiment is already set. If not, it creates
    a default experiment based on the agent name, environment variables, or a standard name.
    
    Args:
        agent_name: Optional agent name to use for experiment naming
        
    Returns:
        Current experiment ID
    """
    # Check if experiment is already set
    experiment = mlflow.get_experiment(mlflow.active_run().info.experiment_id) if mlflow.active_run() else None
    
    if experiment and experiment.experiment_id != "0":
        return experiment.experiment_id
    
    # Check for environment variable
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    
    if not experiment_name:
        # Use agent name if provided, otherwise use default
        if agent_name:
            experiment_name = f"dspy-agents/{agent_name.lower().replace(' ', '_')}"
        else:
            experiment_name = "dspy-agents-default"
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Get the experiment ID
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    return experiment.experiment_id