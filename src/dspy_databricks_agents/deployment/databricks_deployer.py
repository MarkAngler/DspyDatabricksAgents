"""Databricks deployment implementation for DSPy agents."""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import time

from dspy_databricks_agents.config.schema import AgentConfig
from dspy_databricks_agents.core.agent import Agent


class DatabricksDeployer:
    """Handles deployment of DSPy agents to Databricks."""

    def __init__(self, host: Optional[str] = None, token: Optional[str] = None):
        """Initialize Databricks deployer.

        Args:
            host: Databricks workspace host (or from DATABRICKS_HOST env)
            token: Databricks access token (or from DATABRICKS_TOKEN env)
        """
        self.host = host or os.environ.get("DATABRICKS_HOST")
        self.token = token or os.environ.get("DATABRICKS_TOKEN")

        if not self.host or not self.token:
            raise ValueError(
                "Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN "
                "environment variables or pass them to the constructor."
            )

        # Import Databricks SDK when needed
        try:
            from databricks.sdk import WorkspaceClient

            self.client = WorkspaceClient(host=self.host, token=self.token)
        except ImportError:
            raise ImportError(
                "Databricks SDK not installed. Install with: pip install databricks-sdk"
            )

    def deploy(
        self, config: AgentConfig, environment: str = "dev", dry_run: bool = False,
        enable_auto_rollback: bool = True
    ) -> Dict[str, Any]:
        """Deploy agent to Databricks Model Serving.

        Args:
            config: Agent configuration
            environment: Deployment environment
            dry_run: If True, only validate without deploying
            enable_auto_rollback: Enable automatic rollback on failure

        Returns:
            Deployment result with endpoint information
        """
        # Validate deployment configuration
        if not config.deployment:
            raise ValueError("Agent configuration missing deployment section")

        deployment = config.deployment

        # Build full names with environment
        catalog = deployment.catalog or f"{environment}_ml"
        schema = deployment.schema_name or "agents"
        model_name = deployment.model_name or f"{config.name}_{environment}"
        endpoint_name = deployment.serving_endpoint or f"{environment}-{config.name}"

        result = {
            "agent_name": config.name,
            "agent_version": config.version,
            "environment": environment,
            "catalog": catalog,
            "schema": schema,
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "compute_size": deployment.compute_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if dry_run:
            result["status"] = "dry_run"
            result["message"] = "Deployment validated successfully (dry run)"
            return result

        # Store previous version for potential rollback
        previous_version = None
        if enable_auto_rollback:
            try:
                endpoint = self.client.serving_endpoints.get(name=endpoint_name)
                previous_config = endpoint.config.served_entities[0]
                previous_version = previous_config.entity_version
                result["previous_version"] = previous_version
            except Exception:
                # No existing endpoint - this is a new deployment
                pass

        try:
            # Step 1: Register the model in Unity Catalog
            model_uri, model_version = self._register_model(config, catalog, schema, model_name)
            result["model_uri"] = model_uri
            result["model_version"] = model_version

            # Step 2: Create or update serving endpoint
            endpoint_info = self._create_serving_endpoint(
                model_uri=model_uri,
                model_version=model_version,
                endpoint_name=endpoint_name,
                compute_size=deployment.compute_size,
                environment_vars=deployment.environment_vars,
            )
            result["endpoint_url"] = endpoint_info["url"]
            result["endpoint_state"] = endpoint_info["state"]

            # Step 3: Configure rate limiting if specified
            if deployment.rate_limits:
                self._configure_rate_limits(
                    endpoint_name=endpoint_name, rate_limits=deployment.rate_limits
                )
                result["rate_limits"] = deployment.rate_limits

            # Step 4: Validate deployment health
            if enable_auto_rollback:
                health_check = self._validate_endpoint_health(endpoint_name, timeout=30)
                result["health_check"] = health_check
                
                if not health_check.get("healthy", False):
                    # Deployment failed health check - trigger rollback
                    if previous_version:
                        rollback_result = self.rollback_deployment(
                            endpoint_name=endpoint_name,
                            target_version=previous_version,
                            validate_health=False
                        )
                        result["auto_rollback"] = rollback_result
                        result["status"] = "failed_with_rollback"
                        result["message"] = (
                            f"Deployment failed health check and was rolled back to version {previous_version}"
                        )
                        raise Exception(f"failed_with_rollback: {result['message']}")
                    else:
                        result["status"] = "failed"
                        result["message"] = "Deployment failed health check (no previous version for rollback)"
                        raise Exception(result["message"])

            result["status"] = "success"
            result["message"] = f"Agent deployed successfully to {endpoint_info['url']}"

        except Exception as e:
            if "failed_with_rollback" not in str(e):
                result["status"] = "failed"
                result["error"] = str(e)
                
                # Attempt automatic rollback on failure
                if enable_auto_rollback and previous_version:
                    try:
                        rollback_result = self.rollback_deployment(
                            endpoint_name=endpoint_name,
                            target_version=previous_version,
                            validate_health=False
                        )
                        result["auto_rollback"] = rollback_result
                        result["status"] = "failed_with_rollback"
                        result["message"] = (
                            f"Deployment failed: {str(e)}. Rolled back to version {previous_version}"
                        )
                    except Exception as rollback_error:
                        result["rollback_error"] = str(rollback_error)
                        result["message"] = (
                            f"Deployment failed and rollback also failed: {str(rollback_error)}"
                        )
            raise

        return result

    def _register_model(
        self, config: AgentConfig, catalog: str, schema: str, model_name: str
    ) -> tuple[str, str]:
        """Register agent as MLflow model in Unity Catalog.

        Returns:
            Tuple of (Model URI for the registered model, version number)
        """
        import mlflow
        import shutil
        from dspy_databricks_agents.deployment.model_signature import get_signature_for_config
        from dspy_databricks_agents.deployment.mlflow_utils import set_experiment_with_environment

        # Configure MLflow to use Unity Catalog
        mlflow.set_registry_uri("databricks-uc")
        
        # Set up MLflow experiment to avoid default experiment warning
        environment = catalog.split("_")[0] if "_" in catalog else "prod"
        set_experiment_with_environment(
            base_name=config.name,
            environment=environment,
            project_prefix="databricks"
        )

        # Create temporary directory for model artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config directory for artifacts
            config_dir = temp_path / "config"
            config_dir.mkdir(exist_ok=True)

            # Save agent configuration
            config_path = config_dir / "agent_config.json"
            with open(config_path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)

            # Get the mlflow_model.py module source
            mlflow_model_src = Path(__file__).parent / "mlflow_model.py"

            # Check if the source file exists
            if not mlflow_model_src.exists():
                raise FileNotFoundError(
                    f"MLflow model file not found at {mlflow_model_src}. "
                    f"Expected file in deployment package directory."
                )

            # Create conda environment specification
            conda_env = {
                "name": "dspy-agent-env",
                "channels": ["defaults", "conda-forge"],
                "dependencies": [
                    "python=3.9",
                    "pip",
                    {
                        "pip": [
                            "dspy-ai",
                            "mlflow>=2.0",
                            "pydantic>=2.0",
                            "dspy-databricks-agents",
                            "databricks-sdk",
                        ]
                    },
                ],
            }

            # Create the model loader script path
            model_loader_path = Path(__file__).parent / "model_loader.py"
            
            # Get the appropriate signature for this agent configuration
            signature = get_signature_for_config(config)
            
            # Log the model using code-based approach
            with mlflow.start_run() as run:
                # Log agent metadata
                mlflow.log_param("agent_name", config.name)
                mlflow.log_param("agent_version", config.version)
                mlflow.log_param("dspy_model", config.dspy.lm_model)
                mlflow.log_param("num_modules", len(config.modules))
                mlflow.log_param("num_workflow_steps", len(config.workflow))

                # Copy config to a data directory for code-based loading
                data_dir = temp_path / "data"
                data_dir.mkdir(exist_ok=True)
                
                # Copy config file to data directory
                import shutil
                shutil.copy2(config_path, data_dir / "agent_config.json")
                
                # Log the model using code-based logging with signature
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="agent",
                    loader_module="dspy_databricks_agents.deployment.model_loader",
                    data_path=str(data_dir),  # Use data_path instead of artifacts
                    code_path=[str(mlflow_model_src.parent.parent.parent)],  # Include the entire package
                    conda_env=conda_env,
                    signature=signature,  # Add the model signature
                    registered_model_name=f"{catalog}.{schema}.{model_name}",
                )

                model_uri = f"models:/{catalog}.{schema}.{model_name}/latest"
                
                # Get the version number from the registered model
                # model_info.registered_model_version is the version string
                version_number = model_info.registered_model_version if model_info.registered_model_version else "1"

        return model_uri, version_number

    def _create_serving_endpoint(
        self,
        model_uri: str,
        model_version: str,
        endpoint_name: str,
        compute_size: str = "Small",
        environment_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create or update a model serving endpoint.

        Returns:
            Endpoint information including URL and state
        """
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput,
            ServedEntityInput,
            TrafficConfig,
            Route,
        )

        # Map compute sizes to workload sizes
        workload_size_map = {"Small": "Small", "Medium": "Medium", "Large": "Large"}

        workload_size = workload_size_map.get(compute_size, "Small")

        # Extract the model name from the URI (remove models:/ prefix and /latest suffix)
        if model_uri.startswith("models:/"):
            model_full_name = model_uri.replace("models:/", "").replace("/latest", "")
        else:
            model_full_name = model_uri
            
        # Configure the served model
        served_models = [
            ServedEntityInput(
                entity_name=model_full_name,
                entity_version=model_version,
                workload_size=workload_size,
                scale_to_zero_enabled=True,
                environment_vars=environment_vars or {},
            )
        ]

        # Extract just the model name (without catalog.schema prefix)
        model_name_only = model_full_name.split(".")[-1]
        
        # Configure traffic routing (100% to latest version)
        # Databricks expects the served model name in format: {model_name}-{version}
        served_model_name = f"{model_name_only}-{model_version}"
        
        traffic_config = TrafficConfig(
            routes=[
                Route(
                    served_model_name=served_model_name,
                    traffic_percentage=100,
                )
            ]
        )

        endpoint_config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=served_models,
            traffic_config=traffic_config,
        )

        try:
            # Try to get existing endpoint
            endpoint = self.client.serving_endpoints.get(name=endpoint_name)

            # Update existing endpoint
            self.client.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=served_models,
                traffic_config=traffic_config,
            )

            action = "updated"

        except Exception:
            # Create new endpoint
            endpoint = self.client.serving_endpoints.create(
                name=endpoint_name, config=endpoint_config
            )

            action = "created"

        # Wait for endpoint to be ready (with timeout)
        import time

        max_wait = 600  # 10 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            endpoint = self.client.serving_endpoints.get(name=endpoint_name)
            if endpoint.state.ready == "READY":
                break
            time.sleep(10)

        # Build endpoint URL
        workspace_url = self.host.rstrip("/")
        endpoint_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"

        return {"url": endpoint_url, "state": endpoint.state.ready, "action": action}

    def _configure_rate_limits(
        self, endpoint_name: str, rate_limits: Dict[str, Any]
    ) -> None:
        """Configure rate limiting for the endpoint.

        Note: This is a placeholder for future implementation.
        Actual rate limiting may need to be configured through
        Databricks policies or custom middleware.
        """
        # TODO: Implement rate limiting configuration
        # This might involve:
        # 1. Setting up API Gateway rules
        # 2. Configuring endpoint policies
        # 3. Adding custom middleware
        pass

    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Get status of a deployed endpoint.

        Args:
            endpoint_name: Name of the serving endpoint

        Returns:
            Endpoint status information
        """
        try:
            endpoint = self.client.serving_endpoints.get(name=endpoint_name)

            return {
                "name": endpoint.name,
                "state": endpoint.state.ready,
                "creation_timestamp": endpoint.creation_timestamp,
                "last_updated_timestamp": endpoint.last_updated_timestamp,
                "creator": endpoint.creator,
                "config": {
                    "served_entities": [
                        {
                            "entity_name": entity.entity_name,
                            "entity_version": entity.entity_version,
                            "workload_size": entity.workload_size,
                            "scale_to_zero_enabled": entity.scale_to_zero_enabled,
                        }
                        for entity in endpoint.config.served_entities
                    ]
                },
            }
        except Exception as e:
            return {"error": f"Failed to get endpoint status: {str(e)}"}

    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a serving endpoint.

        Args:
            endpoint_name: Name of the endpoint to delete

        Returns:
            True if deletion successful
        """
        try:
            self.client.serving_endpoints.delete(name=endpoint_name)
            return True
        except Exception:
            return False

    def rollback_deployment(
        self,
        endpoint_name: str,
        target_version: Optional[str] = None,
        validate_health: bool = True,
    ) -> Dict[str, Any]:
        """Rollback deployment to a previous version within 30 seconds.

        Args:
            endpoint_name: Name of the serving endpoint
            target_version: Specific version to rollback to (or previous if None)
            validate_health: Whether to validate health after rollback

        Returns:
            Rollback result with status and details
        """
        import time

        start_time = time.time()
        result = {
            "endpoint_name": endpoint_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_version": target_version,
        }

        try:
            # Step 1: Get current endpoint configuration
            endpoint = self.client.serving_endpoints.get(name=endpoint_name)
            current_config = endpoint.config.served_entities[0]
            current_model_name = current_config.entity_name
            current_version = current_config.entity_version

            result["current_version"] = current_version

            # Extract model details from URI
            if "models:/" in current_model_name:
                model_parts = current_model_name.replace("models:/", "").split("/")
                full_model_name = model_parts[0]
            else:
                # Handle case where entity_name might be different format
                raise ValueError(f"Cannot parse model name from: {current_model_name}")

            # Step 2: Determine target version
            if target_version is None:
                # Get previous version
                versions = self._get_model_versions_for_rollback(full_model_name)
                if len(versions) < 2:
                    raise ValueError("No previous version available for rollback")

                # Find the previous version (assuming versions are sorted desc)
                for v in versions:
                    if v["version"] != current_version:
                        target_version = v["version"]
                        break

                if target_version is None:
                    raise ValueError("Could not determine target version for rollback")

            result["rollback_version"] = target_version

            # Step 3: Create rollback configuration
            from databricks.sdk.service.serving import ServedEntityInput

            rollback_model_uri = f"models:/{full_model_name}/{target_version}"

            served_entities = [
                ServedEntityInput(
                    entity_name=rollback_model_uri,
                    entity_version=target_version,
                    workload_size=current_config.workload_size,
                    scale_to_zero_enabled=current_config.scale_to_zero_enabled,
                    environment_vars=current_config.environment_vars or {},
                )
            ]

            # Step 4: Execute rollback with immediate traffic switch
            self.client.serving_endpoints.update_config(
                name=endpoint_name, served_entities=served_entities
            )

            # Step 5: Wait for rollback to complete (with 30-second timeout)
            max_wait = 30  # 30-second requirement
            poll_interval = 2

            while time.time() - start_time < max_wait:
                endpoint = self.client.serving_endpoints.get(name=endpoint_name)
                if endpoint.state.ready == "READY":
                    break
                time.sleep(poll_interval)

            rollback_time = time.time() - start_time
            result["rollback_time_seconds"] = rollback_time

            # Step 6: Validate health if requested
            if validate_health:
                health_result = self._validate_endpoint_health(endpoint_name)
                result["health_check"] = health_result

                if not health_result.get("healthy", False):
                    result["status"] = "warning"
                    result["message"] = (
                        f"Rollback completed in {rollback_time:.1f}s but health check failed"
                    )
                else:
                    result["status"] = "success"
                    result["message"] = f"Rollback completed successfully in {rollback_time:.1f}s"
            else:
                result["status"] = "success"
                result["message"] = f"Rollback completed in {rollback_time:.1f}s (health check skipped)"

            # Check if we met the 30-second requirement
            if rollback_time > 30:
                result["warning"] = f"Rollback took {rollback_time:.1f}s, exceeding 30-second target"

            # Record rollback event
            self._record_rollback_event(endpoint_name, result)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["rollback_time_seconds"] = time.time() - start_time
            
            # Record failed rollback event
            self._record_rollback_event(endpoint_name, result)
            raise

        return result

    def _get_model_versions_for_rollback(self, model_name: str) -> List[Dict[str, Any]]:
        """Get available model versions for rollback.

        Args:
            model_name: Full model name (catalog.schema.model)

        Returns:
            List of model versions sorted by version number descending
        """
        try:
            import mlflow

            mlflow.set_registry_uri("databricks-uc")
            client = mlflow.tracking.MlflowClient()

            # Search for model versions
            versions = client.search_model_versions(f"name='{model_name}'")

            # Convert to list and sort by version number descending
            version_list = []
            for v in versions:
                version_list.append(
                    {
                        "version": v.version,
                        "creation_timestamp": v.creation_timestamp,
                        "status": v.status,
                        "run_id": v.run_id,
                    }
                )

            # Sort by version number (descending)
            version_list.sort(key=lambda x: int(x["version"]), reverse=True)

            return version_list

        except Exception as e:
            # Fallback - return empty list
            return []

    def _validate_endpoint_health(
        self, endpoint_name: str, timeout: int = 10
    ) -> Dict[str, Any]:
        """Validate endpoint health after rollback.

        Args:
            endpoint_name: Name of the endpoint
            timeout: Health check timeout in seconds

        Returns:
            Health status dictionary
        """
        import requests
        import time

        health_result = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "timeout": timeout,
        }

        try:
            # Get endpoint URL
            endpoint_url = f"{self.host}/serving-endpoints/{endpoint_name}/invocations"

            # Prepare test request
            test_payload = {
                "messages": [{"role": "user", "content": "health check"}],
                "max_tokens": 10,
            }

            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }

            # Send health check request
            start_time = time.time()
            response = requests.post(
                endpoint_url, json=test_payload, headers=headers, timeout=timeout
            )
            response_time = time.time() - start_time

            health_result["response_time_seconds"] = response_time
            health_result["status_code"] = response.status_code

            if response.status_code == 200:
                health_result["healthy"] = True
                health_result["message"] = "Endpoint responding normally"
            else:
                health_result["healthy"] = False
                health_result["message"] = f"Endpoint returned status {response.status_code}"

        except requests.exceptions.Timeout:
            health_result["healthy"] = False
            health_result["message"] = f"Health check timed out after {timeout}s"
        except Exception as e:
            health_result["healthy"] = False
            health_result["message"] = f"Health check failed: {str(e)}"

        return health_result

    def setup_automatic_rollback(
        self,
        endpoint_name: str,
        error_threshold: float = 0.05,
        latency_threshold_ms: int = 1000,
        monitoring_window_minutes: int = 5,
    ) -> Dict[str, Any]:
        """Configure automatic rollback triggers based on thresholds.

        Args:
            endpoint_name: Name of the serving endpoint
            error_threshold: Error rate threshold (0.05 = 5%)
            latency_threshold_ms: Latency threshold in milliseconds
            monitoring_window_minutes: Time window for monitoring metrics

        Returns:
            Configuration result
        """
        try:
            # Set up monitoring tags for automatic rollback
            self.client.serving_endpoints.set_tag(
                name=endpoint_name,
                key="auto_rollback_enabled",
                value="true",
            )
            self.client.serving_endpoints.set_tag(
                name=endpoint_name,
                key="auto_rollback_error_threshold",
                value=str(error_threshold),
            )
            self.client.serving_endpoints.set_tag(
                name=endpoint_name,
                key="auto_rollback_latency_threshold_ms",
                value=str(latency_threshold_ms),
            )
            self.client.serving_endpoints.set_tag(
                name=endpoint_name,
                key="auto_rollback_window_minutes",
                value=str(monitoring_window_minutes),
            )

            # Store current version as fallback
            endpoint = self.client.serving_endpoints.get(name=endpoint_name)
            current_version = endpoint.config.served_entities[0].entity_version

            self.client.serving_endpoints.set_tag(
                name=endpoint_name,
                key="auto_rollback_stable_version",
                value=current_version,
            )

            return {
                "status": "success",
                "endpoint_name": endpoint_name,
                "auto_rollback_enabled": True,
                "error_threshold": error_threshold,
                "latency_threshold_ms": latency_threshold_ms,
                "monitoring_window_minutes": monitoring_window_minutes,
                "stable_version": current_version,
                "message": "Automatic rollback configured successfully",
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": "Failed to configure automatic rollback",
            }

    def get_rollback_history(self, endpoint_name: str) -> List[Dict[str, Any]]:
        """Get rollback history for an endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            List of rollback events
        """
        try:
            # Get rollback history from tags
            tags_response = self.client.serving_endpoints.list_tags(name=endpoint_name)
            rollback_events = []

            # Parse rollback event tags
            for tag in tags_response.tags:
                if tag.key.startswith("rollback_event_"):
                    import json

                    event_data = json.loads(tag.value)
                    rollback_events.append(event_data)

            # Sort by timestamp descending
            rollback_events.sort(
                key=lambda x: x.get("timestamp", ""), reverse=True
            )

            return rollback_events

        except Exception:
            return []

    def _record_rollback_event(
        self, endpoint_name: str, event_data: Dict[str, Any]
    ) -> None:
        """Record a rollback event in endpoint tags.

        Args:
            endpoint_name: Name of the endpoint
            event_data: Rollback event details
        """
        try:
            import json

            # Create unique event key
            event_key = f"rollback_event_{int(time.time() * 1000)}"

            # Store event as JSON in tag
            self.client.serving_endpoints.set_tag(
                name=endpoint_name,
                key=event_key,
                value=json.dumps(event_data, default=str),
            )
        except Exception:
            # Non-critical - don't fail rollback if recording fails
            pass
