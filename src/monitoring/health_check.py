"""Health check endpoint handlers for DSPy agents."""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Health status for a single component."""
    
    def __init__(
        self, 
        name: str, 
        status: HealthStatus, 
        message: str = "",
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.message = message
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.message:
            result["message"] = self.message
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


class HealthCheckManager:
    """Manages health checks for deployed agents and endpoints."""
    
    def __init__(self, timeout_seconds: float = 10.0, max_workers: int = 5):
        """Initialize health check manager.
        
        Args:
            timeout_seconds: Timeout for individual health checks
            max_workers: Maximum concurrent health check workers
        """
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self._health_checks: Dict[str, Callable] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def register_check(self, name: str, check_func: Callable[[], ComponentHealth]):
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns ComponentHealth
        """
        self._health_checks[name] = check_func
        
    def unregister_check(self, name: str):
        """Unregister a health check.
        
        Args:
            name: Name of the health check to remove
        """
        self._health_checks.pop(name, None)
        
    async def check_endpoint_health(
        self, 
        endpoint_url: str,
        expected_status_code: int = 200,
        timeout: Optional[float] = None
    ) -> ComponentHealth:
        """Check health of a model serving endpoint.
        
        Args:
            endpoint_url: URL of the endpoint
            expected_status_code: Expected HTTP status code
            timeout: Request timeout (uses default if not specified)
            
        Returns:
            ComponentHealth object with status
        """
        import aiohttp
        
        timeout = timeout or self.timeout_seconds
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Construct health check URL
                health_url = endpoint_url.rstrip('/') + '/health'
                
                async with session.get(
                    health_url, 
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == expected_status_code:
                        return ComponentHealth(
                            name=f"endpoint:{endpoint_url}",
                            status=HealthStatus.HEALTHY,
                            message="Endpoint is responsive",
                            latency_ms=latency_ms,
                            metadata={"status_code": response.status}
                        )
                    else:
                        return ComponentHealth(
                            name=f"endpoint:{endpoint_url}",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Unexpected status code: {response.status}",
                            latency_ms=latency_ms,
                            metadata={"status_code": response.status}
                        )
                        
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=f"endpoint:{endpoint_url}",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s"
            )
        except Exception as e:
            return ComponentHealth(
                name=f"endpoint:{endpoint_url}",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}"
            )
    
    def check_model_registry_health(
        self,
        catalog: str,
        schema: str,
        model_name: str,
        databricks_client: Any
    ) -> ComponentHealth:
        """Check health of a model in Unity Catalog.
        
        Args:
            catalog: Catalog name
            schema: Schema name
            model_name: Model name
            databricks_client: Databricks workspace client
            
        Returns:
            ComponentHealth object with status
        """
        start_time = time.time()
        full_model_name = f"{catalog}.{schema}.{model_name}"
        
        try:
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient(registry_uri="databricks-uc")
            
            # Try to get model info
            model = client.get_registered_model(full_model_name)
            latency_ms = (time.time() - start_time) * 1000
            
            # Check if model has any versions
            versions = client.search_model_versions(f"name='{full_model_name}'")
            if not versions:
                return ComponentHealth(
                    name=f"model:{full_model_name}",
                    status=HealthStatus.DEGRADED,
                    message="Model has no versions",
                    latency_ms=latency_ms
                )
            
            # Get latest version status
            latest_version = max(versions, key=lambda v: int(v.version))
            
            if latest_version.status == "READY":
                return ComponentHealth(
                    name=f"model:{full_model_name}",
                    status=HealthStatus.HEALTHY,
                    message=f"Model version {latest_version.version} is ready",
                    latency_ms=latency_ms,
                    metadata={
                        "latest_version": latest_version.version,
                        "version_count": len(versions)
                    }
                )
            else:
                return ComponentHealth(
                    name=f"model:{full_model_name}",
                    status=HealthStatus.DEGRADED,
                    message=f"Latest version {latest_version.version} status: {latest_version.status}",
                    latency_ms=latency_ms,
                    metadata={"latest_version": latest_version.version}
                )
                
        except Exception as e:
            return ComponentHealth(
                name=f"model:{full_model_name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check model: {str(e)}"
            )
    
    def check_vector_store_health(
        self,
        catalog: str,
        schema: str,
        index_name: str,
        databricks_client: Any
    ) -> ComponentHealth:
        """Check health of a vector search index.
        
        Args:
            catalog: Catalog name
            schema: Schema name  
            index_name: Vector index name
            databricks_client: Databricks workspace client
            
        Returns:
            ComponentHealth object with status
        """
        start_time = time.time()
        full_index_name = f"{catalog}.{schema}.{index_name}"
        
        try:
            # Check if vector search client is available
            if not hasattr(databricks_client, 'vector_search'):
                return ComponentHealth(
                    name=f"vector_index:{full_index_name}",
                    status=HealthStatus.DEGRADED,
                    message="Vector search not available in workspace"
                )
            
            # Get index info
            index = databricks_client.vector_search.get_index(name=full_index_name)
            latency_ms = (time.time() - start_time) * 1000
            
            # Check index status
            if index.status.ready:
                return ComponentHealth(
                    name=f"vector_index:{full_index_name}",
                    status=HealthStatus.HEALTHY,
                    message="Vector index is ready",
                    latency_ms=latency_ms,
                    metadata={
                        "index_type": index.index_type,
                        "dimension": getattr(index, 'dimension', None)
                    }
                )
            else:
                return ComponentHealth(
                    name=f"vector_index:{full_index_name}",
                    status=HealthStatus.DEGRADED,
                    message=f"Index not ready: {index.status.message}",
                    latency_ms=latency_ms
                )
                
        except Exception as e:
            return ComponentHealth(
                name=f"vector_index:{full_index_name}",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check vector index: {str(e)}"
            )
    
    async def run_all_checks_async(self) -> Dict[str, Any]:
        """Run all registered health checks asynchronously.
        
        Returns:
            Dictionary with overall health status and individual component statuses
        """
        results: List[ComponentHealth] = []
        
        # Run registered checks
        tasks = []
        for name, check_func in self._health_checks.items():
            task = asyncio.create_task(self._run_check_async(name, check_func))
            tasks.append(task)
        
        # Wait for all checks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to ComponentHealth
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    valid_results.append(
                        ComponentHealth(
                            name="unknown",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Check failed: {str(result)}"
                        )
                    )
                else:
                    valid_results.append(result)
            
            results = valid_results
        
        # Determine overall health
        overall_status = self._determine_overall_health(results)
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results if r.status == HealthStatus.UNHEALTHY),
            }
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks synchronously.
        
        Returns:
            Dictionary with overall health status and individual component statuses
        """
        # Run async checks in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.run_all_checks_async())
    
    async def _run_check_async(
        self, 
        name: str, 
        check_func: Callable
    ) -> ComponentHealth:
        """Run a single health check asynchronously.
        
        Args:
            name: Name of the check
            check_func: Check function to run
            
        Returns:
            ComponentHealth result
        """
        try:
            # If check_func is async, await it
            if asyncio.iscoroutinefunction(check_func):
                return await check_func()
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._executor, check_func)
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}"
            )
    
    def _determine_overall_health(self, results: List[ComponentHealth]) -> HealthStatus:
        """Determine overall health from component results.
        
        Args:
            results: List of component health results
            
        Returns:
            Overall health status
        """
        if not results:
            return HealthStatus.HEALTHY
            
        # If any component is unhealthy, overall is unhealthy
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY
            
        # If any component is degraded, overall is degraded
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED
            
        return HealthStatus.HEALTHY
    
    def create_endpoint_check(
        self,
        endpoint_name: str,
        endpoint_url: str,
        databricks_client: Any
    ) -> Callable[[], ComponentHealth]:
        """Create a health check function for a serving endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            endpoint_url: URL of the endpoint
            databricks_client: Databricks client
            
        Returns:
            Health check function
        """
        def check_endpoint() -> ComponentHealth:
            try:
                # Get endpoint status from Databricks
                endpoint = databricks_client.serving_endpoints.get(name=endpoint_name)
                
                if endpoint.state.ready != "READY":
                    return ComponentHealth(
                        name=f"endpoint:{endpoint_name}",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Endpoint state: {endpoint.state.ready}",
                        metadata={"state": endpoint.state.ready}
                    )
                
                # Check if endpoint is serving properly
                # This would ideally make an actual request
                return ComponentHealth(
                    name=f"endpoint:{endpoint_name}",
                    status=HealthStatus.HEALTHY,
                    message="Endpoint is ready",
                    metadata={
                        "state": endpoint.state.ready,
                        "url": endpoint_url
                    }
                )
                
            except Exception as e:
                return ComponentHealth(
                    name=f"endpoint:{endpoint_name}",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check endpoint: {str(e)}"
                )
        
        return check_endpoint
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)