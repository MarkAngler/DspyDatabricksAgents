"""Databricks deployment implementation for DSPy agents with monitoring integration."""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import time
import warnings

from config.schema import AgentConfig
from core.agent import Agent
from deployment.model_signature import get_signature_for_config
from monitoring import (
    HealthCheckManager,
    MetricsCollector,
    ErrorTracker,
    DashboardIntegration
)
from monitoring.error_tracker import ErrorSeverity

# Import the original DatabricksDeployer
from .databricks_deployer import DatabricksDeployer as OriginalDatabricksDeployer


class DatabricksDeployer(OriginalDatabricksDeployer):
    """Extended Databricks deployer with monitoring capabilities."""
    
    def __init__(self, host: Optional[str] = None, token: Optional[str] = None):
        """Initialize Databricks deployer with monitoring.

        Args:
            host: Databricks workspace host (or from DATABRICKS_HOST env)
            token: Databricks access token (or from DATABRICKS_TOKEN env)
        """
        # Initialize parent class
        super().__init__(host, token)
        
        # Initialize monitoring components
        self.health_check_manager = HealthCheckManager()
        self.metrics_collector = MetricsCollector()
        self.error_tracker = ErrorTracker()
        
        # Set up error tracker with Databricks client
        self.error_tracker.set_databricks_client(self.client)
        
        # Create default error patterns
        self.error_tracker.create_default_patterns()
        
        # Initialize dashboard integration
        self.dashboard = DashboardIntegration(
            health_check_manager=self.health_check_manager,
            metrics_collector=self.metrics_collector,
            error_tracker=self.error_tracker,
            databricks_client=self.client
        )
    
    def deploy(
        self, config: AgentConfig, environment: str = "dev", dry_run: bool = False
    ) -> Dict[str, Any]:
        """Deploy agent to Databricks Model Serving with monitoring.

        Args:
            config: Agent configuration
            environment: Deployment environment
            dry_run: If True, only validate without deploying

        Returns:
            Deployment result with endpoint information
        """
        # Track deployment start time  
        deployment_start = time.time()
        
        try:
            # Log deployment start
            self.metrics_collector.increment_counter(
                "deployment_attempts",
                labels={"agent": config.name, "environment": environment}
            )
            
            # Get the base result structure
            result = super().deploy(config, environment, dry_run)
            
            if dry_run:
                return result
                
        except Exception as e:
            # Track deployment error
            self.error_tracker.track_deployment_error(
                endpoint_name=config.deployment.serving_endpoint if config.deployment else "unknown",
                error_type="deployment_failed",
                message=str(e),
                exception=e,
                context={
                    "agent": config.name,
                    "environment": environment
                }
            )
            
            # Record failure metric
            self.metrics_collector.increment_counter(
                "deployment_failures",
                labels={"agent": config.name, "environment": environment}
            )
            
            # Re-raise the exception
            raise
            
        # Add monitoring to the deployment process
        try:
            # Record deployment metrics
            deployment_duration_ms = (time.time() - deployment_start) * 1000
            self.metrics_collector.record_latency(
                "deployment_total",
                deployment_duration_ms,
                labels={"agent": config.name, "environment": environment}
            )
            
            if result.get("status") == "success":
                # Record success
                self.metrics_collector.increment_counter(
                    "deployment_success",
                    labels={"agent": config.name, "environment": environment}
                )
                
                # Set deployment gauge
                self.metrics_collector.set_gauge(
                    "deployment_active",
                    1.0,
                    labels={"agent": config.name, "endpoint": result.get("endpoint_name")}
                )
                
                # Register health check for the deployed endpoint
                if result.get("endpoint_url"):
                    health_check = self.health_check_manager.create_endpoint_check(
                        endpoint_name=result["endpoint_name"],
                        endpoint_url=result["endpoint_url"],
                        databricks_client=self.client
                    )
                    self.health_check_manager.register_check(
                        f"endpoint:{result['endpoint_name']}", 
                        health_check
                    )
                
                # Create monitoring notebook
                notebook_result = self.dashboard.create_monitoring_notebook(
                    agent_name=config.name,
                    endpoint_name=result["endpoint_name"],
                    catalog=result.get("catalog", "ml"),
                    schema=result.get("schema", "agents")
                )
                if notebook_result["status"] == "success":
                    result["monitoring_notebook"] = notebook_result["notebook_path"]
                    
        except Exception as e:
            # Track deployment error
            self.error_tracker.track_deployment_error(
                endpoint_name=result.get("endpoint_name", "unknown"),
                error_type="monitoring_setup_failed",
                message=str(e),
                exception=e,
                context={
                    "agent": config.name,
                    "environment": environment
                }
            )
            # Don't fail deployment due to monitoring setup issues
            warnings.warn(f"Failed to set up monitoring: {str(e)}")
            
        return result
    
    def get_monitoring_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Get comprehensive monitoring status for an endpoint.
        
        Args:
            endpoint_name: Name of the serving endpoint
            
        Returns:
            Dictionary with monitoring status
        """
        # Run health checks
        health_status = self.health_check_manager.run_all_checks()
        
        # Get recent metrics
        metrics_summary = self.dashboard._get_metrics_summary(60)
        
        # Get recent errors
        error_summary = self.error_tracker.get_error_summary(60)
        
        # Get endpoint metrics
        endpoint_metrics = self.get_endpoint_status(endpoint_name)
        
        return {
            "endpoint_name": endpoint_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": health_status,
            "metrics": metrics_summary,
            "errors": error_summary,
            "endpoint_metrics": endpoint_metrics,
            "overall_status": self.dashboard._calculate_overall_status({
                "health": health_status,
                "metrics": metrics_summary,
                "errors": error_summary
            })
        }
    
    def configure_alerts(
        self,
        endpoint_name: str,
        alert_webhook_url: Optional[str] = None,
        alert_email: Optional[str] = None,
        alert_thresholds: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Configure alerting for an endpoint.
        
        Args:
            endpoint_name: Name of the serving endpoint
            alert_webhook_url: Webhook URL for alerts (e.g., Slack, Teams)
            alert_email: Email address for alerts
            alert_thresholds: Custom thresholds for alerts
            
        Returns:
            Alert configuration result
        """
        try:
            # Set webhook URL if provided
            if alert_webhook_url:
                self.error_tracker.set_alert_webhook(alert_webhook_url)
            
            # Configure custom alert patterns if thresholds provided
            if alert_thresholds:
                from monitoring.error_tracker import ErrorPattern
                
                # High latency alert
                if "latency_p99_threshold_ms" in alert_thresholds:
                    pattern = ErrorPattern(
                        name=f"{endpoint_name}_high_latency",
                        error_types={"serving.high_latency"},
                        threshold_count=alert_thresholds.get("latency_threshold_count", 10),
                        time_window_seconds=alert_thresholds.get("latency_window_seconds", 300)
                    )
                    self.error_tracker.register_pattern(pattern)
                
                # Error rate alert
                if "error_rate_threshold_percent" in alert_thresholds:
                    pattern = ErrorPattern(
                        name=f"{endpoint_name}_high_error_rate",
                        error_types={"serving.request_error"},
                        threshold_count=alert_thresholds.get("error_threshold_count", 20),
                        time_window_seconds=alert_thresholds.get("error_window_seconds", 300)
                    )
                    self.error_tracker.register_pattern(pattern)
            
            return {
                "status": "success",
                "endpoint_name": endpoint_name,
                "alerts_configured": True,
                "webhook_url": alert_webhook_url,
                "email": alert_email,
                "thresholds": alert_thresholds
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "endpoint_name": endpoint_name,
                "error": str(e)
            }
    
    def export_monitoring_data(
        self,
        endpoint_name: str,
        export_format: str = "delta",
        catalog: Optional[str] = None,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export monitoring data for analysis.
        
        Args:
            endpoint_name: Name of the serving endpoint
            export_format: Format to export (delta, json, prometheus)
            catalog: Catalog for Delta tables
            schema: Schema for Delta tables
            
        Returns:
            Export result
        """
        try:
            if export_format == "delta":
                # Use provided catalog/schema or defaults
                catalog = catalog or "ml"
                schema = schema or "monitoring"
                
                return self.dashboard.export_to_delta_table(
                    catalog=catalog,
                    schema=schema,
                    table_prefix=f"{endpoint_name}_monitoring"
                )
                
            elif export_format == "json":
                # Export as JSON
                data = {
                    "endpoint_name": endpoint_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": self.metrics_collector.get_all_metrics(),
                    "errors": [e.to_dict() for e in self.error_tracker.get_recent_errors(minutes=1440)],  # Last 24h
                    "health": self.health_check_manager.run_all_checks()
                }
                
                # Save to workspace
                export_path = f"/Shared/dspy-agents/exports/{endpoint_name}_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.client.workspace.upload(
                    path=export_path,
                    content=json.dumps(data, indent=2).encode('utf-8'),
                    format="AUTO",
                    overwrite=True
                )
                
                return {
                    "status": "success",
                    "format": "json",
                    "export_path": export_path
                }
                
            elif export_format == "prometheus":
                # Export in Prometheus format
                prometheus_data = self.metrics_collector.export_prometheus_format()
                
                export_path = f"/Shared/dspy-agents/exports/{endpoint_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prom"
                self.client.workspace.upload(
                    path=export_path,
                    content=prometheus_data.encode('utf-8'),
                    format="AUTO",
                    overwrite=True
                )
                
                return {
                    "status": "success",
                    "format": "prometheus",
                    "export_path": export_path
                }
                
            else:
                return {
                    "status": "failed",
                    "error": f"Unsupported export format: {export_format}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def run_health_check(self, endpoint_name: str) -> Dict[str, Any]:
        """Run health check for a specific endpoint.
        
        Args:
            endpoint_name: Name of the serving endpoint
            
        Returns:
            Health check result
        """
        # Get endpoint info
        endpoint_status = self.get_endpoint_status(endpoint_name)
        
        if "error" in endpoint_status:
            return {
                "endpoint_name": endpoint_name,
                "status": "unhealthy",
                "error": endpoint_status["error"]
            }
        
        # Run specific health checks
        health_results = []
        
        # Check endpoint state
        if endpoint_status["state"] == "READY":
            health_results.append({
                "check": "endpoint_state",
                "status": "healthy",
                "message": "Endpoint is ready"
            })
        else:
            health_results.append({
                "check": "endpoint_state",
                "status": "unhealthy",
                "message": f"Endpoint state: {endpoint_status['state']}"
            })
        
        # Determine overall health
        overall_status = "healthy"
        if any(r["status"] == "unhealthy" for r in health_results):
            overall_status = "unhealthy"
        elif any(r["status"] == "degraded" for r in health_results):
            overall_status = "degraded"
        
        return {
            "endpoint_name": endpoint_name,
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": health_results
        }