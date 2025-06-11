"""Dashboard integration for monitoring DSPy agents."""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings

from .health_check import HealthCheckManager, HealthStatus
from .metrics import MetricsCollector
from .error_tracker import ErrorTracker, ErrorSeverity


class DashboardIntegration:
    """Integrates monitoring components with Databricks dashboards and notebooks."""
    
    def __init__(
        self,
        health_check_manager: HealthCheckManager,
        metrics_collector: MetricsCollector,
        error_tracker: ErrorTracker,
        databricks_client: Optional[Any] = None
    ):
        """Initialize dashboard integration.
        
        Args:
            health_check_manager: Health check manager instance
            metrics_collector: Metrics collector instance
            error_tracker: Error tracker instance
            databricks_client: Optional Databricks client
        """
        self.health_check_manager = health_check_manager
        self.metrics_collector = metrics_collector
        self.error_tracker = error_tracker
        self.databricks_client = databricks_client
        
    def get_dashboard_data(
        self,
        time_range_minutes: int = 60,
        include_health: bool = True,
        include_metrics: bool = True,
        include_errors: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data.
        
        Args:
            time_range_minutes: Time range for data collection
            include_health: Include health check data
            include_metrics: Include performance metrics
            include_errors: Include error data
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time_range_minutes": time_range_minutes,
        }
        
        if include_health:
            dashboard_data["health"] = self.health_check_manager.run_all_checks()
            
        if include_metrics:
            dashboard_data["metrics"] = self._get_metrics_summary(time_range_minutes)
            
        if include_errors:
            dashboard_data["errors"] = self.error_tracker.get_error_summary(time_range_minutes)
            
        # Add overall status
        dashboard_data["overall_status"] = self._calculate_overall_status(dashboard_data)
        
        return dashboard_data
        
    def create_monitoring_notebook(
        self,
        agent_name: str,
        endpoint_name: str,
        catalog: str = "ml",
        schema: str = "monitoring"
    ) -> Dict[str, Any]:
        """Create a Databricks notebook for monitoring an agent.
        
        Args:
            agent_name: Name of the agent
            endpoint_name: Name of the serving endpoint
            catalog: Catalog for storing monitoring data
            schema: Schema for storing monitoring data
            
        Returns:
            Notebook creation result
        """
        notebook_content = self._generate_monitoring_notebook(
            agent_name, endpoint_name, catalog, schema
        )
        
        if self.databricks_client:
            try:
                # Create notebook in Databricks workspace
                notebook_path = f"/Shared/dspy-agents/monitoring/{agent_name}_monitor"
                
                self.databricks_client.workspace.upload(
                    path=notebook_path,
                    content=notebook_content.encode('utf-8'),
                    format="SOURCE",
                    language="PYTHON",
                    overwrite=True
                )
                
                return {
                    "status": "success",
                    "notebook_path": notebook_path,
                    "message": f"Monitoring notebook created at {notebook_path}"
                }
                
            except Exception as e:
                return {
                    "status": "failed",
                    "error": str(e),
                    "message": "Failed to create monitoring notebook"
                }
        else:
            # Return notebook content if no client
            return {
                "status": "no_client",
                "notebook_content": notebook_content,
                "message": "Databricks client not configured, returning notebook content"
            }
            
    def export_to_delta_table(
        self,
        catalog: str,
        schema: str,
        table_prefix: str = "agent_monitoring"
    ) -> Dict[str, Any]:
        """Export monitoring data to Delta tables.
        
        Args:
            catalog: Catalog name
            schema: Schema name
            table_prefix: Prefix for table names
            
        Returns:
            Export result
        """
        if not self.databricks_client:
            return {
                "status": "failed",
                "error": "Databricks client not configured"
            }
            
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.types import (
                StructType, StructField, StringType, 
                FloatType, TimestampType, MapType
            )
            
            spark = SparkSession.builder.appName("DSPyAgentMonitoring").getOrCreate()
            
            # Export health data
            health_data = self.health_check_manager.run_all_checks()
            health_df = self._health_to_dataframe(spark, health_data)
            health_table = f"{catalog}.{schema}.{table_prefix}_health"
            health_df.write.mode("append").saveAsTable(health_table)
            
            # Export metrics data
            metrics_data = self.metrics_collector.get_all_metrics()
            metrics_df = self._metrics_to_dataframe(spark, metrics_data)
            metrics_table = f"{catalog}.{schema}.{table_prefix}_metrics"
            metrics_df.write.mode("append").saveAsTable(metrics_table)
            
            # Export error data
            errors = self.error_tracker.get_recent_errors(minutes=60)
            errors_df = self._errors_to_dataframe(spark, errors)
            errors_table = f"{catalog}.{schema}.{table_prefix}_errors"
            errors_df.write.mode("append").saveAsTable(errors_table)
            
            return {
                "status": "success",
                "tables_created": [health_table, metrics_table, errors_table],
                "message": "Monitoring data exported to Delta tables"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": "Failed to export monitoring data"
            }
            
    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration.
        
        Returns:
            Grafana dashboard JSON configuration
        """
        return {
            "dashboard": {
                "title": "DSPy Agent Monitoring",
                "panels": [
                    self._create_health_panel(),
                    self._create_latency_panel(),
                    self._create_throughput_panel(),
                    self._create_error_rate_panel(),
                    self._create_token_usage_panel(),
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
    def _get_metrics_summary(self, time_range_minutes: int) -> Dict[str, Any]:
        """Get metrics summary for dashboard.
        
        Args:
            time_range_minutes: Time range in minutes
            
        Returns:
            Metrics summary
        """
        window_seconds = time_range_minutes * 60
        
        # Get key metrics
        request_latency = self.metrics_collector.get_metric_summary(
            "request_latency_ms", window_seconds=window_seconds
        )
        
        token_usage = self.metrics_collector.get_metric_summary(
            "tokens_total", window_seconds=window_seconds
        )
        
        workflow_duration = self.metrics_collector.get_metric_summary(
            "workflow_duration_ms", window_seconds=window_seconds
        )
        
        # Get current metric values
        all_metrics = self.metrics_collector.get_all_metrics()
        
        summary = {
            "request_latency": request_latency.to_dict() if request_latency else None,
            "token_usage": token_usage.to_dict() if token_usage else None,
            "workflow_duration": workflow_duration.to_dict() if workflow_duration else None,
            "counters": all_metrics.get("counters", {}),
            "gauges": all_metrics.get("gauges", {})
        }
        
        # Calculate derived metrics
        if "request_total" in all_metrics["counters"]:
            total_requests = all_metrics["counters"]["request_total"]
            error_requests = all_metrics["counters"].get("request_errors", 0)
            summary["error_rate"] = (error_requests / total_requests * 100) if total_requests > 0 else 0
            summary["success_rate"] = 100 - summary["error_rate"]
            
        return summary
        
    def _calculate_overall_status(self, dashboard_data: Dict[str, Any]) -> str:
        """Calculate overall system status.
        
        Args:
            dashboard_data: Dashboard data
            
        Returns:
            Overall status string
        """
        # Check health status
        if "health" in dashboard_data:
            health_status = dashboard_data["health"].get("status", "healthy")
            if health_status == "unhealthy":
                return "critical"
            elif health_status == "degraded":
                return "warning"
                
        # Check error rate
        if "metrics" in dashboard_data:
            error_rate = dashboard_data["metrics"].get("error_rate", 0)
            if error_rate > 10:
                return "warning"
            if error_rate > 25:
                return "critical"
                
        # Check recent errors
        if "errors" in dashboard_data:
            total_errors = dashboard_data["errors"].get("total_errors", 0)
            critical_errors = dashboard_data["errors"]["by_severity"].get("critical", 0)
            
            if critical_errors > 0:
                return "critical"
            if total_errors > 100:
                return "warning"
                
        return "healthy"
        
    def _generate_monitoring_notebook(
        self,
        agent_name: str,
        endpoint_name: str,
        catalog: str,
        schema: str
    ) -> str:
        """Generate monitoring notebook content.
        
        Args:
            agent_name: Agent name
            endpoint_name: Endpoint name
            catalog: Catalog name
            schema: Schema name
            
        Returns:
            Notebook content as string
        """
        return f'''# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy Agent Monitoring: {agent_name}
# MAGIC 
# MAGIC This notebook monitors the performance and health of the {agent_name} agent.

# COMMAND ----------

# MAGIC %pip install dspy-databricks-agents

# COMMAND ----------

from dspy_databricks_agents.monitoring import DashboardIntegration
from dspy_databricks_agents.deployment import DatabricksDeployer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# COMMAND ----------

# Initialize monitoring
deployer = DatabricksDeployer()
dashboard = DashboardIntegration(
    health_check_manager=deployer.health_check_manager,
    metrics_collector=deployer.metrics_collector,
    error_tracker=deployer.error_tracker,
    databricks_client=deployer.client
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Health Status

# COMMAND ----------

# Get current health status
health_data = dashboard.health_check_manager.run_all_checks()
print(f"Overall Health: {{health_data['status']}}")
print(f"\\nComponent Status:")
for component in health_data['components']:
    print(f"  - {{component['name']}}: {{component['status']}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Metrics

# COMMAND ----------

# Get metrics for the last hour
metrics_data = dashboard.get_dashboard_data(time_range_minutes=60)

# Display latency statistics
if metrics_data['metrics']['request_latency']:
    latency = metrics_data['metrics']['request_latency']
    print(f"Request Latency (last hour):")
    print(f"  - Average: {{latency['avg']:.2f}}ms")
    print(f"  - P95: {{latency['p95']:.2f}}ms") 
    print(f"  - P99: {{latency['p99']:.2f}}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Error Analysis

# COMMAND ----------

# Get error summary
error_summary = dashboard.error_tracker.get_error_summary(time_window_minutes=60)
print(f"Total Errors (last hour): {{error_summary['total_errors']}}")
print(f"Unique Errors: {{error_summary['unique_errors']}}")

# Display top errors
if error_summary['top_errors']:
    print("\\nTop Errors:")
    for error in error_summary['top_errors'][:5]:
        print(f"  - {{error['error_type']}}: {{error['count']}} occurrences")
        print(f"    {{error['message'][:100]}}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Metrics

# COMMAND ----------

# Get endpoint-specific metrics
endpoint_metrics = deployer.get_endpoint_metrics("{endpoint_name}")
print(f"Endpoint: {endpoint_name}")
print(f"State: {{endpoint_metrics.get('state', 'Unknown')}}")

if 'served_models' in endpoint_metrics:
    print("\\nServed Models:")
    for model in endpoint_metrics['served_models']:
        print(f"  - {{model['model_name']}} (v{{model['model_version']}})")
        print(f"    Traffic: {{model.get('traffic_percentage', 100)}}%")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query monitoring tables
# MAGIC SELECT 
# MAGIC   date_trunc('minute', timestamp) as minute,
# MAGIC   AVG(value) as avg_latency,
# MAGIC   MAX(value) as max_latency,
# MAGIC   COUNT(*) as request_count
# MAGIC FROM {catalog}.{schema}.agent_monitoring_metrics
# MAGIC WHERE metric_name = 'request_latency_ms'
# MAGIC   AND timestamp > current_timestamp() - INTERVAL 1 HOUR
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1 DESC

# COMMAND ----------

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Latency over time
ax = axes[0, 0]
# Plot latency data here
ax.set_title('Request Latency Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Latency (ms)')

# Error rate
ax = axes[0, 1]
# Plot error rate here
ax.set_title('Error Rate')
ax.set_xlabel('Time')
ax.set_ylabel('Error Rate (%)')

# Token usage
ax = axes[1, 0]
# Plot token usage here
ax.set_title('Token Usage')
ax.set_xlabel('Time')
ax.set_ylabel('Tokens')

# Throughput
ax = axes[1, 1]
# Plot throughput here
ax.set_title('Request Throughput')
ax.set_xlabel('Time')
ax.set_ylabel('Requests/min')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alerts Configuration

# COMMAND ----------

# Configure alerts
from dspy_databricks_agents.monitoring import ErrorPattern, ErrorSeverity

# High latency alert
def latency_alert(errors):
    print(f"ALERT: High latency detected! {{len(errors)}} slow requests")
    # Send notification logic here

pattern = ErrorPattern(
    name="high_latency",
    error_types={{"request.timeout", "request.slow"}},
    threshold_count=10,
    time_window_seconds=300,
    alert_callback=latency_alert
)

dashboard.error_tracker.register_pattern(pattern)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Monitoring Data

# COMMAND ----------

# Export to Delta tables for long-term storage
export_result = dashboard.export_to_delta_table(
    catalog="{catalog}",
    schema="{schema}",
    table_prefix="agent_monitoring"
)

print(f"Export Status: {{export_result['status']}}")
if export_result['status'] == 'success':
    print(f"Tables created: {{export_result['tables_created']}}")
'''
        
    def _health_to_dataframe(self, spark: Any, health_data: Dict[str, Any]) -> Any:
        """Convert health data to Spark DataFrame.
        
        Args:
            spark: Spark session
            health_data: Health check data
            
        Returns:
            Spark DataFrame
        """
        from pyspark.sql import Row
        
        rows = []
        for component in health_data.get("components", []):
            row = Row(
                timestamp=datetime.fromisoformat(component["timestamp"]),
                component_name=component["name"],
                status=component["status"],
                message=component.get("message", ""),
                latency_ms=component.get("latency_ms"),
                metadata=json.dumps(component.get("metadata", {}))
            )
            rows.append(row)
            
        return spark.createDataFrame(rows)
        
    def _metrics_to_dataframe(self, spark: Any, metrics_data: Dict[str, Any]) -> Any:
        """Convert metrics data to Spark DataFrame.
        
        Args:
            spark: Spark session
            metrics_data: Metrics data
            
        Returns:
            Spark DataFrame
        """
        from pyspark.sql import Row
        
        rows = []
        timestamp = datetime.fromisoformat(metrics_data["timestamp"])
        
        # Convert counters
        for key, value in metrics_data.get("counters", {}).items():
            metric_name, labels = self._parse_metric_key(key)
            row = Row(
                timestamp=timestamp,
                metric_name=metric_name,
                metric_type="counter",
                value=float(value),
                labels=json.dumps(labels)
            )
            rows.append(row)
            
        # Convert gauges
        for key, value in metrics_data.get("gauges", {}).items():
            metric_name, labels = self._parse_metric_key(key)
            row = Row(
                timestamp=timestamp,
                metric_name=metric_name,
                metric_type="gauge",
                value=float(value),
                labels=json.dumps(labels)
            )
            rows.append(row)
            
        return spark.createDataFrame(rows)
        
    def _errors_to_dataframe(self, spark: Any, errors: List[Any]) -> Any:
        """Convert errors to Spark DataFrame.
        
        Args:
            spark: Spark session
            errors: List of error events
            
        Returns:
            Spark DataFrame
        """
        from pyspark.sql import Row
        
        rows = []
        for error in errors:
            row = Row(
                timestamp=error.timestamp,
                severity=error.severity.value,
                error_type=error.error_type,
                message=error.message,
                error_hash=error.error_hash,
                agent_name=error.agent_name,
                endpoint_name=error.endpoint_name,
                workflow_name=error.workflow_name,
                module_name=error.module_name,
                stack_trace=error.stack_trace,
                context=json.dumps(error.context)
            )
            rows.append(row)
            
        return spark.createDataFrame(rows)
        
    def _parse_metric_key(self, key: str) -> Tuple[str, Dict[str, str]]:
        """Parse metric key into name and labels.
        
        Args:
            key: Metric key
            
        Returns:
            Tuple of (metric_name, labels)
        """
        if "{" not in key:
            return key, {}
            
        metric_name = key.split("{")[0]
        label_str = key.split("{")[1].rstrip("}")
        
        labels = {}
        for part in label_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k] = v.strip('"')
                
        return metric_name, labels
        
    def _create_health_panel(self) -> Dict[str, Any]:
        """Create Grafana health status panel."""
        return {
            "title": "Health Status",
            "type": "stat",
            "targets": [{
                "expr": 'dspy_agent_health_status'
            }],
            "fieldConfig": {
                "defaults": {
                    "mappings": [{
                        "type": "value",
                        "options": {
                            "healthy": {"color": "green", "text": "Healthy"},
                            "degraded": {"color": "yellow", "text": "Degraded"}, 
                            "unhealthy": {"color": "red", "text": "Unhealthy"}
                        }
                    }]
                }
            }
        }
        
    def _create_latency_panel(self) -> Dict[str, Any]:
        """Create Grafana latency panel."""
        return {
            "title": "Request Latency",
            "type": "graph",
            "targets": [
                {"expr": 'histogram_quantile(0.5, dspy_request_latency_ms)', "legendFormat": "p50"},
                {"expr": 'histogram_quantile(0.95, dspy_request_latency_ms)', "legendFormat": "p95"},
                {"expr": 'histogram_quantile(0.99, dspy_request_latency_ms)', "legendFormat": "p99"}
            ]
        }
        
    def _create_throughput_panel(self) -> Dict[str, Any]:
        """Create Grafana throughput panel."""
        return {
            "title": "Request Throughput",
            "type": "graph",
            "targets": [{
                "expr": 'rate(dspy_request_total[1m])',
                "legendFormat": "Requests/min"
            }]
        }
        
    def _create_error_rate_panel(self) -> Dict[str, Any]:
        """Create Grafana error rate panel."""
        return {
            "title": "Error Rate",
            "type": "graph",
            "targets": [{
                "expr": 'rate(dspy_request_errors[5m]) / rate(dspy_request_total[5m]) * 100',
                "legendFormat": "Error %"
            }]
        }
        
    def _create_token_usage_panel(self) -> Dict[str, Any]:
        """Create Grafana token usage panel."""
        return {
            "title": "Token Usage",
            "type": "graph",
            "targets": [
                {"expr": 'rate(dspy_tokens_input[5m])', "legendFormat": "Input tokens/min"},
                {"expr": 'rate(dspy_tokens_output[5m])', "legendFormat": "Output tokens/min"}
            ]
        }