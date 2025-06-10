# DSPy-Databricks Agents Monitoring

Comprehensive monitoring and observability for DSPy agents deployed on Databricks.

## Overview

The monitoring module provides production-grade observability features for DSPy agents:

- **Health Checks**: Asynchronous health monitoring for endpoints, models, and dependencies
- **Metrics Collection**: Performance metrics with Prometheus export support
- **Error Tracking**: Intelligent error tracking with pattern-based alerting
- **Dashboard Integration**: Pre-built Databricks notebooks and Grafana dashboards

## Quick Start

### Basic Deployment with Monitoring

```python
from dspy_databricks_agents.deployment import DatabricksDeployerWithMonitoring

# Initialize deployer with monitoring
deployer = DatabricksDeployerWithMonitoring()

# Deploy agent (monitoring is automatic)
result = deployer.deploy(config, environment="prod")

# Configure alerts
deployer.configure_alerts(
    endpoint_name=result["endpoint_name"],
    alert_webhook_url="https://hooks.slack.com/...",
    alert_thresholds={
        "latency_p99_threshold_ms": 1000,
        "error_rate_threshold_percent": 5
    }
)
```

### Standalone Monitoring Components

```python
from dspy_databricks_agents.monitoring import (
    HealthCheckManager,
    MetricsCollector,
    ErrorTracker,
    DashboardIntegration
)

# Health checks
health_manager = HealthCheckManager()
health_status = health_manager.run_all_checks()

# Metrics collection
metrics = MetricsCollector()
metrics.record_request("endpoint", latency_ms=150, status_code=200)

# Error tracking
error_tracker = ErrorTracker()
error_tracker.track_error("api.timeout", "Request timed out")
```

## Components

### Health Check Manager

Monitors the health of deployed endpoints and dependencies:

```python
# Register custom health check
def check_vector_store():
    # Your health check logic
    return ComponentHealth(
        name="vector_store",
        status=HealthStatus.HEALTHY,
        message="Vector store is accessible"
    )

health_manager.register_check("vector_store", check_vector_store)

# Run all checks
status = health_manager.run_all_checks()
print(f"Overall status: {status['status']}")
```

### Metrics Collector

Thread-safe metrics collection with multiple export formats:

```python
# Record metrics
metrics.increment_counter("requests_total", labels={"endpoint": "qa"})
metrics.record_latency("inference_time", 250.0)
metrics.set_gauge("active_connections", 42)

# Export metrics
prometheus_data = metrics.export_prometheus_format()
mlflow_metrics = metrics.export_mlflow_format()
```

### Error Tracker

Intelligent error tracking with pattern-based alerting:

```python
# Define alert pattern
pattern = ErrorPattern(
    name="high_error_rate",
    error_types={"api.error", "model.error"},
    threshold_count=10,
    time_window_seconds=300,
    alert_callback=lambda errors: send_alert(errors)
)
error_tracker.register_pattern(pattern)

# Track errors
error_tracker.track_error(
    error_type="api.error",
    message="Model inference failed",
    severity=ErrorSeverity.ERROR
)
```

### Dashboard Integration

Generate monitoring dashboards automatically:

```python
dashboard = DashboardIntegration(
    health_manager, metrics, error_tracker
)

# Create Databricks notebook
notebook_result = dashboard.create_monitoring_notebook(
    agent_name="qa_agent",
    endpoint_name="prod-qa-agent"
)

# Generate Grafana dashboard
grafana_config = dashboard.generate_grafana_dashboard()

# Export to Delta tables
dashboard.export_to_delta_table(
    catalog="ml",
    schema="monitoring"
)
```

## Monitoring Notebook

The generated monitoring notebook includes:

1. **Health Status Overview**: Real-time health checks
2. **Performance Metrics**: Latency, throughput, token usage
3. **Error Analysis**: Error rates and patterns
4. **Endpoint Metrics**: Model serving statistics
5. **Alerts Configuration**: Set up custom alerts
6. **Data Export**: Long-term storage options

## Alert Configuration

### Webhook Alerts

```python
deployer.configure_alerts(
    endpoint_name="prod-agent",
    alert_webhook_url="https://hooks.slack.com/services/...",
    alert_thresholds={
        "latency_p99_threshold_ms": 1000,
        "error_rate_threshold_percent": 5,
        "latency_threshold_count": 10,
        "latency_window_seconds": 300
    }
)
```

### Databricks Alerts

The system automatically creates Databricks SQL alerts when Delta tables are configured:

```sql
-- High latency alert
SELECT COUNT(*) as slow_requests
FROM ml.monitoring.agent_monitoring_metrics
WHERE metric_name = 'request_latency_ms' 
  AND value > 1000
  AND timestamp > current_timestamp() - INTERVAL 5 MINUTES
HAVING slow_requests > 10
```

## Best Practices

### 1. Configure Appropriate Thresholds

```python
# Development environment
dev_thresholds = {
    "latency_p99_threshold_ms": 2000,  # More lenient
    "error_rate_threshold_percent": 10
}

# Production environment
prod_thresholds = {
    "latency_p99_threshold_ms": 500,   # Stricter
    "error_rate_threshold_percent": 2
}
```

### 2. Use Labels for Better Insights

```python
metrics.record_request(
    endpoint_name="qa-agent",
    latency_ms=150,
    status_code=200,
    model_version="v2",
    labels={
        "environment": "prod",
        "region": "us-west",
        "customer_tier": "premium"
    }
)
```

### 3. Set Up Monitoring Early

```python
# Enable monitoring during deployment
result = deployer.deploy(config, enable_monitoring=True)

# Don't wait for issues - set up alerts immediately
deployer.configure_alerts(result["endpoint_name"], ...)
```

### 4. Export Data Regularly

```python
# Schedule regular exports
def export_monitoring_data():
    deployer.export_monitoring_data(
        endpoint_name="prod-agent",
        export_format="delta",
        catalog="ml",
        schema="monitoring"
    )

# Run daily
schedule.every().day.at("02:00").do(export_monitoring_data)
```

## Troubleshooting

### Common Issues

1. **Health check timeouts**
   ```python
   # Increase timeout for slow endpoints
   health_manager = HealthCheckManager(timeout_seconds=30.0)
   ```

2. **Metric memory usage**
   ```python
   # Limit history size
   metrics = MetricsCollector(max_history_size=10000)
   ```

3. **Alert fatigue**
   ```python
   # Adjust thresholds based on baseline
   baseline = metrics.get_metric_summary("latency_ms", window_seconds=3600)
   threshold = baseline.p95 * 1.5  # 50% above baseline
   ```

## Integration with MLflow

Monitoring integrates seamlessly with MLflow:

```python
import mlflow

# Log metrics during training
with mlflow.start_run():
    # Training code...
    
    # Log deployment metrics
    mlflow.log_metric("deployment_latency_p50", metrics.p50)
    mlflow.log_metric("deployment_latency_p99", metrics.p99)
    mlflow.log_metric("deployment_success_rate", success_rate)
```

## Performance Considerations

- Health checks run asynchronously to avoid blocking
- Metrics use circular buffers to limit memory usage
- Error tracking uses hash-based deduplication
- Dashboard queries are optimized for large datasets

## Security

- Webhook URLs are never logged or exported
- Sensitive error details can be redacted
- Delta tables support fine-grained access control
- Monitoring data respects Databricks workspace permissions