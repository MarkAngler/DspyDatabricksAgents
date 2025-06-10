"""Unit tests for monitoring components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

from dspy_databricks_agents.monitoring import (
    HealthCheckManager,
    MetricsCollector,
    ErrorTracker,
    DashboardIntegration,
)
from dspy_databricks_agents.monitoring.health_check import HealthStatus, ComponentHealth
from dspy_databricks_agents.monitoring.error_tracker import ErrorSeverity, ErrorEvent, ErrorPattern
from dspy_databricks_agents.monitoring.metrics import MetricPoint, MetricSummary


class TestHealthCheckManager:
    """Test HealthCheckManager functionality."""
    
    def test_init(self):
        """Test health check manager initialization."""
        manager = HealthCheckManager(timeout_seconds=5.0, max_workers=3)
        assert manager.timeout_seconds == 5.0
        assert manager.max_workers == 3
        assert len(manager._health_checks) == 0
    
    def test_register_check(self):
        """Test registering health checks."""
        manager = HealthCheckManager()
        
        def dummy_check():
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="Test healthy"
            )
        
        manager.register_check("test_check", dummy_check)
        assert "test_check" in manager._health_checks
        assert manager._health_checks["test_check"] == dummy_check
    
    def test_unregister_check(self):
        """Test unregistering health checks."""
        manager = HealthCheckManager()
        
        def dummy_check():
            return ComponentHealth("test", HealthStatus.HEALTHY)
        
        manager.register_check("test_check", dummy_check)
        manager.unregister_check("test_check")
        assert "test_check" not in manager._health_checks
    
    def test_check_model_registry_health(self):
        """Test model registry health check."""
        manager = HealthCheckManager()
        mock_client = Mock()
        
        # Mock MLflow client
        with patch("mlflow.tracking.MlflowClient") as mock_mlflow:
            mock_mlflow_instance = Mock()
            mock_mlflow.return_value = mock_mlflow_instance
            
            # Mock successful model retrieval
            mock_model = Mock()
            mock_mlflow_instance.get_registered_model.return_value = mock_model
            
            # Mock model versions
            mock_version = Mock()
            mock_version.version = "1"
            mock_version.status = "READY"
            mock_mlflow_instance.search_model_versions.return_value = [mock_version]
            
            health = manager.check_model_registry_health(
                catalog="ml",
                schema="agents",
                model_name="test_model",
                databricks_client=mock_client
            )
            
            assert health.name == "model:ml.agents.test_model"
            assert health.status == HealthStatus.HEALTHY
            assert "Model version 1 is ready" in health.message
    
    def test_run_all_checks(self):
        """Test running all health checks."""
        manager = HealthCheckManager()
        
        # Register multiple checks
        def healthy_check():
            return ComponentHealth("healthy", HealthStatus.HEALTHY)
        
        def degraded_check():
            return ComponentHealth("degraded", HealthStatus.DEGRADED)
        
        manager.register_check("check1", healthy_check)
        manager.register_check("check2", degraded_check)
        
        result = manager.run_all_checks()
        
        assert result["status"] == "degraded"  # Overall status
        assert len(result["components"]) == 2
        assert result["summary"]["total"] == 2
        assert result["summary"]["healthy"] == 1
        assert result["summary"]["degraded"] == 1
        assert result["summary"]["unhealthy"] == 0
    
    def test_create_endpoint_check(self):
        """Test creating endpoint health check."""
        manager = HealthCheckManager()
        mock_client = Mock()
        
        # Mock endpoint response
        mock_endpoint = Mock()
        mock_endpoint.state.ready = "READY"
        mock_client.serving_endpoints.get.return_value = mock_endpoint
        
        check_func = manager.create_endpoint_check(
            endpoint_name="test-endpoint",
            endpoint_url="https://example.com/endpoint",
            databricks_client=mock_client
        )
        
        # Execute the check
        health = check_func()
        
        assert health.name == "endpoint:test-endpoint"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "Endpoint is ready"


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_init(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_history_size=5000, aggregation_window_seconds=30)
        assert collector.max_history_size == 5000
        assert collector.aggregation_window_seconds == 30
        assert len(collector._metrics) == 0
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0
    
    def test_record_latency(self):
        """Test recording latency metrics."""
        collector = MetricsCollector()
        
        collector.record_latency("test_operation", 150.5, labels={"service": "api"})
        
        # Check metric was recorded
        key = 'test_operation_latency_ms{service="api"}'
        assert key in collector._metrics
        assert len(collector._metrics[key]) == 1
        assert collector._metrics[key][0].value == 150.5
    
    def test_increment_counter(self):
        """Test incrementing counters."""
        collector = MetricsCollector()
        
        collector.increment_counter("requests", labels={"status": "200"})
        collector.increment_counter("requests", value=2.0, labels={"status": "200"})
        
        key = 'requests{status="200"}'
        assert collector._counters[key] == 3.0
    
    def test_set_gauge(self):
        """Test setting gauge values."""
        collector = MetricsCollector()
        
        collector.set_gauge("active_connections", 42.0, labels={"server": "web1"})
        
        key = 'active_connections{server="web1"}'
        assert collector._gauges[key] == 42.0
    
    def test_record_request(self):
        """Test recording request metrics."""
        collector = MetricsCollector()
        
        collector.record_request(
            endpoint_name="test-endpoint",
            latency_ms=250.0,
            status_code=200,
            model_version="v1"
        )
        
        # Check latency was recorded
        assert any("request_latency_ms" in key for key in collector._metrics)
        
        # Check counter was incremented
        assert any("request_total" in key for key in collector._counters)
    
    def test_get_metric_summary(self):
        """Test getting metric summary statistics."""
        collector = MetricsCollector()
        
        # Record multiple values
        for i in range(10):
            collector.record_latency("test", float(i * 10))
        
        summary = collector.get_metric_summary("test_latency_ms")
        
        assert summary is not None
        assert summary.count == 10
        assert summary.min == 0.0
        assert summary.max == 90.0
        assert summary.avg == 45.0
        assert summary.p50 == 50.0  # Median of [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert summary.p95 == 90.0  # 95th percentile
        assert summary.p99 == 90.0  # 99th percentile
    
    def test_export_prometheus_format(self):
        """Test exporting metrics in Prometheus format."""
        collector = MetricsCollector()
        
        # Add some metrics
        collector.increment_counter("requests", labels={"status": "200"})
        collector.set_gauge("connections", 5.0)
        collector.record_latency("api_call", 100.0)
        
        prometheus_data = collector.export_prometheus_format()
        
        assert 'requests_total{status="200"}' in prometheus_data
        assert "connections 5.0" in prometheus_data
        assert "api_call_latency_ms_count 1" in prometheus_data


class TestErrorTracker:
    """Test ErrorTracker functionality."""
    
    def test_init(self):
        """Test error tracker initialization."""
        tracker = ErrorTracker(
            max_history_size=1000,
            default_alert_threshold=5,
            default_time_window_seconds=60
        )
        assert tracker.max_history_size == 1000
        assert tracker.default_alert_threshold == 5
        assert tracker.default_time_window_seconds == 60
    
    def test_track_error(self):
        """Test tracking errors."""
        tracker = ErrorTracker()
        
        error = tracker.track_error(
            error_type="api.timeout",
            message="Request timed out",
            severity=ErrorSeverity.ERROR,
            agent_name="test-agent",
            context={"url": "/api/v1/test"}
        )
        
        assert error.error_type == "api.timeout"
        assert error.message == "Request timed out"
        assert error.severity == ErrorSeverity.ERROR
        assert error.agent_name == "test-agent"
        assert error.error_hash is not None
        
        # Check error was stored
        assert len(tracker._errors) == 1
        assert tracker._error_counts[error.error_hash] == 1
    
    def test_track_agent_error(self):
        """Test tracking agent-specific errors."""
        tracker = ErrorTracker()
        
        error = tracker.track_agent_error(
            agent_name="test-agent",
            error_type="workflow_failed",
            message="Workflow execution failed"
        )
        
        assert error.error_type == "agent.workflow_failed"
        assert error.agent_name == "test-agent"
    
    def test_track_deployment_error(self):
        """Test tracking deployment-specific errors."""
        tracker = ErrorTracker()
        
        error = tracker.track_deployment_error(
            endpoint_name="test-endpoint",
            error_type="model_registration",
            message="Failed to register model"
        )
        
        assert error.error_type == "deployment.model_registration"
        assert error.endpoint_name == "test-endpoint"
    
    def test_get_recent_errors(self):
        """Test getting recent errors with filters."""
        tracker = ErrorTracker()
        
        # Track various errors
        tracker.track_error("api.error", "API error", severity=ErrorSeverity.ERROR)
        tracker.track_error("db.error", "DB error", severity=ErrorSeverity.WARNING)
        tracker.track_agent_error("agent1", "timeout", "Timeout")
        
        # Get all recent errors
        all_errors = tracker.get_recent_errors(minutes=5)
        assert len(all_errors) == 3
        
        # Filter by severity
        error_only = tracker.get_recent_errors(
            minutes=5,
            severity_filter={ErrorSeverity.ERROR}
        )
        assert len(error_only) == 2
        
        # Filter by type
        api_errors = tracker.get_recent_errors(
            minutes=5,
            error_type_filter="api"
        )
        assert len(api_errors) == 1
    
    def test_error_patterns_and_alerts(self):
        """Test error pattern matching and alerts."""
        tracker = ErrorTracker()
        alert_triggered = False
        
        def alert_callback(errors):
            nonlocal alert_triggered
            alert_triggered = True
        
        # Register pattern
        pattern = ErrorPattern(
            name="high_error_rate",
            error_types={"api.error"},
            threshold_count=3,
            time_window_seconds=60,
            alert_callback=alert_callback
        )
        tracker.register_pattern(pattern)
        
        # Track errors below threshold
        tracker.track_error("api.error", "Error 1")
        tracker.track_error("api.error", "Error 2")
        assert not alert_triggered
        
        # Track error that triggers alert
        tracker.track_error("api.error", "Error 3")
        assert alert_triggered
    
    def test_error_summary(self):
        """Test getting error summary."""
        tracker = ErrorTracker()
        
        # Track various errors
        for i in range(5):
            tracker.track_error("api.timeout", "Timeout", severity=ErrorSeverity.ERROR)
        for i in range(3):
            tracker.track_error("db.connection", "Connection lost", severity=ErrorSeverity.WARNING)
        
        summary = tracker.get_error_summary(time_window_minutes=60)
        
        assert summary["total_errors"] == 8
        assert summary["unique_errors"] == 2
        assert summary["by_severity"]["error"] == 5
        assert summary["by_severity"]["warning"] == 3
        assert summary["by_type"]["api.timeout"] == 5
        assert summary["by_type"]["db.connection"] == 3


class TestDashboardIntegration:
    """Test DashboardIntegration functionality."""
    
    def test_init(self):
        """Test dashboard integration initialization."""
        health_manager = Mock()
        metrics_collector = Mock()
        error_tracker = Mock()
        
        dashboard = DashboardIntegration(
            health_check_manager=health_manager,
            metrics_collector=metrics_collector,
            error_tracker=error_tracker
        )
        
        assert dashboard.health_check_manager == health_manager
        assert dashboard.metrics_collector == metrics_collector
        assert dashboard.error_tracker == error_tracker
    
    def test_get_dashboard_data(self):
        """Test getting dashboard data."""
        # Mock components
        health_manager = Mock()
        health_manager.run_all_checks.return_value = {
            "status": "healthy",
            "components": []
        }
        
        metrics_collector = Mock()
        error_tracker = Mock()
        error_tracker.get_error_summary.return_value = {
            "total_errors": 0,
            "by_severity": {}
        }
        
        dashboard = DashboardIntegration(
            health_manager, metrics_collector, error_tracker
        )
        
        # Mock internal method
        dashboard._get_metrics_summary = Mock(return_value={})
        
        data = dashboard.get_dashboard_data(
            time_range_minutes=30,
            include_health=True,
            include_metrics=True,
            include_errors=True
        )
        
        assert "timestamp" in data
        assert "time_range_minutes" in data
        assert data["time_range_minutes"] == 30
        assert "health" in data
        assert "metrics" in data
        assert "errors" in data
        assert "overall_status" in data
    
    def test_calculate_overall_status(self):
        """Test calculating overall system status."""
        dashboard = DashboardIntegration(Mock(), Mock(), Mock())
        
        # Test healthy status
        data = {
            "health": {"status": "healthy"},
            "metrics": {"error_rate": 2.0},
            "errors": {"total_errors": 5, "by_severity": {"critical": 0}}
        }
        assert dashboard._calculate_overall_status(data) == "healthy"
        
        # Test degraded due to health
        data["health"]["status"] = "degraded"
        assert dashboard._calculate_overall_status(data) == "warning"
        
        # Test critical due to errors
        data["health"]["status"] = "healthy"
        data["errors"]["by_severity"]["critical"] = 1
        assert dashboard._calculate_overall_status(data) == "critical"
        
        # Test warning due to error rate
        data["errors"]["by_severity"]["critical"] = 0
        data["metrics"]["error_rate"] = 15.0
        assert dashboard._calculate_overall_status(data) == "warning"
    
    def test_generate_monitoring_notebook(self):
        """Test generating monitoring notebook content."""
        dashboard = DashboardIntegration(Mock(), Mock(), Mock())
        
        notebook_content = dashboard._generate_monitoring_notebook(
            agent_name="test-agent",
            endpoint_name="test-endpoint",
            catalog="ml",
            schema="agents"
        )
        
        assert "# DSPy Agent Monitoring: test-agent" in notebook_content
        assert "test-endpoint" in notebook_content
        assert 'catalog="ml"' in notebook_content  # Direct values in export call
        assert 'schema="agents"' in notebook_content  # Direct values in export call
        assert "%pip install dspy-databricks-agents" in notebook_content
    
    def test_generate_grafana_dashboard(self):
        """Test generating Grafana dashboard configuration."""
        dashboard = DashboardIntegration(Mock(), Mock(), Mock())
        
        grafana_config = dashboard.generate_grafana_dashboard()
        
        assert "dashboard" in grafana_config
        assert grafana_config["dashboard"]["title"] == "DSPy Agent Monitoring"
        assert len(grafana_config["dashboard"]["panels"]) == 5
        
        # Check panel types
        panel_titles = [p["title"] for p in grafana_config["dashboard"]["panels"]]
        assert "Health Status" in panel_titles
        assert "Request Latency" in panel_titles
        assert "Request Throughput" in panel_titles
        assert "Error Rate" in panel_titles
        assert "Token Usage" in panel_titles


class TestMonitoringIntegration:
    """Test integration between monitoring components."""
    
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        # Create components
        health_manager = HealthCheckManager()
        metrics_collector = MetricsCollector()
        error_tracker = ErrorTracker()
        dashboard = DashboardIntegration(
            health_manager, metrics_collector, error_tracker
        )
        
        # Simulate activity
        metrics_collector.record_request(
            endpoint_name="test-endpoint",
            latency_ms=150.0,
            status_code=200
        )
        
        error_tracker.track_error(
            error_type="api.warning",
            message="High latency detected",
            severity=ErrorSeverity.WARNING
        )
        
        # Register health check
        def endpoint_check():
            return ComponentHealth(
                name="endpoint:test",
                status=HealthStatus.HEALTHY,
                message="Endpoint healthy"
            )
        health_manager.register_check("endpoint", endpoint_check)
        
        # Get dashboard data
        data = dashboard.get_dashboard_data()
        
        assert data["overall_status"] == "healthy"
        assert data["health"]["status"] == "healthy"
        assert data["errors"]["total_errors"] == 1
        
    def test_monitoring_with_databricks_client(self):
        """Test monitoring with Databricks client integration."""
        mock_client = Mock()
        
        # Create components with client
        health_manager = HealthCheckManager()
        metrics_collector = MetricsCollector()
        error_tracker = ErrorTracker()
        error_tracker.set_databricks_client(mock_client)
        
        dashboard = DashboardIntegration(
            health_manager, metrics_collector, error_tracker,
            databricks_client=mock_client
        )
        
        # Test notebook creation
        mock_client.workspace.upload.return_value = None
        
        result = dashboard.create_monitoring_notebook(
            agent_name="test-agent",
            endpoint_name="test-endpoint"
        )
        
        assert result["status"] == "success"
        assert "/Shared/dspy-agents/monitoring/test-agent_monitor" in result["notebook_path"]
        mock_client.workspace.upload.assert_called_once()