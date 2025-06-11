"""Unit tests for Databricks deployment with monitoring functionality."""

import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

import pytest

from config.schema import (
    AgentConfig,
    DSPyConfig,
    ModuleConfig,
    ModuleType,
    WorkflowStep,
    DeploymentConfig,
)
from deployment import DatabricksDeployerWithMonitoring


class TestDatabricksDeployerWithMonitoring:
    """Test Databricks deployment with monitoring functionality."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration with deployment settings."""
        return AgentConfig(
            name="test-agent",
            version="1.0.0",
            dspy=DSPyConfig(lm_model="databricks-dbrx"),
            modules=[
                ModuleConfig(
                    name="qa", type=ModuleType.SIGNATURE, signature="question -> answer"
                )
            ],
            workflow=[WorkflowStep(step="answer", module="qa")],
            deployment=DeploymentConfig(
                catalog="test_catalog",
                schema="test_schema",
                model_name="test_model",
                serving_endpoint="test-endpoint",
                compute_size="Small",
            ),
        )

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_deployer_initialization_with_monitoring(self, mock_workspace_client):
        """Test deployer initialization with monitoring components."""
        deployer = DatabricksDeployerWithMonitoring()

        assert deployer.host == "https://test.databricks.com"
        assert deployer.token == "test-token"
        
        # Check monitoring components are initialized
        assert deployer.health_check_manager is not None
        assert deployer.metrics_collector is not None
        assert deployer.error_tracker is not None
        assert deployer.dashboard is not None

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_deploy_with_monitoring(self, mock_workspace_client, agent_config):
        """Test deployment with monitoring integration."""
        # Setup mocks
        deployer = DatabricksDeployerWithMonitoring()
        deployer._register_model = Mock(
            return_value="models:/catalog.schema.model/1"
        )
        deployer._create_serving_endpoint = Mock(
            return_value={
                "url": "https://test.databricks.com/serving-endpoints/test-endpoint/invocations",
                "state": "READY",
                "action": "created",
            }
        )
        deployer._configure_rate_limits = Mock()

        # Add rate limits to config
        agent_config.deployment.rate_limits = {"requests_per_minute": 100}

        result = deployer.deploy(agent_config, environment="prod", dry_run=False)

        assert result["status"] == "success"
        assert result["model_uri"] == "models:/catalog.schema.model/1"
        
        # Check monitoring metrics were recorded
        assert len(deployer.metrics_collector._counters) > 0
        assert any("deployment_attempts" in key for key in deployer.metrics_collector._counters)
        assert any("deployment_success" in key for key in deployer.metrics_collector._counters)
        
        # Check deployment gauge was set
        assert len(deployer.metrics_collector._gauges) > 0
        assert any("deployment_active" in key for key in deployer.metrics_collector._gauges)

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_deploy_with_error_tracking(self, mock_workspace_client, agent_config):
        """Test deployment error tracking."""
        deployer = DatabricksDeployerWithMonitoring()
        
        # Force an error
        deployer._register_model = Mock(side_effect=Exception("Registration failed"))
        
        with pytest.raises(Exception) as exc_info:
            deployer.deploy(agent_config)
        
        assert "Registration failed" in str(exc_info.value)
        
        # Check error was tracked
        assert len(deployer.error_tracker._errors) > 0
        error = deployer.error_tracker._errors[0]
        assert "deployment.deployment_failed" in error.error_type
        assert "Registration failed" in error.message
        
        # Check failure metrics
        assert any("deployment_failures" in key for key in deployer.metrics_collector._counters)

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_get_monitoring_status(self, mock_workspace_client):
        """Test getting monitoring status."""
        deployer = DatabricksDeployerWithMonitoring()
        
        # Mock endpoint status
        deployer.get_endpoint_status = Mock(return_value={
            "state": "READY",
            "name": "test-endpoint"
        })
        
        status = deployer.get_monitoring_status("test-endpoint")
        
        assert status["endpoint_name"] == "test-endpoint"
        assert "timestamp" in status
        assert "health" in status
        assert "metrics" in status
        assert "errors" in status
        assert "endpoint_metrics" in status
        assert "overall_status" in status

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_configure_alerts(self, mock_workspace_client):
        """Test alert configuration."""
        deployer = DatabricksDeployerWithMonitoring()
        
        result = deployer.configure_alerts(
            "test-endpoint",
            alert_webhook_url="https://hooks.slack.com/test",
            alert_thresholds={
                "latency_p99_threshold_ms": 1000,
                "error_rate_threshold_percent": 5
            }
        )
        
        assert result["status"] == "success"
        assert result["alerts_configured"] is True
        assert len(deployer.error_tracker._patterns) >= 2

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_export_monitoring_data_json(self, mock_workspace_client):
        """Test exporting monitoring data as JSON."""
        deployer = DatabricksDeployerWithMonitoring()
        
        result = deployer.export_monitoring_data(
            "test-endpoint",
            export_format="json"
        )
        
        assert result["status"] == "success"
        assert result["format"] == "json"
        assert "export_path" in result
        
        # Verify workspace upload was called
        deployer.client.workspace.upload.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_export_monitoring_data_prometheus(self, mock_workspace_client):
        """Test exporting monitoring data in Prometheus format."""
        deployer = DatabricksDeployerWithMonitoring()
        
        # Add some metrics
        deployer.metrics_collector.increment_counter("test_counter")
        deployer.metrics_collector.set_gauge("test_gauge", 42.0)
        
        result = deployer.export_monitoring_data(
            "test-endpoint",
            export_format="prometheus"
        )
        
        assert result["status"] == "success"
        assert result["format"] == "prometheus"
        assert "export_path" in result

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_run_health_check(self, mock_workspace_client):
        """Test running health check."""
        deployer = DatabricksDeployerWithMonitoring()
        
        # Mock endpoint status
        deployer.get_endpoint_status = Mock(return_value={
            "state": "READY",
            "name": "test-endpoint"
        })
        
        result = deployer.run_health_check("test-endpoint")
        
        assert result["endpoint_name"] == "test-endpoint"
        assert result["status"] == "healthy"
        assert len(result["checks"]) > 0
        assert result["checks"][0]["status"] == "healthy"

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_monitoring_notebook_creation(self, mock_workspace_client, agent_config):
        """Test monitoring notebook creation during deployment."""
        deployer = DatabricksDeployerWithMonitoring()
        
        # Setup mocks
        deployer._register_model = Mock(
            return_value="models:/catalog.schema.model/1"
        )
        deployer._create_serving_endpoint = Mock(
            return_value={
                "url": "https://test.databricks.com/serving-endpoints/test-endpoint/invocations",
                "state": "READY",
                "action": "created",
            }
        )
        deployer._configure_rate_limits = Mock()
        
        # Mock dashboard notebook creation
        deployer.dashboard.create_monitoring_notebook = Mock(
            return_value={
                "status": "success",
                "notebook_path": "/Shared/dspy-agents/monitoring/test-agent_monitor"
            }
        )

        result = deployer.deploy(agent_config, environment="prod", dry_run=False)

        assert result["status"] == "success"
        assert "monitoring_notebook" in result
        assert "/Shared/dspy-agents/monitoring/" in result["monitoring_notebook"]
        
        # Verify notebook was created
        deployer.dashboard.create_monitoring_notebook.assert_called_once_with(
            agent_name="test-agent",
            endpoint_name="test-endpoint",
            catalog="test_catalog",
            schema="test_schema"
        )

    @patch.dict(
        os.environ,
        {
            "DATABRICKS_HOST": "https://test.databricks.com",
            "DATABRICKS_TOKEN": "test-token",
        },
    )
    @patch("databricks.sdk.WorkspaceClient")
    def test_health_check_registration(self, mock_workspace_client, agent_config):
        """Test health check registration during deployment."""
        deployer = DatabricksDeployerWithMonitoring()
        
        # Setup mocks
        deployer._register_model = Mock(
            return_value="models:/catalog.schema.model/1"
        )
        deployer._create_serving_endpoint = Mock(
            return_value={
                "url": "https://test.databricks.com/serving-endpoints/test-endpoint/invocations",
                "state": "READY",
                "action": "created",
            }
        )
        deployer._configure_rate_limits = Mock()

        result = deployer.deploy(agent_config, environment="prod", dry_run=False)

        assert result["status"] == "success"
        
        # Check health check was registered
        assert f"endpoint:{result['endpoint_name']}" in deployer.health_check_manager._health_checks