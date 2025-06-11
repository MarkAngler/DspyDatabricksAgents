"""Example: Deploy an agent with comprehensive monitoring to Databricks.

This example shows how to use the monitoring-aware deployer to:
1. Deploy an agent with monitoring integration
2. Configure alerts for performance issues
3. Export monitoring data for analysis
4. Check endpoint health status
"""

import os
from pathlib import Path

from dspy_databricks_agents.config.parser import YamlConfigParser
from dspy_databricks_agents.deployment import DatabricksDeployerWithMonitoring


def main():
    """Deploy an agent with monitoring to Databricks."""
    # Load agent configuration
    config_path = Path(__file__).parent / "simple_qa_agent.yaml"
    parser = YamlConfigParser()
    config = parser.parse_file(str(config_path))
    
    # Initialize the monitoring-aware deployer
    deployer = DatabricksDeployerWithMonitoring(
        host=os.environ.get("DATABRICKS_HOST"),
        token=os.environ.get("DATABRICKS_TOKEN")
    )
    
    print("Deploying agent with monitoring...")
    
    # Deploy the agent
    result = deployer.deploy(
        config=config,
        environment="dev",
        dry_run=False  # Set to True to test without actual deployment
    )
    
    if result["status"] == "success":
        print(f"✅ Agent deployed successfully!")
        print(f"   Endpoint URL: {result['endpoint_url']}")
        print(f"   Model URI: {result['model_uri']}")
        
        if "monitoring_notebook" in result:
            print(f"   Monitoring Notebook: {result['monitoring_notebook']}")
        
        # Configure alerts
        print("\nConfiguring alerts...")
        alert_result = deployer.configure_alerts(
            endpoint_name=result["endpoint_name"],
            alert_webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),  # Optional
            alert_thresholds={
                "latency_p99_threshold_ms": 1000,  # Alert if p99 latency > 1s
                "error_rate_threshold_percent": 5,  # Alert if error rate > 5%
                "latency_threshold_count": 10,     # Alert after 10 high latency requests
                "latency_window_seconds": 300,     # Within 5 minutes
                "error_threshold_count": 20,       # Alert after 20 errors
                "error_window_seconds": 300        # Within 5 minutes
            }
        )
        
        if alert_result["status"] == "success":
            print("✅ Alerts configured successfully!")
        
        # Get monitoring status
        print("\nChecking monitoring status...")
        status = deployer.get_monitoring_status(result["endpoint_name"])
        
        print(f"Overall Status: {status['overall_status']}")
        print(f"Health Status: {status['health']['status']}")
        
        # Export monitoring data
        print("\nExporting monitoring data...")
        
        # Export as JSON for analysis
        json_export = deployer.export_monitoring_data(
            endpoint_name=result["endpoint_name"],
            export_format="json"
        )
        if json_export["status"] == "success":
            print(f"✅ JSON export saved to: {json_export['export_path']}")
        
        # Export in Prometheus format for Grafana
        prom_export = deployer.export_monitoring_data(
            endpoint_name=result["endpoint_name"],
            export_format="prometheus"
        )
        if prom_export["status"] == "success":
            print(f"✅ Prometheus export saved to: {prom_export['export_path']}")
        
        # Export to Delta tables for long-term storage
        delta_export = deployer.export_monitoring_data(
            endpoint_name=result["endpoint_name"],
            export_format="delta",
            catalog="ml",
            schema="monitoring"
        )
        if delta_export["status"] == "success":
            print(f"✅ Delta tables created: {', '.join(delta_export['tables_created'])}")
        
        # Run health check
        print("\nRunning health check...")
        health = deployer.run_health_check(result["endpoint_name"])
        
        print(f"Health Check Result: {health['status']}")
        for check in health["checks"]:
            print(f"  - {check['check']}: {check['status']} - {check.get('message', '')}")
            
    else:
        print(f"❌ Deployment failed: {result.get('error', 'Unknown error')}")
        
        # Check error tracking
        error_summary = deployer.error_tracker.get_error_summary(time_window_minutes=5)
        print(f"\nRecent errors: {error_summary['total_errors']}")
        if error_summary["top_errors"]:
            print("Top errors:")
            for error in error_summary["top_errors"][:3]:
                print(f"  - {error['error_type']}: {error['count']} occurrences")


if __name__ == "__main__":
    main()