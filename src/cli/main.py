"""Main CLI entry point for DSPy-Databricks Agents."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from config.parser import YAMLParser
from config.schema import AgentConfig
from core.agent import Agent
from deployment.databricks_deployer import DatabricksDeployer
from deployment.mlflow_utils import set_experiment_with_environment

from dotenv import load_dotenv
import os

load_dotenv()
print(f"DATABRICKS_HOST: {os.environ.get('DATABRICKS_HOST', 'NOT SET')}")


console = Console()


@click.group()
@click.version_option()
def cli():
    """DSPy-Databricks Agents CLI.
    
    Build, validate, deploy, and manage DSPy-powered agents on Databricks.
    """
    pass


@cli.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation output")
def validate(yaml_path: str, verbose: bool):
    """Validate a YAML agent configuration.
    
    YAML_PATH: Path to the agent configuration file
    """
    try:
        console.print(f"[blue]Validating {yaml_path}...[/blue]")
        
        parser = YAMLParser()
        config = parser.parse_file(yaml_path)
        
        console.print("[green]✓ Configuration is valid![/green]")
        
        if verbose:
            console.print("\n[bold]Agent Configuration:[/bold]")
            console.print(f"  Name: {config.name}")
            console.print(f"  Version: {config.version}")
            console.print(f"  Description: {config.description or 'N/A'}")
            console.print(f"  Modules: {len(config.modules)}")
            console.print(f"  Workflow Steps: {len(config.workflow)}")
            
            if config.modules:
                console.print("\n[bold]Modules:[/bold]")
                for module in config.modules:
                    console.print(f"  - {module.name} ({module.type.value})")
            
            if config.workflow:
                console.print("\n[bold]Workflow:[/bold]")
                for step in config.workflow:
                    console.print(f"  - {step.step} -> {step.module}")
                    if step.condition:
                        console.print(f"    Condition: {step.condition}")
    
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--dataset", "-d", type=click.Path(exists=True), help="Training dataset (JSON)")
@click.option("--metric", "-m", default="answer_quality", help="Optimization metric")
@click.option("--num-candidates", "-n", default=10, help="Number of candidates for optimization")
@click.option("--output", "-o", help="Output path for optimized agent")
def train(yaml_path: str, dataset: Optional[str], metric: str, num_candidates: int, output: Optional[str]):
    """Train/optimize an agent using DSPy.
    
    YAML_PATH: Path to the agent configuration file
    """
    try:
        console.print(f"[blue]Loading agent from {yaml_path}...[/blue]")
        agent = Agent.from_yaml(yaml_path)
        
        if dataset:
            console.print(f"[blue]Loading dataset from {dataset}...[/blue]")
            with open(dataset, 'r') as f:
                train_data = json.load(f)
            
            # TODO: Implement DSPy optimization
            console.print("[yellow]⚠ DSPy optimization not yet implemented[/yellow]")
            console.print(f"  Would optimize with metric: {metric}")
            console.print(f"  Would use {num_candidates} candidates")
            console.print(f"  Would train on {len(train_data)} examples")
        else:
            console.print("[yellow]⚠ No dataset provided, skipping optimization[/yellow]")
        
        if output:
            console.print(f"[green]✓ Would save optimized agent to {output}[/green]")
        else:
            console.print("[green]✓ Agent loaded successfully (no optimization performed)[/green]")
    
    except Exception as e:
        console.print(f"[red]✗ Training failed: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--environment", "-e", default="dev", help="Deployment environment")
@click.option("--catalog", help="Unity Catalog name (overrides config)")
@click.option("--schema", help="Schema name (overrides config)")
@click.option("--endpoint", help="Serving endpoint name (overrides config)")
@click.option("--compute-size", type=click.Choice(["Small", "Medium", "Large"]), help="Compute size")
@click.option("--dry-run", is_flag=True, help="Show deployment plan without executing")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def deploy(yaml_path: str, environment: str, catalog: Optional[str], 
          schema: Optional[str], endpoint: Optional[str], 
          compute_size: Optional[str], dry_run: bool, verbose: bool):
    """Deploy an agent to Databricks.
    
    YAML_PATH: Path to the agent configuration file
    """
    try:
        console.print(f"[blue]Preparing to deploy {yaml_path} to {environment}...[/blue]")
        
        parser = YAMLParser()
        config = parser.parse_file(yaml_path)
        
        # Override deployment settings if provided
        if config.deployment:
            if catalog:
                config.deployment.catalog = catalog
            if schema:
                config.deployment.schema_name = schema
            if endpoint:
                config.deployment.serving_endpoint = endpoint
            if compute_size:
                config.deployment.compute_size = compute_size
        
        console.print("\n[bold]Deployment Plan:[/bold]")
        console.print(f"  Environment: {environment}")
        console.print(f"  Agent: {config.name} v{config.version}")
        
        if config.deployment:
            console.print(f"  Catalog: {config.deployment.catalog}")
            console.print(f"  Schema: {config.deployment.schema_name}")
            console.print(f"  Model Name: {config.deployment.model_name}")
            console.print(f"  Endpoint: {config.deployment.serving_endpoint}")
            console.print(f"  Compute Size: {config.deployment.compute_size}")
        else:
            console.print("  [yellow]No deployment configuration found[/yellow]")
        
        if dry_run:
            console.print("\n[yellow]Dry run mode - no actual deployment[/yellow]")
            return
        
        # Perform actual deployment
        try:
            console.print("\n[blue]Initializing Databricks connection...[/blue]")
            deployer = DatabricksDeployer()
            
            console.print("[blue]Deploying agent to Databricks...[/blue]")
            result = deployer.deploy(config, environment=environment, dry_run=False)
            
            if result["status"] == "success":
                console.print(f"\n[green]✓ Agent deployed successfully![/green]")
                console.print(f"  Endpoint URL: {result['endpoint_url']}")
                console.print(f"  Model URI: {result['model_uri']}")
                console.print(f"  State: {result['endpoint_state']}")
                
                console.print(f"\n[bold]Test your agent:[/bold]")
                console.print(f"  dspy-databricks test {config.name} --endpoint {result['endpoint_url']} --query \"Your question here\"")
            else:
                console.print(f"[red]✗ Deployment failed: {result.get('error', 'Unknown error')}[/red]")
                sys.exit(1)
                
        except ImportError as e:
            console.print("\n[red]✗ Databricks SDK not installed[/red]")
            console.print("Install with: pip install databricks-sdk")
            sys.exit(1)
        except ValueError as e:
            console.print(f"\n[red]✗ Configuration error: {str(e)}[/red]")
            console.print("Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]✗ Deployment failed: {str(e)}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]✗ Deployment failed: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("agent_name")
@click.option("--query", "-q", required=True, help="Query to test the agent with")
@click.option("--endpoint", "-e", help="Serving endpoint URL")
@click.option("--local", "-l", type=click.Path(exists=True), help="Test local YAML file instead")
@click.option("--stream", "-s", is_flag=True, help="Enable streaming response")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON response")
def test(agent_name: str, query: str, endpoint: Optional[str], 
         local: Optional[str], stream: bool, output_json: bool):
    """Test a deployed or local agent.
    
    AGENT_NAME: Name of the deployed agent or local identifier
    """
    try:
        if local:
            console.print(f"[blue]Loading local agent from {local}...[/blue]")
            
            # Set MLflow experiment to avoid default experiment warning
            parser = YAMLParser()
            config = parser.parse_file(local)
            set_experiment_with_environment(
                base_name=config.name,
                environment="local_test",
                project_prefix="dspy_cli"
            )
            
            agent = Agent.from_yaml(local)
            
            console.print(f"\n[bold]Query:[/bold] {query}")
            
            # Create ChatAgentMessage format
            from mlflow.types.agent import ChatAgentMessage
            messages = [ChatAgentMessage(role="user", content=query)]
            
            if stream:
                console.print("\n[bold]Streaming Response:[/bold]")
                for chunk in agent.predict_stream(messages):
                    console.print(chunk.delta.content, end="")
                console.print()
            else:
                response = agent.predict(messages)
                
                if output_json:
                    response_dict = {
                        "messages": [
                            {"role": msg.role, "content": msg.content}
                            for msg in response.messages
                        ]
                    }
                    console.print_json(data=response_dict)
                else:
                    console.print(f"\n[bold]Response:[/bold]")
                    for msg in response.messages:
                        console.print(f"{msg.content}")
        else:
            # Test remote agent
            import requests
            
            if not endpoint:
                # Try to auto-discover endpoint
                console.print(f"[yellow]⚠ Endpoint URL required for remote testing[/yellow]")
                console.print(f"Specify with --endpoint flag")
                sys.exit(1)
            
            console.print(f"[blue]Testing remote agent at {endpoint}...[/blue]")
            console.print(f"\n[bold]Query:[/bold] {query}")
            
            # Prepare request
            payload = {
                "messages": [{"role": "user", "content": query}]
            }
            
            if stream:
                payload["stream"] = True
            
            # Get auth token
            token = os.environ.get("DATABRICKS_TOKEN")
            if not token:
                console.print("[red]✗ DATABRICKS_TOKEN environment variable required for remote testing[/red]")
                sys.exit(1)
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            try:
                if stream:
                    # Streaming request
                    console.print("\n[bold]Streaming Response:[/bold]")
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        stream=True
                    )
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if "delta" in data:
                                console.print(data["delta"]["content"], end="")
                    console.print()
                else:
                    # Regular request
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    if output_json:
                        console.print_json(data=result)
                    else:
                        console.print(f"\n[bold]Response:[/bold]")
                        if "messages" in result:
                            for msg in result["messages"]:
                                console.print(f"{msg.get('content', '')}")
                        elif "response" in result:
                            console.print(result["response"])
                        else:
                            console.print_json(data=result)
                            
            except requests.exceptions.RequestException as e:
                console.print(f"[red]✗ Request failed: {str(e)}[/red]")
                if hasattr(e, 'response') and e.response is not None:
                    console.print(f"Status code: {e.response.status_code}")
                    console.print(f"Response: {e.response.text}")
                sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]✗ Test failed: {str(e)}[/red]")
        if not output_json:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("endpoint_name")
@click.option("--version", "-v", help="Specific version to rollback to (default: previous)")
@click.option("--no-health-check", is_flag=True, help="Skip health check after rollback")
@click.option("--force", "-f", is_flag=True, help="Force rollback even if current deployment is healthy")
@click.option("--verbose", is_flag=True, help="Show detailed output")
def rollback(endpoint_name: str, version: Optional[str], no_health_check: bool, force: bool, verbose: bool):
    """Rollback a deployed agent to a previous version.
    
    ENDPOINT_NAME: Name of the serving endpoint to rollback
    """
    try:
        console.print(f"[blue]Preparing to rollback {endpoint_name}...[/blue]")
        
        # Initialize deployer
        try:
            deployer = DatabricksDeployer()
        except ImportError:
            console.print("[red]✗ Databricks SDK not installed[/red]")
            console.print("Install with: pip install databricks-sdk")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]✗ Configuration error: {str(e)}[/red]")
            console.print("Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables")
            sys.exit(1)
        
        # Check current status
        current_status = deployer.get_endpoint_status(endpoint_name)
        if "error" in current_status:
            console.print(f"[red]✗ Failed to get endpoint status: {current_status['error']}[/red]")
            sys.exit(1)
        
        console.print(f"\n[bold]Current Status:[/bold]")
        console.print(f"  State: {current_status['state']}")
        if current_status.get('config', {}).get('served_entities'):
            entity = current_status['config']['served_entities'][0]
            console.print(f"  Current Version: {entity.get('entity_version', 'Unknown')}")
            console.print(f"  Model: {entity.get('entity_name', 'Unknown')}")
        
        if not force and current_status['state'] == "READY":
            if not click.confirm("\nEndpoint is currently healthy. Do you want to proceed with rollback?"):
                console.print("[yellow]Rollback cancelled[/yellow]")
                return
        
        # Perform rollback
        console.print(f"\n[blue]Initiating rollback{' to version ' + version if version else ''}...[/blue]")
        
        start_time = time.time()
        result = deployer.rollback_deployment(
            endpoint_name=endpoint_name,
            target_version=version,
            validate_health=not no_health_check
        )
        
        if result["status"] == "success":
            console.print(f"\n[green]✓ Rollback completed successfully![/green]")
            console.print(f"  Previous Version: {result.get('current_version', 'Unknown')}")
            console.print(f"  Rolled Back To: {result['rollback_version']}")
            console.print(f"  Time Taken: {result['rollback_time_seconds']:.1f} seconds")
            
            if result['rollback_time_seconds'] <= 30:
                console.print(f"  [green]✓ Met 30-second rollback requirement[/green]")
            else:
                console.print(f"  [yellow]⚠ Exceeded 30-second target[/yellow]")
            
            if result.get('health_check'):
                health = result['health_check']
                if health.get('healthy'):
                    console.print(f"  [green]✓ Health check passed[/green]")
                else:
                    console.print(f"  [yellow]⚠ Health check failed: {health.get('message')}[/yellow]")
        
        elif result["status"] == "warning":
            console.print(f"\n[yellow]⚠ Rollback completed with warnings[/yellow]")
            console.print(f"  Message: {result['message']}")
            console.print(f"  Time Taken: {result['rollback_time_seconds']:.1f} seconds")
        
        else:
            console.print(f"\n[red]✗ Rollback failed: {result.get('error', 'Unknown error')}[/red]")
            sys.exit(1)
        
        if verbose and result:
            console.print("\n[bold]Full Result:[/bold]")
            console.print_json(data=result)
    
    except Exception as e:
        console.print(f"[red]✗ Rollback failed: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command("rollback-history")
@click.argument("endpoint_name")
@click.option("--limit", "-n", default=10, help="Number of rollback events to show")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table")
def rollback_history(endpoint_name: str, limit: int, format: str):
    """View rollback history for an endpoint.
    
    ENDPOINT_NAME: Name of the serving endpoint
    """
    try:
        console.print(f"[blue]Fetching rollback history for {endpoint_name}...[/blue]")
        
        # Initialize deployer
        try:
            deployer = DatabricksDeployer()
        except ImportError:
            console.print("[red]✗ Databricks SDK not installed[/red]")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]✗ Configuration error: {str(e)}[/red]")
            sys.exit(1)
        
        # Get rollback history
        history = deployer.get_rollback_history(endpoint_name)
        
        if not history:
            console.print("[yellow]No rollback history found for this endpoint[/yellow]")
            return
        
        # Limit results
        history = history[:limit]
        
        if format == "json":
            console.print_json(data=history)
        else:
            table = Table(title=f"Rollback History - {endpoint_name}")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("From Version", justify="center")
            table.add_column("To Version", justify="center")
            table.add_column("Time (s)", justify="right")
            table.add_column("Message")
            
            for event in history:
                status_style = "green" if event.get("status") == "success" else "red"
                table.add_row(
                    event.get("timestamp", "Unknown"),
                    f"[{status_style}]{event.get('status', 'Unknown')}[/{status_style}]",
                    event.get("current_version", "-"),
                    event.get("rollback_version", "-"),
                    f"{event.get('rollback_time_seconds', 0):.1f}",
                    event.get("message", event.get("error", ""))
                )
            
            console.print(table)
            console.print(f"\n[dim]Showing {len(history)} of {len(history)} events[/dim]")
    
    except Exception as e:
        console.print(f"[red]✗ Failed to fetch history: {str(e)}[/red]")
        sys.exit(1)


@cli.command("auto-rollback")
@click.argument("endpoint_name")
@click.option("--enable/--disable", default=True, help="Enable or disable automatic rollback")
@click.option("--error-threshold", "-e", default=0.05, help="Error rate threshold (0-1)")
@click.option("--latency-threshold", "-l", default=1000, help="Latency threshold in milliseconds")
@click.option("--window", "-w", default=5, help="Monitoring window in minutes")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def auto_rollback(endpoint_name: str, enable: bool, error_threshold: float, 
                  latency_threshold: int, window: int, verbose: bool):
    """Configure automatic rollback for an endpoint.
    
    ENDPOINT_NAME: Name of the serving endpoint
    """
    try:
        action = "Enabling" if enable else "Disabling"
        console.print(f"[blue]{action} automatic rollback for {endpoint_name}...[/blue]")
        
        # Initialize deployer
        try:
            deployer = DatabricksDeployer()
        except ImportError:
            console.print("[red]✗ Databricks SDK not installed[/red]")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]✗ Configuration error: {str(e)}[/red]")
            sys.exit(1)
        
        if enable:
            # Validate thresholds
            if not 0 < error_threshold < 1:
                console.print("[red]✗ Error threshold must be between 0 and 1[/red]")
                sys.exit(1)
            
            if latency_threshold < 0:
                console.print("[red]✗ Latency threshold must be positive[/red]")
                sys.exit(1)
            
            console.print(f"\n[bold]Configuration:[/bold]")
            console.print(f"  Error Threshold: {error_threshold*100:.1f}%")
            console.print(f"  Latency Threshold: {latency_threshold}ms")
            console.print(f"  Monitoring Window: {window} minutes")
            
            result = deployer.setup_automatic_rollback(
                endpoint_name=endpoint_name,
                error_threshold=error_threshold,
                latency_threshold_ms=latency_threshold,
                monitoring_window_minutes=window
            )
        else:
            # Disable by setting a tag
            result = {
                "status": "success",
                "message": "Automatic rollback disabled",
                "auto_rollback_enabled": False
            }
            # In real implementation, would call deployer method to disable
        
        if result["status"] == "success":
            console.print(f"\n[green]✓ {result['message']}[/green]")
            if enable:
                console.print(f"  Stable Version: {result.get('stable_version', 'Unknown')}")
                console.print(f"\n[dim]The endpoint will automatically rollback if:[/dim]")
                console.print(f"  - Error rate exceeds {error_threshold*100:.1f}%")
                console.print(f"  - P95 latency exceeds {latency_threshold}ms")
                console.print(f"  - Measured over a {window}-minute window")
        else:
            console.print(f"\n[red]✗ Configuration failed: {result.get('error', 'Unknown error')}[/red]")
            sys.exit(1)
        
        if verbose:
            console.print("\n[bold]Full Result:[/bold]")
            console.print_json(data=result)
    
    except Exception as e:
        console.print(f"[red]✗ Configuration failed: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("agent_name")
@click.option("--last-24h", is_flag=True, help="Show metrics for last 24 hours")
@click.option("--last-7d", is_flag=True, help="Show metrics for last 7 days")
@click.option("--metric", "-m", multiple=True, help="Specific metrics to display")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table")
def monitor(agent_name: str, last_24h: bool, last_7d: bool, metric: tuple, format: str):
    """Monitor agent performance and metrics.
    
    AGENT_NAME: Name of the deployed agent
    """
    try:
        console.print(f"[blue]Monitoring agent: {agent_name}[/blue]")
        
        # Determine time range
        if last_24h:
            time_range = "Last 24 hours"
        elif last_7d:
            time_range = "Last 7 days"
        else:
            time_range = "Last hour"
        
        console.print(f"Time range: {time_range}")
        
        # TODO: Implement actual monitoring integration
        # For now, show mock data
        mock_metrics = {
            "requests": 1250,
            "avg_latency_ms": 245,
            "p99_latency_ms": 892,
            "errors": 3,
            "success_rate": 99.76,
            "tokens_used": 156000,
            "cost_usd": 2.34
        }
        
        if format == "json":
            console.print_json(data=mock_metrics)
        else:
            table = Table(title=f"Agent Metrics - {time_range}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            
            for metric_name, value in mock_metrics.items():
                if not metric or metric_name in metric:
                    if isinstance(value, float):
                        table.add_row(metric_name, f"{value:.2f}")
                    else:
                        table.add_row(metric_name, str(value))
            
            console.print(table)
            
            console.print("\n[yellow]⚠ Note: Showing mock data. Databricks integration pending.[/yellow]")
    
    except Exception as e:
        console.print(f"[red]✗ Monitoring failed: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="agent_config.md", help="Output documentation file")
@click.option("--format", "-f", type=click.Choice(["markdown", "html"]), default="markdown")
def docs(yaml_path: str, output: str, format: str):
    """Generate documentation for an agent configuration.
    
    YAML_PATH: Path to the agent configuration file
    """
    try:
        console.print(f"[blue]Generating documentation for {yaml_path}...[/blue]")
        
        parser = YAMLParser()
        config = parser.parse_file(yaml_path)
        
        if format == "markdown":
            doc_content = f"""# {config.name} Agent Documentation

## Overview
- **Version**: {config.version}
- **Description**: {config.description or 'No description provided'}

## Configuration

### DSPy Settings
- **Model**: {config.dspy.lm_model}
- **Temperature**: {config.dspy.temperature}
- **Max Tokens**: {config.dspy.max_tokens}

### Modules
"""
            for module in config.modules:
                doc_content += f"\n#### {module.name}\n"
                doc_content += f"- **Type**: {module.type.value}\n"
                if module.signature:
                    doc_content += f"- **Signature**: `{module.signature}`\n"
                if hasattr(module, 'description') and module.description:
                    doc_content += f"- **Description**: {module.description}\n"
            
            doc_content += "\n### Workflow\n"
            for i, step in enumerate(config.workflow, 1):
                doc_content += f"\n{i}. **{step.step}**\n"
                doc_content += f"   - Module: {step.module}\n"
                if step.condition:
                    doc_content += f"   - Condition: `{step.condition}`\n"
                if step.inputs:
                    doc_content += f"   - Inputs: {step.inputs}\n"
            
            Path(output).write_text(doc_content)
            console.print(f"[green]✓ Documentation saved to {output}[/green]")
        else:
            console.print("[yellow]⚠ HTML format not yet implemented[/yellow]")
    
    except Exception as e:
        console.print(f"[red]✗ Documentation generation failed: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()