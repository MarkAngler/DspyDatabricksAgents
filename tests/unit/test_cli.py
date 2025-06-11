"""Unit tests for CLI interface."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from cli.main import cli
from config.schema import (
    AgentConfig,
    DeploymentConfig,
    DSPyConfig,
    ModuleConfig,
    ModuleType,
    WorkflowStep,
)


class TestCLIValidate:
    """Test the validate command."""

    def test_validate_valid_config(self, tmp_path):
        """Test validating a valid configuration."""
        yaml_content = """
agent:
  name: test-agent
  version: 1.0.0
  
  dspy:
    lm_model: gpt-4
    temperature: 0.7
    
  modules:
    - name: responder
      type: signature
      signature: "query -> response"
      
  workflow:
    - step: respond
      module: responder
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(yaml_file)])
        
        assert result.exit_code == 0
        assert "Configuration is valid!" in result.output

    def test_validate_verbose_output(self, tmp_path):
        """Test validate with verbose output."""
        yaml_content = """
agent:
  name: test-agent
  version: 1.0.0
  description: Test agent for validation
  
  dspy:
    lm_model: gpt-4
    
  modules:
    - name: classifier
      type: signature
      signature: "text -> category"
    - name: generator
      type: chain_of_thought
      signature: "prompt -> response"
      
  workflow:
    - step: classify
      module: classifier
    - step: generate
      module: generator
      condition: "$classify.category == 'important'"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(yaml_file), "--verbose"])
        
        assert result.exit_code == 0
        assert "Configuration is valid!" in result.output
        assert "Name: test-agent" in result.output
        assert "Version: 1.0.0" in result.output
        assert "Description: Test agent for validation" in result.output
        assert "Modules: 2" in result.output
        assert "Workflow Steps: 2" in result.output
        assert "classifier (signature)" in result.output
        assert "generator (chain_of_thought)" in result.output
        assert "Condition:" in result.output

    def test_validate_invalid_config(self, tmp_path):
        """Test validating an invalid configuration."""
        yaml_content = """
agent:
  name: test-agent
  # Missing required fields
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(yaml_file)])
        
        assert result.exit_code == 1
        assert "Validation failed" in result.output

    def test_validate_nonexistent_file(self):
        """Test validating a non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "nonexistent.yaml"])
        
        assert result.exit_code == 2  # Click's exit code for missing file


class TestCLITrain:
    """Test the train command."""

    @patch('dspy_databricks_agents.cli.main.Agent')
    def test_train_without_dataset(self, mock_agent, tmp_path):
        """Test train command without dataset."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("agent:\n  name: test\n  version: 1.0")
        
        runner = CliRunner()
        result = runner.invoke(cli, ["train", str(yaml_file)])
        
        assert result.exit_code == 0
        assert "No dataset provided" in result.output
        mock_agent.from_yaml.assert_called_once()

    @patch('dspy_databricks_agents.cli.main.Agent')
    def test_train_with_dataset(self, mock_agent, tmp_path):
        """Test train command with dataset."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("agent:\n  name: test\n  version: 1.0")
        
        dataset_file = tmp_path / "train.json"
        dataset_file.write_text(json.dumps([
            {"input": "test1", "output": "response1"},
            {"input": "test2", "output": "response2"}
        ]))
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "train", str(yaml_file),
            "--dataset", str(dataset_file),
            "--metric", "accuracy",
            "--num-candidates", "5",
            "--output", "optimized.yaml"
        ])
        
        assert result.exit_code == 0
        assert "Loading dataset" in result.output
        assert "Would optimize with metric: accuracy" in result.output
        assert "Would use 5 candidates" in result.output
        assert "Would train on 2 examples" in result.output


class TestCLIDeploy:
    """Test the deploy command."""

    def test_deploy_dry_run(self, tmp_path):
        """Test deploy command in dry-run mode."""
        yaml_content = """
agent:
  name: test-agent
  version: 1.0.0
  
  dspy:
    lm_model: gpt-4
    
  modules:
    - name: test
      type: signature
      signature: "a -> b"
      
  workflow:
    - step: test
      module: test
      
  deployment:
    catalog: ml
    schema: agents
    model_name: test_model
    serving_endpoint: test-endpoint
    compute_size: Small
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "deploy", str(yaml_file),
            "--environment", "prod",
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert "Deployment Plan:" in result.output
        assert "Environment: prod" in result.output
        assert "Catalog: ml" in result.output
        assert "Schema: agents" in result.output
        assert "Dry run mode" in result.output

    def test_deploy_with_overrides(self, tmp_path):
        """Test deploy with parameter overrides."""
        yaml_content = """
agent:
  name: test-agent
  version: 1.0.0
  
  dspy:
    lm_model: gpt-4
    
  modules:
    - name: test
      type: signature
      signature: "a -> b"
      
  workflow:
    - step: test
      module: test
      
  deployment:
    catalog: ml
    schema: agents
    model_name: test_model
    serving_endpoint: test-endpoint
    compute_size: Small
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "deploy", str(yaml_file),
            "--catalog", "prod_ml",
            "--schema", "prod_agents",
            "--endpoint", "prod-endpoint",
            "--compute-size", "Large",
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert "Catalog: prod_ml" in result.output
        assert "Schema: prod_agents" in result.output
        assert "Endpoint: prod-endpoint" in result.output
        assert "Compute Size: Large" in result.output


class TestCLITest:
    """Test the test command."""

    @patch('dspy_databricks_agents.cli.main.Agent')
    def test_local_agent_test(self, mock_agent_class, tmp_path):
        """Test running a local agent."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("agent:\n  name: test\n  version: 1.0")
        
        # Mock the agent instance
        mock_agent = Mock()
        mock_agent_class.from_yaml.return_value = mock_agent
        
        # Mock predict response
        from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
        mock_agent.predict.return_value = ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content="Test response", id="123")]
        )
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "test", "test-agent",
            "--query", "Hello world",
            "--local", str(yaml_file)
        ])
        
        assert result.exit_code == 0
        assert "Query: Hello world" in result.output
        assert "Test response" in result.output
        mock_agent_class.from_yaml.assert_called_once_with(str(yaml_file))

    @patch('dspy_databricks_agents.cli.main.Agent')
    def test_local_agent_test_streaming(self, mock_agent_class, tmp_path):
        """Test running a local agent with streaming."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("agent:\n  name: test\n  version: 1.0")
        
        # Mock the agent instance
        mock_agent = Mock()
        mock_agent_class.from_yaml.return_value = mock_agent
        
        # Mock streaming response
        from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage
        def mock_stream(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield ChatAgentChunk(
                    delta=ChatAgentMessage(role="assistant", content=chunk, id="123")
                )
        
        mock_agent.predict_stream.side_effect = mock_stream
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "test", "test-agent",
            "--query", "Hello",
            "--local", str(yaml_file),
            "--stream"
        ])
        
        assert result.exit_code == 0
        assert "Streaming Response:" in result.output

    @patch('dspy_databricks_agents.cli.main.Agent')
    def test_local_agent_test_json_output(self, mock_agent_class, tmp_path):
        """Test running a local agent with JSON output."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("agent:\n  name: test\n  version: 1.0")
        
        # Mock the agent instance
        mock_agent = Mock()
        mock_agent_class.from_yaml.return_value = mock_agent
        
        # Mock predict response
        from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
        mock_agent.predict.return_value = ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content="Test response", id="123")]
        )
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "test", "test-agent",
            "--query", "Hello",
            "--local", str(yaml_file),
            "--json"
        ])
        
        assert result.exit_code == 0
        # Rich formats JSON output, so check for key parts
        assert '"messages"' in result.output
        assert '"role"' in result.output
        assert '"content"' in result.output

    def test_remote_agent_test_placeholder(self):
        """Test remote agent testing (placeholder)."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "test", "remote-agent",
            "--query", "Hello"
            # No endpoint provided to test error handling
        ])
        
        assert result.exit_code == 1
        assert "Endpoint URL required for remote testing" in result.output


class TestCLIMonitor:
    """Test the monitor command."""

    def test_monitor_default(self):
        """Test monitor with default settings."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor", "test-agent"])
        
        assert result.exit_code == 0
        assert "Monitoring agent: test-agent" in result.output
        assert "Time range: Last hour" in result.output
        assert "Agent Metrics" in result.output

    def test_monitor_last_24h(self):
        """Test monitor with last 24 hours."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor", "test-agent", "--last-24h"])
        
        assert result.exit_code == 0
        assert "Time range: Last 24 hours" in result.output

    def test_monitor_specific_metrics(self):
        """Test monitor with specific metrics."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "monitor", "test-agent",
            "--metric", "requests",
            "--metric", "avg_latency_ms"
        ])
        
        assert result.exit_code == 0
        assert "requests" in result.output
        assert "avg_latency_ms" in result.output

    def test_monitor_json_format(self):
        """Test monitor with JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "monitor", "test-agent",
            "--format", "json"
        ])
        
        assert result.exit_code == 0
        # Check for JSON structure
        assert '"requests"' in result.output
        assert '"avg_latency_ms"' in result.output


class TestCLIDocs:
    """Test the docs command."""

    def test_generate_docs(self, tmp_path):
        """Test generating documentation."""
        yaml_content = """
agent:
  name: test-agent
  version: 1.0.0
  description: A test agent for documentation
  
  dspy:
    lm_model: gpt-4
    temperature: 0.7
    max_tokens: 100
    
  modules:
    - name: classifier
      type: signature
      signature: "text -> category"
      description: Classifies input text
      
    - name: generator
      type: chain_of_thought
      signature: "prompt -> response"
      
  workflow:
    - step: classify
      module: classifier
      inputs:
        text: "$input.query"
        
    - step: generate
      module: generator
      condition: "$classify.category == 'important'"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        output_file = tmp_path / "docs.md"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "docs", str(yaml_file),
            "--output", str(output_file)
        ])
        
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            if result.exception:
                print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        assert "Documentation saved" in result.output
        
        # Check generated documentation
        doc_content = output_file.read_text()
        assert "# test-agent Agent Documentation" in doc_content
        assert "**Version**: 1.0.0" in doc_content
        assert "A test agent for documentation" in doc_content
        assert "**Model**: gpt-4" in doc_content
        assert "classifier" in doc_content
        assert "Workflow" in doc_content

    def test_generate_docs_html_placeholder(self, tmp_path):
        """Test HTML documentation generation (placeholder)."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("agent:\n  name: test\n  version: 1.0.0\n  dspy:\n    lm_model: gpt-4\n  modules: []\n  workflow: []")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "docs", str(yaml_file),
            "--format", "html"
        ])
        
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
        assert result.exit_code == 0
        assert "HTML format not yet implemented" in result.output


class TestCLIMain:
    """Test main CLI functionality."""

    def test_cli_help(self):
        """Test CLI help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "DSPy-Databricks Agents CLI" in result.output
        assert "validate" in result.output
        assert "train" in result.output
        assert "deploy" in result.output
        assert "test" in result.output
        assert "monitor" in result.output

    def test_cli_version(self):
        """Test CLI version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        
        assert result.exit_code == 0
        # Version output format varies, just check it runs

    def test_command_help(self):
        """Test individual command help."""
        commands = ["validate", "train", "deploy", "test", "monitor", "docs"]
        runner = CliRunner()
        
        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0
            assert cmd.upper() in result.output or cmd in result.output