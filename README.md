# DSPy-Databricks Agents

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](tests/)

A Python package for creating and deploying DSPy-powered agentic workflows via YAML configuration on Databricks.

## 🌟 Overview

DSPy-Databricks Agents enables you to:
- 🚀 Define complex agentic workflows using simple YAML configurations
- 🔧 Leverage DSPy's automatic prompt optimization capabilities
- ☁️ Deploy seamlessly to Databricks Model Serving
- 📊 Track and monitor with MLflow integration
- 🔍 Use Databricks Vector Search for RAG applications
- 🎯 Build production-ready conversational agents

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Examples](#-examples)
- [CLI Usage](#-cli-usage)
- [Configuration](#-configuration)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Capabilities
- **Zero-Code Agent Definition**: Define agents entirely through YAML
- **DSPy Integration**: Automatic prompt optimization and few-shot learning
- **MLflow ChatAgent**: Production-ready conversational agent interface
- **Workflow Orchestration**: Multi-step workflows with conditional logic
- **Vector Search**: Native integration with Databricks Vector Search
- **Multi-Agent Support**: Compose complex systems from multiple agents

### Module Types
- **Signature**: Basic input/output transformations
- **Chain of Thought**: Step-by-step reasoning
- **ReAct**: Tool-using agents with action/observation loops
- **Retriever**: Vector search and RAG capabilities
- **Custom**: Bring your own DSPy modules

### Built-in Tools for ReAct Agents
- **calculator**: Performs mathematical calculations
- **web_search**: Searches the web (mock - integrate with real API)
- **database_query**: Queries databases (mock - integrate with real DB)
- **python_repl**: Executes Python code (mock - needs sandboxing)
- **file_reader**: Reads file contents
- **json_parser**: Parses and formats JSON data

### Production Features
- **Deployment Automation**: One-command deployment to Databricks
- **Monitoring**: Built-in metrics and observability
- **Rate Limiting**: Control API usage and costs
- **A/B Testing**: Compare agent versions (coming soon)
- **Auto-scaling**: Handle varying workloads efficiently

## 📦 Installation

### From PyPI (Coming Soon)
```bash
pip install dspy-databricks-agents
```

### From Source
```bash
git clone https://github.com/your-org/dspy-databricks-agents.git
cd dspy-databricks-agents
pip install -e .
```

### Using Poetry
```bash
poetry add dspy-databricks-agents
```

## 🚀 Quick Start

### 1. Create Your First Agent

Create a file `my_agent.yaml`:

```yaml
agent:
  name: simple-qa-agent
  version: 1.0.0
  description: A simple question-answering agent
  
  dspy:
    lm_model: databricks-dbrx-instruct
    temperature: 0.7
    
  modules:
    - name: qa_module
      type: signature
      signature: "question -> answer"
      
  workflow:
    - step: answer_question
      module: qa_module
      inputs:
        question: "$input.query"
```

### 2. Validate Your Configuration

```bash
dspy-databricks validate my_agent.yaml
```

### 3. Test Locally

```bash
dspy-databricks test my-agent \
  --local my_agent.yaml \
  --query "What is machine learning?"
```

### 4. Deploy to Databricks

```bash
dspy-databricks deploy my_agent.yaml --environment production
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        YAML Configuration                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │    Agent    │  │   Modules    │  │      Workflow          │ │
│  │  Metadata   │  │ (DSPy Types) │  │  (Orchestration)       │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DSPy-Databricks Core                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Parser    │  │    Module    │  │    Workflow            │ │
│  │   & Valid.  │  │   Factory    │  │   Orchestrator         │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MLflow ChatAgent                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Predict   │  │   Stream     │  │    Monitoring          │ │
│  │   Method    │  │   Support    │  │    & Tracing           │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Databricks Platform                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Model     │  │   Unity      │  │    Vector              │ │
│  │  Serving    │  │   Catalog    │  │    Search              │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Configuration as Code**: Everything defined in YAML
2. **Modular Architecture**: Compose complex agents from simple modules
3. **Production-First**: Built for enterprise deployment
4. **Extensible**: Easy to add custom modules and optimizers
5. **Observable**: Comprehensive logging and monitoring

## 📚 Examples

### Customer Support Agent
```yaml
agent:
  name: customer-support-agent
  modules:
    - name: intent_classifier
      type: signature
      signature: "query -> intent, urgency"
    - name: knowledge_retriever
      type: retriever
      vector_store:
        index: support_knowledge_base
    - name: response_generator
      type: chain_of_thought
      signature: "query, context, intent -> response"
```

[See more examples →](examples/)

## 🛠️ CLI Usage

### Available Commands

```bash
# Validate configuration
dspy-databricks validate <yaml_file>

# Train/optimize agent
dspy-databricks train <yaml_file> --dataset <data.json>

# Deploy to Databricks
dspy-databricks deploy <yaml_file> --environment <env>

# Test agent
dspy-databricks test <agent_name> --query "<question>"

# Monitor performance
dspy-databricks monitor <agent_name> --last-24h

# Generate documentation
dspy-databricks docs <yaml_file> --output <docs.md>
```

### CLI Options

| Command | Description | Key Options |
|---------|-------------|-------------|
| `validate` | Check YAML syntax and schema | `--verbose` |
| `train` | Optimize prompts with DSPy | `--metric`, `--num-candidates` |
| `deploy` | Deploy to Databricks | `--environment`, `--endpoint` |
| `test` | Test agent locally or remotely | `--local`, `--stream` |
| `monitor` | View metrics and performance | `--format`, `--metric` |
| `docs` | Generate documentation | `--format`, `--output` |

## ⚙️ Configuration

### Environment Variables

```bash
# Databricks Configuration
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"

# Model Configuration
export LLM_MODEL="databricks-dbrx-instruct"
export EMBEDDING_MODEL="databricks-bge-large-en"

# Deployment Settings
export DEPLOYMENT_CATALOG="ml"
export DEPLOYMENT_SCHEMA="agents"
```

### YAML Configuration Reference

```yaml
agent:
  name: string                    # Required: Agent identifier
  version: string                 # Required: Semantic version
  description: string             # Optional: Agent description
  
  dspy:
    lm_model: string             # Required: Language model
    temperature: float           # Optional: 0.0-2.0 (default: 0.7)
    max_tokens: integer          # Optional: Max response tokens
    optimizer:                   # Optional: DSPy optimizer config
      type: string               # Optimizer type
      metric: string             # Optimization metric
      num_candidates: integer    # Number of candidates
  
  modules:                       # Required: List of modules
    - name: string               # Module identifier
      type: string               # Module type
      signature: string          # DSPy signature (if applicable)
      vector_store:              # Vector store config (retrievers)
        catalog: string
        schema: string
        index: string
        k: integer
      
  workflow:                      # Required: Execution flow
    - step: string               # Step identifier
      module: string             # Module to execute
      condition: string          # Optional: Conditional execution
      inputs: dict               # Optional: Input mappings
      
  deployment:                    # Optional: Deployment config
    catalog: string
    schema: string
    model_name: string
    serving_endpoint: string
    compute_size: string         # Small, Medium, Large
```

## 🧪 Testing

### Run Unit Tests
```bash
poetry run pytest tests/unit/ -v
```

### Run Integration Tests
```bash
poetry run pytest tests/integration/ -v
```

### Run All Tests with Coverage
```bash
poetry run pytest --cov=dspy_databricks_agents tests/
```

### Test Statistics
- **Total Tests**: 108
- **Unit Tests**: 95
- **Integration Tests**: 13
- **Coverage**: 100%

## 🚢 Deployment

### Prerequisites
1. Databricks workspace with Model Serving enabled
2. Unity Catalog configured
3. Appropriate permissions for model registration
4. Databricks SDK installed (included with package)

### Deployment Steps

1. **Configure Environment**
   ```bash
   export DATABRICKS_HOST="https://your-workspace.databricks.com"
   export DATABRICKS_TOKEN="your-token"
   ```

2. **Validate Configuration**
   ```bash
   dspy-databricks validate production_agent.yaml
   ```

3. **Deploy Agent**
   ```bash
   dspy-databricks deploy production_agent.yaml \
     --environment production \
     --compute-size Medium
   ```
   
   The deployment process will:
   - Register your agent as an MLflow model in Unity Catalog
   - Create or update a Model Serving endpoint
   - Configure rate limiting (if specified)
   - Return the endpoint URL for testing

4. **Verify Deployment**
   ```bash
   # Test with the returned endpoint URL
   dspy-databricks test my-agent \
     --endpoint https://your-workspace.databricks.com/serving-endpoints/production-my-agent/invocations \
     --query "Your test question"
   ```

### Deployment Configuration

Add a deployment section to your YAML:

```yaml
deployment:
  catalog: ml              # Unity Catalog name
  schema: agents          # Schema for models
  model_name: my_agent    # Model registration name
  serving_endpoint: prod-my-agent  # Endpoint name
  compute_size: Medium    # Small, Medium, or Large
  
  # Optional: Environment variables for the endpoint
  environment_vars:
    LOG_LEVEL: INFO
    CACHE_ENABLED: "true"
  
  # Optional: Rate limiting
  rate_limits:
    requests_per_minute: 100
    requests_per_user: 10
```

### Production Considerations

- **Rate Limiting**: Configure appropriate limits for your use case
- **Monitoring**: Set up alerts for latency and error rates
- **Scaling**: Choose compute size based on expected load
- **Security**: Use Databricks secrets for sensitive configuration
- **Versioning**: Use semantic versioning for your agents

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-org/dspy-databricks-agents.git
cd dspy-databricks-agents

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black src/ tests/
poetry run ruff src/ tests/
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://docs.your-org.com/dspy-databricks-agents)
- [PyPI Package](https://pypi.org/project/dspy-databricks-agents)
- [Issue Tracker](https://github.com/your-org/dspy-databricks-agents/issues)
- [Discussions](https://github.com/your-org/dspy-databricks-agents/discussions)

## 🙏 Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) - The declarative language model programming framework
- [Databricks](https://databricks.com) - Unified analytics platform
- [MLflow](https://mlflow.org) - Open source platform for ML lifecycle

---

Built with ❤️ by the DSPy-Databricks Team