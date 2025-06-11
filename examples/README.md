# DSPy-Databricks Agents Examples

This directory contains example YAML configurations demonstrating various use cases and features of the DSPy-Databricks Agents framework.

## 📚 Example Agents

### 1. Simple Q&A Agent (`simple_qa_agent.yaml`)
- **Purpose**: Basic question-answering with minimal configuration
- **Key Features**: Single DSPy signature module
- **Use Case**: Quick demos, simple chatbots
- **Complexity**: ⭐

### 2. Customer Support Agent (`customer_support_agent.yaml`)
- **Purpose**: Multi-stage customer query handling
- **Key Features**: 
  - Intent classification
  - Conditional knowledge retrieval
  - Response validation
- **Use Case**: Customer service automation
- **Complexity**: ⭐⭐⭐

### 3. Data Analysis Agent (`data_analysis_agent.yaml`)
- **Purpose**: Natural language to SQL with insights
- **Key Features**:
  - SQL generation from questions
  - Query validation
  - Visualization recommendations
- **Use Case**: Business intelligence, data exploration
- **Complexity**: ⭐⭐⭐⭐

### 4. Code Review Agent (`code_review_agent.yaml`)
- **Purpose**: Automated code review and suggestions
- **Key Features**:
  - ReAct pattern with tools
  - Multi-language support
  - Security scanning
- **Use Case**: Development workflow automation
- **Complexity**: ⭐⭐⭐⭐

### 5. RAG Research Agent (`rag_research_agent.yaml`)
- **Purpose**: Multi-source research with citations
- **Key Features**:
  - Multiple retrieval sources
  - Fact checking
  - Academic paper integration
- **Use Case**: Research assistance, content creation
- **Complexity**: ⭐⭐⭐⭐⭐

### 6. Multi-Agent Workflow (`multi_agent_workflow.yaml`)
- **Purpose**: Complex content creation pipeline
- **Key Features**:
  - Agent composition
  - Multi-phase workflow
  - Imports and reusability
- **Use Case**: Content generation, complex workflows
- **Complexity**: ⭐⭐⭐⭐⭐

### 7. Advanced Configuration (`advanced_config_example.yaml`)
- **Purpose**: Demonstrate all configuration features
- **Key Features**:
  - Environment variables
  - Conditional logic
  - Advanced deployment options
- **Use Case**: Production deployments, configuration reference
- **Complexity**: ⭐⭐⭐⭐⭐

## 🚀 Quick Start

### 1. Validate an Agent Configuration

```bash
dspy-databricks validate examples/simple_qa_agent.yaml
```

### 2. Test Locally

```bash
dspy-databricks test simple-qa \
  --local examples/simple_qa_agent.yaml \
  --query "What is machine learning?"
```

### 3. Deploy to Databricks

```bash
dspy-databricks deploy examples/customer_support_agent.yaml \
  --environment production
```

## 🔧 Configuration Features

### Environment Variables
Use `${VAR_NAME}` or `${VAR_NAME:-default_value}` syntax:

```yaml
dspy:
  lm_model: "${LLM_MODEL:-databricks-dbrx-instruct}"
  temperature: ${TEMPERATURE:-0.7}
```

### Imports and Reusability
Import shared configurations:

```yaml
imports:
  - path: ./shared/common_modules.yaml
    prefix: common
```

### Conditional Workflow Steps
Execute steps based on conditions:

```yaml
workflow:
  - step: retrieve_context
    module: retriever
    condition: "$classify.intent != 'general'"
```

### Vector Store Configuration
Configure Databricks Vector Search:

```yaml
vector_store:
  catalog: ml
  schema: embeddings
  index: knowledge_base
  k: 10
```

## 📊 Module Types

1. **Signature**: Basic input/output transformations
   ```yaml
   type: signature
   signature: "question -> answer"
   ```

2. **Chain of Thought**: Step-by-step reasoning
   ```yaml
   type: chain_of_thought
   signature: "problem -> reasoning, solution"
   ```

3. **ReAct**: Tool-using agents
   ```yaml
   type: react
   tools: ["calculator", "web_search"]
   ```

4. **Retriever**: Vector search integration
   ```yaml
   type: retriever
   vector_store:
     index: knowledge_base
   ```

5. **Custom**: User-defined modules
   ```yaml
   type: custom
   custom_class: "my_package.CustomModule"
   ```

## 🏗️ Deployment Options

### Compute Sizes
- `Small`: 1-2 concurrent requests
- `Medium`: 3-10 concurrent requests  
- `Large`: 10+ concurrent requests

### Environment Variables
Pass runtime configuration:

```yaml
deployment:
  environment_vars:
    LOG_LEVEL: "${LOG_LEVEL:-INFO}"
    ENABLE_CACHE: "true"
```

### Rate Limiting
Control API usage:

```yaml
rate_limits:
  requests_per_minute: 100
  requests_per_user: 10
```

## 📝 Best Practices

1. **Start Simple**: Begin with `simple_qa_agent.yaml` and add complexity
2. **Use Environment Variables**: Keep secrets and environment-specific config external
3. **Modular Design**: Break complex agents into reusable modules
4. **Test Locally**: Always validate and test before deployment
5. **Monitor Performance**: Use the monitoring commands to track agent behavior
6. **Version Control**: Use semantic versioning for your agents

## 🔗 Resources

- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/)
- [MLflow ChatAgent](https://mlflow.org/docs/latest/llms/chat-agent.html)

## 💡 Tips

- Use lower temperatures (0.1-0.5) for factual/analytical agents
- Use higher temperatures (0.7-0.9) for creative agents
- Start with fewer optimization candidates and increase based on quality
- Use conditional steps to optimize token usage
- Leverage caching for frequently accessed data

## 🐛 Troubleshooting

If validation fails:
1. Check YAML syntax
2. Ensure all module references exist
3. Verify environment variables are set
4. Confirm module types are valid

For deployment issues:
1. Verify Databricks credentials
2. Check catalog/schema permissions
3. Ensure endpoint names are unique
4. Monitor compute resource limits