# Contributing to DSPy-Databricks Agents

Thank you for your interest in contributing to DSPy-Databricks Agents! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the Repository**: Click the "Fork" button on GitHub to create your own copy
2. **Clone Your Fork**: 
   ```bash
   git clone https://github.com/your-username/dspy-databricks-agents.git
   cd dspy-databricks-agents
   ```
3. **Add Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/original-org/dspy-databricks-agents.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- Git
- Databricks workspace (for integration testing)

### Installation

1. **Install Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Activate Virtual Environment**:
   ```bash
   poetry shell
   ```

4. **Install Pre-commit Hooks**:
   ```bash
   poetry run pre-commit install
   ```

### Environment Configuration

Create a `.env` file for local development:

```bash
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-token

# Test Configuration
TEST_CATALOG=test_catalog
TEST_SCHEMA=test_schema
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix identified issues
2. **Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve tests
5. **Performance**: Optimize existing code
6. **Examples**: Add new example configurations

### Finding Issues

- Check the [Issues](https://github.com/your-org/dspy-databricks-agents/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on an issue to indicate you're working on it

### Creating Issues

When creating an issue, please:
- Use a clear and descriptive title
- Provide detailed description
- Include steps to reproduce (for bugs)
- Add relevant labels
- Include system information if relevant

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, documented code
- Follow the coding standards
- Add or update tests
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_specific.py

# Run with coverage
poetry run pytest --cov=dspy_databricks_agents
```

### 4. Format and Lint

```bash
# Format code
poetry run black src/ tests/

# Run linter
poetry run ruff src/ tests/

# Type checking
poetry run mypy src/
```

### 5. Commit Your Changes

Follow conventional commit messages:

```bash
# Format: <type>(<scope>): <subject>

feat(agent): add streaming support for ChatAgent
fix(parser): handle empty YAML files correctly
docs(readme): update installation instructions
test(workflow): add tests for conditional execution
refactor(modules): simplify module factory pattern
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Include screenshots if UI changes
- List any breaking changes

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 88 characters (Black default)
- Use type hints for all functions
- Use docstrings for all public functions

### Code Organization

```python
# Import order
import standard_library_imports
import third_party_imports

from dspy_databricks_agents import local_imports

# Class definitions
class ExampleClass:
    """Class documentation."""
    
    def __init__(self, param: str) -> None:
        """Initialize with parameter.
        
        Args:
            param: Description of parameter
        """
        self.param = param
    
    def public_method(self) -> str:
        """Public method documentation.
        
        Returns:
            Description of return value
        """
        return self._private_method()
    
    def _private_method(self) -> str:
        """Private methods start with underscore."""
        return self.param
```

### Type Hints

Always use type hints:

```python
from typing import Dict, List, Optional, Union

def process_data(
    data: List[Dict[str, Any]], 
    config: Optional[Dict[str, str]] = None
) -> Union[str, None]:
    """Process data with optional configuration."""
    pass
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestModuleName:
    """Test suite for module_name."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data."""
        return {"key": "value"}
    
    def test_functionality(self, sample_data):
        """Test specific functionality."""
        # Arrange
        expected = "expected_result"
        
        # Act
        result = function_under_test(sample_data)
        
        # Assert
        assert result == expected
    
    @patch('module.external_dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependencies."""
        mock_dep.return_value = "mocked"
        result = function_using_dep()
        assert result == "mocked"
```

### Test Categories

1. **Unit Tests** (`tests/unit/`): Test individual components
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **E2E Tests** (`tests/e2e/`): Test complete workflows

### Test Coverage

- Aim for 100% coverage for new code
- Minimum 80% coverage for modified code
- Use `# pragma: no cover` sparingly

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> Dict[str, Any]:
    """Short description of function.
    
    Longer description if needed. Can span multiple lines
    and include examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

### Documentation Updates

When adding features, update:
1. README.md (if major feature)
2. Inline documentation
3. Example configurations
4. API documentation

## Release Process

### Version Numbering

We use Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Deploy to PyPI

## Questions?

- Open a [Discussion](https://github.com/your-org/dspy-databricks-agents/discussions)
- Join our Slack channel
- Email: dspy-databricks@your-org.com

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to DSPy-Databricks Agents! 🎉