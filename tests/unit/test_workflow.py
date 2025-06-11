"""Unit tests for workflow orchestration."""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from config.schema import WorkflowStep
from core.workflow import (
    ConditionEvaluator,
    WorkflowContext,
    WorkflowOrchestrator,
)


class TestWorkflowContext:
    """Test workflow context functionality."""

    def test_workflow_context_initialization(self):
        """Test workflow context initialization."""
        initial_input = {"query": "test query", "temperature": 0.7}
        context = WorkflowContext(initial_input)
        
        assert context.data["input"] == initial_input
        assert context.step_outputs == {}
        assert context.metadata == {}

    def test_resolve_input_reference(self):
        """Test resolving input references."""
        context = WorkflowContext({"query": "test", "count": 5})
        
        # Test input references
        assert context.resolve_reference("$input.query") == "test"
        assert context.resolve_reference("$input.count") == 5

    def test_resolve_step_reference(self):
        """Test resolving step output references."""
        context = WorkflowContext({"query": "test"})
        context.set_output("step1", {"result": "success", "score": 0.9})
        
        # Test step references
        assert context.resolve_reference("$step1.result") == "success"
        assert context.resolve_reference("$step1.score") == 0.9

    def test_resolve_nested_reference(self):
        """Test resolving nested references."""
        context = WorkflowContext({"data": {"nested": {"value": 42}}})
        context.set_output("process", {"output": {"deep": {"result": "found"}}})
        
        # Test nested references
        assert context.resolve_reference("$input.data.nested.value") == 42
        assert context.resolve_reference("$process.output.deep.result") == "found"

    def test_resolve_non_reference(self):
        """Test that non-references are returned as-is."""
        context = WorkflowContext({})
        
        assert context.resolve_reference("plain string") == "plain string"
        assert context.resolve_reference("123") == "123"

    def test_resolve_invalid_reference(self):
        """Test that invalid references raise errors."""
        context = WorkflowContext({"query": "test"})
        
        with pytest.raises(ValueError, match="Unknown reference"):
            context.resolve_reference("$nonexistent.field")

    def test_set_and_get_output(self):
        """Test setting and getting step outputs."""
        context = WorkflowContext({})
        
        # Set outputs
        context.set_output("step1", {"data": [1, 2, 3]})
        context.set_output("step2", "simple output")
        
        # Verify outputs
        assert context.step_outputs["step1"] == {"data": [1, 2, 3]}
        assert context.step_outputs["step2"] == "simple output"

    def test_get_final_output(self):
        """Test getting final workflow output."""
        context = WorkflowContext({"input": "test"})
        context.set_output("analyze", {"sentiment": "positive"})
        context.set_output("generate", {"response": "Great!"})
        context.metadata["duration"] = 1.5
        
        final = context.get_final_output()
        
        assert final["analyze"] == {"sentiment": "positive"}
        assert final["generate"] == {"response": "Great!"}
        assert final["metadata"]["duration"] == 1.5


class TestConditionEvaluator:
    """Test condition evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Create condition evaluator."""
        return ConditionEvaluator()

    def test_equality_conditions(self, evaluator):
        """Test equality operators."""
        context = WorkflowContext({})
        context.set_output("check", {"status": "success", "count": 5})
        
        # Test ==
        assert evaluator.evaluate("$check.status == 'success'", context) is True
        assert evaluator.evaluate("$check.count == 5", context) is True
        assert evaluator.evaluate("$check.status == 'failure'", context) is False

    def test_inequality_conditions(self, evaluator):
        """Test inequality operators."""
        context = WorkflowContext({})
        context.set_output("check", {"status": "success", "count": 5})
        
        # Test !=
        assert evaluator.evaluate("$check.status != 'failure'", context) is True
        assert evaluator.evaluate("$check.count != 10", context) is True
        assert evaluator.evaluate("$check.status != 'success'", context) is False

    def test_comparison_conditions(self, evaluator):
        """Test comparison operators."""
        context = WorkflowContext({})
        context.set_output("metrics", {"score": 0.85, "threshold": 0.7})
        
        # Test > < >= <=
        assert evaluator.evaluate("$metrics.score > 0.8", context) is True
        assert evaluator.evaluate("$metrics.score < 0.9", context) is True
        assert evaluator.evaluate("$metrics.score >= 0.85", context) is True
        assert evaluator.evaluate("$metrics.score <= 0.85", context) is True
        assert evaluator.evaluate("$metrics.score > $metrics.threshold", context) is True

    def test_membership_conditions(self, evaluator):
        """Test membership operators."""
        context = WorkflowContext({})
        context.set_output("data", {"tags": ["urgent", "customer"], "status": "active"})
        
        # Test in/not in
        assert evaluator.evaluate("$data.status in ['active', 'pending']", context) is True
        assert evaluator.evaluate("$data.status not in ['completed', 'failed']", context) is True

    def test_boolean_conditions(self, evaluator):
        """Test boolean value conditions."""
        context = WorkflowContext({})
        context.set_output("flags", {"enabled": True, "debug": False})
        
        assert evaluator.evaluate("$flags.enabled == true", context) is True
        assert evaluator.evaluate("$flags.debug == false", context) is True

    def test_invalid_condition_format(self, evaluator):
        """Test that invalid conditions raise errors."""
        context = WorkflowContext({})
        
        with pytest.raises(ValueError, match="Invalid condition format"):
            evaluator.evaluate("invalid condition", context)


class TestWorkflowOrchestrator:
    """Test workflow orchestration."""

    @pytest.fixture
    def mock_modules(self):
        """Create mock DSPy modules."""
        return {
            "classifier": Mock(return_value={"category": "technical", "confidence": 0.9}),
            "retriever": Mock(return_value={"documents": ["doc1", "doc2"]}),
            "generator": Mock(return_value={"response": "Generated response"}),
        }

    @pytest.fixture
    def orchestrator(self, mock_modules):
        """Create workflow orchestrator."""
        return WorkflowOrchestrator(mock_modules)

    def test_simple_sequential_workflow(self, orchestrator):
        """Test simple sequential workflow execution."""
        workflow = [
            WorkflowStep(step="classify", module="classifier", inputs={"text": "$input.query"}),
            WorkflowStep(step="generate", module="generator", inputs={"category": "$classify.category"}),
        ]
        
        result = orchestrator.execute(workflow, {"query": "How to fix error?"})
        
        # Verify execution
        assert "classify" in result
        assert result["classify"]["category"] == "technical"
        assert "generate" in result
        assert result["generate"]["response"] == "Generated response"

    def test_conditional_workflow(self, orchestrator, mock_modules):
        """Test workflow with conditional steps."""
        workflow = [
            WorkflowStep(step="classify", module="classifier", inputs={"text": "$input.query"}),
            WorkflowStep(
                step="retrieve",
                module="retriever",
                condition="$classify.confidence > 0.8",
                inputs={"query": "$input.query"},
            ),
            WorkflowStep(step="generate", module="generator"),
        ]
        
        # High confidence - retriever should run
        result = orchestrator.execute(workflow, {"query": "test"})
        assert "retrieve" in result
        mock_modules["retriever"].assert_called_once()
        
        # Low confidence - retriever should be skipped
        mock_modules["classifier"].return_value = {"category": "general", "confidence": 0.6}
        mock_modules["retriever"].reset_mock()
        
        result = orchestrator.execute(workflow, {"query": "test"})
        assert "retrieve" not in result
        mock_modules["retriever"].assert_not_called()

    def test_workflow_with_complex_inputs(self, orchestrator):
        """Test workflow with complex input mapping."""
        workflow = [
            WorkflowStep(
                step="process",
                module="generator",
                inputs={
                    "text": "$input.query",
                    "options": {"temperature": "$input.temp", "max_length": 100},
                    "literal": "fixed_value",
                },
            ),
        ]
        
        orchestrator.execute(workflow, {"query": "test", "temp": 0.8})
        
        # Verify inputs passed to module
        call_args = orchestrator.modules["generator"].call_args[1]
        assert call_args["text"] == "test"
        assert call_args["options"]["temperature"] == 0.8
        assert call_args["options"]["max_length"] == 100
        assert call_args["literal"] == "fixed_value"

    def test_workflow_error_handling(self, orchestrator, mock_modules):
        """Test workflow error handling."""
        # Make classifier raise an error
        mock_modules["classifier"].side_effect = RuntimeError("Classification failed")
        
        workflow = [
            WorkflowStep(step="classify", module="classifier"),
            WorkflowStep(step="generate", module="generator"),
        ]
        
        # Without error handler, should raise
        with pytest.raises(RuntimeError, match="Classification failed"):
            orchestrator.execute(workflow, {"query": "test"})

    def test_workflow_with_error_handler(self, orchestrator, mock_modules):
        """Test workflow with error handler."""
        # Add fallback module
        mock_modules["fallback"] = Mock(return_value={"response": "Fallback response"})
        orchestrator.modules["fallback"] = mock_modules["fallback"]
        
        # Make generator raise an error
        mock_modules["generator"].side_effect = RuntimeError("Generation failed")
        
        workflow = [
            WorkflowStep(step="classify", module="classifier"),
            WorkflowStep(step="generate", module="generator", error_handler="fallback"),
        ]
        
        # Error handler not implemented yet, should still raise
        with pytest.raises(RuntimeError):
            orchestrator.execute(workflow, {"query": "test"})

    def test_missing_module_error(self, orchestrator):
        """Test error when module is not found."""
        workflow = [
            WorkflowStep(step="unknown", module="nonexistent"),
        ]
        
        with pytest.raises(ValueError, match="Module 'nonexistent' not found"):
            orchestrator.execute(workflow, {"query": "test"})

    @pytest.mark.asyncio
    async def test_async_workflow_execution(self, mock_modules):
        """Test asynchronous workflow execution."""
        # Create async mocks
        async_modules = {
            "async1": AsyncMock(return_value={"result": 1}),
            "async2": AsyncMock(return_value={"result": 2}),
            "async3": AsyncMock(return_value={"result": 3}),
        }
        
        orchestrator = WorkflowOrchestrator(async_modules)
        
        # Create workflow with independent steps
        workflow = [
            WorkflowStep(step="step1", module="async1"),
            WorkflowStep(step="step2", module="async2"),
            WorkflowStep(step="step3", module="async3", inputs={"data": "$step1.result"}),
        ]
        
        result = await orchestrator.execute_async(workflow, {"input": "test"})
        
        assert result["step1"]["result"] == 1
        assert result["step2"]["result"] == 2
        assert result["step3"]["result"] == 3

    def test_parallel_step_grouping(self, orchestrator):
        """Test grouping of parallel steps."""
        workflow = [
            WorkflowStep(step="a", module="classifier"),
            WorkflowStep(step="b", module="retriever"),  # Independent
            WorkflowStep(step="c", module="generator", inputs={"data": "$a.category"}),  # Depends on a
            WorkflowStep(step="d", module="classifier"),  # Independent
        ]
        
        groups = orchestrator._group_parallel_steps(workflow)
        
        # Should create groups based on dependencies
        # Note: Current implementation is conservative - once a dependency is found,
        # subsequent steps go in new groups even if independent
        assert len(groups) >= 2
        assert len(groups[0]) == 2  # a and b can run in parallel
        assert any(step.step == "c" for step in groups[1])  # c should be in second group