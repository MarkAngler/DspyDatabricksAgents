"""Workflow orchestration for DSPy agents."""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from dspy_databricks_agents.config.schema import WorkflowStep


class WorkflowContext:
    """Execution context for workflow."""
    
    def __init__(self, initial_input: Dict[str, Any]):
        """Initialize workflow context."""
        self.data = {"input": initial_input}
        self.step_outputs = {}
        self.metadata = {}
    
    def resolve_reference(self, ref: str) -> Any:
        """Resolve $step.field references."""
        if not ref.startswith("$"):
            return ref
        
        # Parse reference: $step.field or $input.field
        parts = ref[1:].split(".", 1)
        
        if parts[0] == "input":
            return self._get_nested(self.data["input"], parts[1] if len(parts) > 1 else None)
        elif parts[0] in self.step_outputs:
            return self._get_nested(self.step_outputs[parts[0]], parts[1] if len(parts) > 1 else None)
        else:
            raise ValueError(f"Unknown reference: {ref}")
    
    def _get_nested(self, data: Any, path: Optional[str]) -> Any:
        """Get nested field from data."""
        if path is None:
            return data
        
        for key in path.split("."):
            if isinstance(data, dict):
                data = data.get(key)
            else:
                data = getattr(data, key, None)
            
            if data is None:
                break
        
        return data
    
    def set_output(self, step_name: str, output: Any):
        """Store step output."""
        self.step_outputs[step_name] = output
    
    def get_final_output(self) -> Dict[str, Any]:
        """Get final workflow output."""
        return {
            **self.step_outputs,
            "metadata": self.metadata
        }


class ConditionEvaluator:
    """Evaluate workflow conditions."""
    
    def __init__(self):
        """Initialize condition evaluator."""
        self.operators = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "in": lambda a, b: a in b,
            "not in": lambda a, b: a not in b,
        }
    
    def evaluate(self, condition: str, context: WorkflowContext) -> bool:
        """Evaluate condition string."""
        # Simple condition parser
        # Format: "$step.field operator value"
        # Handle >= and <= by adjusting the pattern
        pattern = r'(\$[\w.]+)\s*(==|!=|>=|<=|>|<|in|not in)\s*(.+)'
        match = re.match(pattern, condition.strip())
        
        if not match:
            raise ValueError(f"Invalid condition format: {condition}")
        
        ref, operator, value_str = match.groups()
        
        # Resolve reference
        left_value = context.resolve_reference(ref)
        
        # Parse right value
        right_value = self._parse_value(value_str.strip(), context)
        
        # Apply operator
        return self.operators[operator](left_value, right_value)
    
    def _parse_value(self, value_str: str, context: WorkflowContext) -> Any:
        """Parse value from string."""
        # Check if it's a reference
        if value_str.startswith("$"):
            return context.resolve_reference(value_str)
        
        # Try to parse as literal
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]
        elif value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.lower() == "true":
            return True
        elif value_str.lower() == "false":
            return False
        elif value_str.startswith("[") and value_str.endswith("]"):
            # Simple list parsing
            items = value_str[1:-1].split(",")
            return [self._parse_value(item.strip(), context) for item in items if item.strip()]
        elif value_str.isdigit():
            return int(value_str)
        else:
            try:
                return float(value_str)
            except ValueError:
                return value_str


class WorkflowOrchestrator:
    """Orchestrate workflow execution."""
    
    def __init__(self, modules: Dict[str, Any]):
        """Initialize orchestrator with modules."""
        self.modules = modules
        self.condition_evaluator = ConditionEvaluator()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def execute(
        self,
        workflow: List[WorkflowStep],
        initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow synchronously."""
        context = WorkflowContext(initial_input)
        
        for step in workflow:
            if not self._should_execute_step(step, context):
                continue
            
            try:
                output = self._execute_step(step, context)
                context.set_output(step.step, output)
            except Exception as e:
                if step.error_handler:
                    self._handle_error(step, e, context)
                else:
                    raise
        
        return context.get_final_output()
    
    async def execute_async(
        self,
        workflow: List[WorkflowStep],
        initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow asynchronously."""
        context = WorkflowContext(initial_input)
        
        # Group steps that can run in parallel
        step_groups = self._group_parallel_steps(workflow)
        
        for group in step_groups:
            if len(group) == 1:
                # Single step, execute normally
                step = group[0]
                if self._should_execute_step(step, context):
                    output = await self._execute_step_async(step, context)
                    context.set_output(step.step, output)
            else:
                # Multiple steps, execute in parallel
                tasks = []
                for step in group:
                    if self._should_execute_step(step, context):
                        task = self._execute_step_async(step, context)
                        tasks.append((step.step, task))
                
                # Wait for all tasks
                for step_name, task in tasks:
                    output = await task
                    context.set_output(step_name, output)
        
        return context.get_final_output()
    
    def _should_execute_step(self, step: WorkflowStep, context: WorkflowContext) -> bool:
        """Check if step should be executed based on condition."""
        if not step.condition:
            return True
        
        return self.condition_evaluator.evaluate(step.condition, context)
    
    def _execute_step(self, step: WorkflowStep, context: WorkflowContext) -> Any:
        """Execute a single workflow step."""
        # Get module
        if step.module not in self.modules:
            raise ValueError(f"Module '{step.module}' not found")
        
        module = self.modules[step.module]
        
        # Prepare inputs
        inputs = {}
        for key, value in step.inputs.items():
            inputs[key] = self._resolve_input_value(value, context)
        
        # Execute module
        return module(**inputs)
    
    def _resolve_input_value(self, value: Any, context: WorkflowContext) -> Any:
        """Resolve input value, handling references and nested structures."""
        if isinstance(value, str) and value.startswith("$"):
            return context.resolve_reference(value)
        elif isinstance(value, dict):
            # Recursively resolve dictionary values
            return {k: self._resolve_input_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively resolve list items
            return [self._resolve_input_value(item, context) for item in value]
        else:
            return value
    
    async def _execute_step_async(self, step: WorkflowStep, context: WorkflowContext) -> Any:
        """Execute step asynchronously."""
        # Get module
        if step.module not in self.modules:
            raise ValueError(f"Module '{step.module}' not found")
        
        module = self.modules[step.module]
        
        # Prepare inputs
        inputs = {}
        for key, value in step.inputs.items():
            inputs[key] = self._resolve_input_value(value, context)
        
        # Check if module is async
        if asyncio.iscoroutinefunction(module):
            return await module(**inputs)
        else:
            # Run sync module in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                module,
                **inputs
            )
    
    def _group_parallel_steps(self, workflow: List[WorkflowStep]) -> List[List[WorkflowStep]]:
        """Group steps that can run in parallel."""
        # Simple dependency analysis
        # Steps can run in parallel if they don't depend on each other
        groups = []
        current_group = []
        
        for step in workflow:
            # Check if step depends on any step in current group
            depends_on_current = False
            
            for prev_step in current_group:
                # Check if this step references the previous step
                if self._step_depends_on(step, prev_step):
                    depends_on_current = True
                    break
            
            if depends_on_current:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [step]
            else:
                # Add to current group
                current_group.append(step)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _step_depends_on(self, step: WorkflowStep, prev_step: WorkflowStep) -> bool:
        """Check if step depends on prev_step."""
        # Check in inputs
        for value in step.inputs.values():
            if self._contains_reference(value, prev_step.step):
                return True
        
        # Check in condition
        if step.condition and f"${prev_step.step}." in step.condition:
            return True
        
        return False
    
    def _contains_reference(self, value: Any, step_name: str) -> bool:
        """Check if value contains reference to step."""
        if isinstance(value, str) and f"${step_name}." in value:
            return True
        elif isinstance(value, dict):
            return any(self._contains_reference(v, step_name) for v in value.values())
        elif isinstance(value, list):
            return any(self._contains_reference(item, step_name) for item in value)
        return False
    
    def _handle_error(self, step: WorkflowStep, error: Exception, context: WorkflowContext):
        """Handle step execution error."""
        # TODO: Implement error handler logic
        raise error