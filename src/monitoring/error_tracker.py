"""Error tracking and alerting for DSPy agents."""

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import hashlib
import warnings
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Single error event."""
    timestamp: datetime
    severity: ErrorSeverity
    error_type: str
    message: str
    agent_name: Optional[str] = None
    endpoint_name: Optional[str] = None
    workflow_name: Optional[str] = None
    module_name: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate error hash if not provided."""
        if not self.error_hash:
            # Create hash from error type and key parts of message
            hash_content = f"{self.error_type}:{self.message[:100]}"
            self.error_hash = hashlib.md5(hash_content.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "error_type": self.error_type,
            "message": self.message,
            "error_hash": self.error_hash
        }
        
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.endpoint_name:
            result["endpoint_name"] = self.endpoint_name
        if self.workflow_name:
            result["workflow_name"] = self.workflow_name
        if self.module_name:
            result["module_name"] = self.module_name
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace
        if self.context:
            result["context"] = self.context
            
        return result


@dataclass
class ErrorPattern:
    """Pattern for error detection and alerting."""
    name: str
    error_types: Set[str]
    threshold_count: int
    time_window_seconds: int
    severity_filter: Optional[Set[ErrorSeverity]] = None
    alert_callback: Optional[Callable[[List[ErrorEvent]], None]] = None
    
    def matches(self, error: ErrorEvent) -> bool:
        """Check if error matches this pattern.
        
        Args:
            error: Error event to check
            
        Returns:
            True if error matches pattern
        """
        # Check error type
        if self.error_types and error.error_type not in self.error_types:
            if not any(error.error_type.startswith(t) for t in self.error_types):
                return False
        
        # Check severity
        if self.severity_filter and error.severity not in self.severity_filter:
            return False
            
        return True


class ErrorTracker:
    """Tracks errors and triggers alerts based on patterns."""
    
    def __init__(
        self,
        max_history_size: int = 10000,
        default_alert_threshold: int = 10,
        default_time_window_seconds: int = 300
    ):
        """Initialize error tracker.
        
        Args:
            max_history_size: Maximum number of errors to keep in history
            default_alert_threshold: Default error count threshold for alerts
            default_time_window_seconds: Default time window for error patterns
        """
        self.max_history_size = max_history_size
        self.default_alert_threshold = default_alert_threshold
        self.default_time_window_seconds = default_time_window_seconds
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self._errors = deque(maxlen=max_history_size)
        self._error_counts = defaultdict(int)
        self._patterns: Dict[str, ErrorPattern] = {}
        self._alert_history: Dict[str, datetime] = {}
        
        # Alert callbacks
        self._global_alert_callbacks: List[Callable] = []
        
        # Databricks integration
        self._databricks_client: Optional[Any] = None
        self._alert_webhook_url: Optional[str] = None
        
    def set_databricks_client(self, client: Any):
        """Set Databricks client for sending alerts.
        
        Args:
            client: Databricks workspace client
        """
        self._databricks_client = client
        
    def set_alert_webhook(self, webhook_url: str):
        """Set webhook URL for sending alerts.
        
        Args:
            webhook_url: Webhook URL (e.g., Slack, Teams)
        """
        self._alert_webhook_url = webhook_url
        
    def register_alert_callback(self, callback: Callable[[List[ErrorEvent]], None]):
        """Register a global alert callback.
        
        Args:
            callback: Function to call when alerts are triggered
        """
        self._global_alert_callbacks.append(callback)
        
    def register_pattern(self, pattern: ErrorPattern):
        """Register an error pattern for alerting.
        
        Args:
            pattern: Error pattern to monitor
        """
        self._patterns[pattern.name] = pattern
        
    def track_error(
        self,
        error_type: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        agent_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        workflow_name: Optional[str] = None,
        module_name: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """Track an error event.
        
        Args:
            error_type: Type/category of error
            message: Error message
            severity: Error severity level
            agent_name: Name of the agent
            endpoint_name: Name of the endpoint
            workflow_name: Name of the workflow
            module_name: Name of the module
            exception: Optional exception object
            context: Additional context
            
        Returns:
            The tracked error event
        """
        # Extract stack trace if exception provided
        stack_trace = None
        if exception:
            stack_trace = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            stack_trace = "".join(stack_trace)
            
        # Create error event
        error = ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            error_type=error_type,
            message=message,
            agent_name=agent_name,
            endpoint_name=endpoint_name,
            workflow_name=workflow_name,
            module_name=module_name,
            stack_trace=stack_trace,
            context=context or {}
        )
        
        # Store error
        with self._lock:
            self._errors.append(error)
            self._error_counts[error.error_hash] += 1
            
        # Check patterns and trigger alerts
        self._check_patterns(error)
        
        # Log to MLflow if available
        self._log_to_mlflow(error)
        
        return error
        
    def track_agent_error(
        self,
        agent_name: str,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """Track an agent-specific error.
        
        Args:
            agent_name: Name of the agent
            error_type: Type of error
            message: Error message
            exception: Optional exception
            context: Additional context
            
        Returns:
            The tracked error event
        """
        return self.track_error(
            error_type=f"agent.{error_type}",
            message=message,
            severity=ErrorSeverity.ERROR,
            agent_name=agent_name,
            exception=exception,
            context=context
        )
        
    def track_deployment_error(
        self,
        endpoint_name: str,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEvent:
        """Track a deployment-specific error.
        
        Args:
            endpoint_name: Name of the endpoint
            error_type: Type of error
            message: Error message
            exception: Optional exception
            context: Additional context
            
        Returns:
            The tracked error event
        """
        return self.track_error(
            error_type=f"deployment.{error_type}",
            message=message,
            severity=ErrorSeverity.ERROR,
            endpoint_name=endpoint_name,
            exception=exception,
            context=context
        )
        
    def get_recent_errors(
        self,
        minutes: int = 60,
        severity_filter: Optional[Set[ErrorSeverity]] = None,
        error_type_filter: Optional[str] = None,
        agent_filter: Optional[str] = None,
        endpoint_filter: Optional[str] = None
    ) -> List[ErrorEvent]:
        """Get recent errors with optional filters.
        
        Args:
            minutes: Number of minutes to look back
            severity_filter: Filter by severity levels
            error_type_filter: Filter by error type prefix
            agent_filter: Filter by agent name
            endpoint_filter: Filter by endpoint name
            
        Returns:
            List of matching error events
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        with self._lock:
            errors = list(self._errors)
            
        # Apply filters
        filtered = []
        for error in errors:
            if error.timestamp < cutoff_time:
                continue
                
            if severity_filter and error.severity not in severity_filter:
                continue
                
            if error_type_filter and not error.error_type.startswith(error_type_filter):
                continue
                
            if agent_filter and error.agent_name != agent_filter:
                continue
                
            if endpoint_filter and error.endpoint_name != endpoint_filter:
                continue
                
            filtered.append(error)
            
        return filtered
        
    def get_error_summary(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get error summary statistics.
        
        Args:
            time_window_minutes: Time window for summary
            
        Returns:
            Dictionary with error statistics
        """
        recent_errors = self.get_recent_errors(minutes=time_window_minutes)
        
        # Group by various dimensions
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        by_agent = defaultdict(int)
        by_endpoint = defaultdict(int)
        unique_errors = set()
        
        for error in recent_errors:
            by_severity[error.severity.value] += 1
            by_type[error.error_type] += 1
            if error.agent_name:
                by_agent[error.agent_name] += 1
            if error.endpoint_name:
                by_endpoint[error.endpoint_name] += 1
            unique_errors.add(error.error_hash)
            
        return {
            "time_window_minutes": time_window_minutes,
            "total_errors": len(recent_errors),
            "unique_errors": len(unique_errors),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "by_agent": dict(by_agent),
            "by_endpoint": dict(by_endpoint),
            "top_errors": self._get_top_errors(recent_errors, limit=10)
        }
        
    def _get_top_errors(self, errors: List[ErrorEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top occurring errors.
        
        Args:
            errors: List of errors
            limit: Maximum number of top errors
            
        Returns:
            List of top error summaries
        """
        error_groups = defaultdict(list)
        for error in errors:
            error_groups[error.error_hash].append(error)
            
        # Sort by count
        sorted_groups = sorted(
            error_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:limit]
        
        result = []
        for error_hash, group in sorted_groups:
            sample_error = group[0]
            result.append({
                "error_hash": error_hash,
                "count": len(group),
                "error_type": sample_error.error_type,
                "message": sample_error.message,
                "first_seen": min(e.timestamp for e in group).isoformat(),
                "last_seen": max(e.timestamp for e in group).isoformat(),
                "severity": sample_error.severity.value
            })
            
        return result
        
    def _check_patterns(self, error: ErrorEvent):
        """Check if error triggers any alert patterns.
        
        Args:
            error: Error event to check
        """
        for pattern in self._patterns.values():
            if pattern.matches(error):
                self._check_pattern_threshold(pattern)
                
    def _check_pattern_threshold(self, pattern: ErrorPattern):
        """Check if pattern threshold is exceeded.
        
        Args:
            pattern: Pattern to check
        """
        # Get matching errors in time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=pattern.time_window_seconds)
        
        matching_errors = []
        with self._lock:
            for error in self._errors:
                if error.timestamp < cutoff_time:
                    continue
                if pattern.matches(error):
                    matching_errors.append(error)
                    
        # Check threshold
        if len(matching_errors) >= pattern.threshold_count:
            self._trigger_alert(pattern, matching_errors)
            
    def _trigger_alert(self, pattern: ErrorPattern, errors: List[ErrorEvent]):
        """Trigger alert for pattern.
        
        Args:
            pattern: Pattern that triggered
            errors: Matching errors
        """
        # Check if we already alerted recently
        alert_key = f"pattern:{pattern.name}"
        with self._lock:
            last_alert = self._alert_history.get(alert_key)
            if last_alert:
                # Don't alert more than once per time window
                if datetime.now(timezone.utc) - last_alert < timedelta(seconds=pattern.time_window_seconds):
                    return
                    
            self._alert_history[alert_key] = datetime.now(timezone.utc)
            
        # Call pattern-specific callback
        if pattern.alert_callback:
            try:
                pattern.alert_callback(errors)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {str(e)}")
                
        # Call global callbacks
        for callback in self._global_alert_callbacks:
            try:
                callback(errors)
            except Exception as e:
                warnings.warn(f"Global alert callback failed: {str(e)}")
                
        # Send webhook alert
        if self._alert_webhook_url:
            self._send_webhook_alert(pattern, errors)
            
        # Log to Databricks if available
        if self._databricks_client:
            self._log_alert_to_databricks(pattern, errors)
            
    def _send_webhook_alert(self, pattern: ErrorPattern, errors: List[ErrorEvent]):
        """Send alert via webhook.
        
        Args:
            pattern: Pattern that triggered
            errors: Matching errors
        """
        try:
            import requests
            
            # Prepare alert payload
            sample_error = errors[0]
            payload = {
                "text": f"Error Alert: {pattern.name}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Alert:* {pattern.name}\n"
                                   f"*Count:* {len(errors)} errors in {pattern.time_window_seconds}s\n"
                                   f"*Type:* {sample_error.error_type}\n"
                                   f"*Message:* {sample_error.message[:200]}"
                        }
                    }
                ]
            }
            
            # Send webhook
            response = requests.post(
                self._alert_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            warnings.warn(f"Failed to send webhook alert: {str(e)}")
            
    def _log_alert_to_databricks(self, pattern: ErrorPattern, errors: List[ErrorEvent]):
        """Log alert to Databricks.
        
        Args:
            pattern: Pattern that triggered
            errors: Matching errors
        """
        try:
            # Could log to a Delta table or send notification
            # This is a placeholder for Databricks-specific alerting
            pass
        except Exception as e:
            warnings.warn(f"Failed to log alert to Databricks: {str(e)}")
            
    def _log_to_mlflow(self, error: ErrorEvent):
        """Log error to MLflow.
        
        Args:
            error: Error event to log
        """
        try:
            import mlflow
            
            if mlflow.active_run():
                # Log error as tag
                mlflow.set_tag(
                    f"error.{error.error_hash}",
                    json.dumps({
                        "type": error.error_type,
                        "message": error.message[:200],
                        "severity": error.severity.value,
                        "timestamp": error.timestamp.isoformat()
                    })
                )
                
                # Log error count as metric
                mlflow.log_metric(
                    f"error_count_{error.severity.value}",
                    self._error_counts[error.error_hash]
                )
                
        except Exception:
            # Silently ignore MLflow logging errors
            pass
            
    def create_default_patterns(self) -> List[ErrorPattern]:
        """Create default error patterns for common scenarios.
        
        Returns:
            List of default error patterns
        """
        patterns = [
            # High error rate pattern
            ErrorPattern(
                name="high_error_rate",
                error_types=set(),  # Match all types
                threshold_count=self.default_alert_threshold,
                time_window_seconds=self.default_time_window_seconds,
                severity_filter={ErrorSeverity.ERROR, ErrorSeverity.CRITICAL}
            ),
            
            # Critical errors pattern
            ErrorPattern(
                name="critical_errors",
                error_types=set(),
                threshold_count=1,  # Alert on any critical error
                time_window_seconds=60,
                severity_filter={ErrorSeverity.CRITICAL}
            ),
            
            # Deployment failures
            ErrorPattern(
                name="deployment_failures",
                error_types={"deployment."},
                threshold_count=3,
                time_window_seconds=300,
                severity_filter={ErrorSeverity.ERROR, ErrorSeverity.CRITICAL}
            ),
            
            # Agent failures
            ErrorPattern(
                name="agent_failures", 
                error_types={"agent."},
                threshold_count=5,
                time_window_seconds=300,
                severity_filter={ErrorSeverity.ERROR, ErrorSeverity.CRITICAL}
            ),
            
            # Model serving errors
            ErrorPattern(
                name="model_serving_errors",
                error_types={"serving.", "endpoint."},
                threshold_count=10,
                time_window_seconds=300
            )
        ]
        
        # Register all patterns
        for pattern in patterns:
            self.register_pattern(pattern)
            
        return patterns