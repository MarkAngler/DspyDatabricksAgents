"""Performance metrics collection for DSPy agents."""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import json


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass 
class MetricSummary:
    """Summary statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "avg": self.avg,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99
        }


class MetricsCollector:
    """Collects and aggregates performance metrics for agents."""
    
    def __init__(
        self, 
        max_history_size: int = 10000,
        aggregation_window_seconds: int = 60
    ):
        """Initialize metrics collector.
        
        Args:
            max_history_size: Maximum number of data points to keep per metric
            aggregation_window_seconds: Time window for metric aggregation
        """
        self.max_history_size = max_history_size
        self.aggregation_window_seconds = aggregation_window_seconds
        
        # Thread-safe storage for metrics
        self._lock = threading.Lock()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        
        # MLflow integration
        self._mlflow_enabled = False
        self._mlflow_run_id: Optional[str] = None
        
    def enable_mlflow_logging(self, run_id: Optional[str] = None):
        """Enable logging metrics to MLflow.
        
        Args:
            run_id: MLflow run ID (uses active run if not specified)
        """
        try:
            import mlflow
            self._mlflow_enabled = True
            self._mlflow_run_id = run_id or mlflow.active_run().info.run_id
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to enable MLflow logging: {str(e)}")
            self._mlflow_enabled = False
    
    def record_latency(
        self, 
        metric_name: str, 
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a latency measurement.
        
        Args:
            metric_name: Name of the metric
            latency_ms: Latency in milliseconds
            labels: Optional labels for the metric
        """
        self._record_value(f"{metric_name}_latency_ms", latency_ms, labels)
        
    def record_request(
        self,
        endpoint_name: str,
        latency_ms: float,
        status_code: int,
        model_version: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Record a model serving request.
        
        Args:
            endpoint_name: Name of the serving endpoint
            latency_ms: Request latency in milliseconds
            status_code: HTTP status code
            model_version: Model version that served the request
            error: Error message if request failed
        """
        labels = {
            "endpoint": endpoint_name,
            "status_code": str(status_code),
            "success": "true" if 200 <= status_code < 300 else "false"
        }
        
        if model_version:
            labels["model_version"] = model_version
        if error:
            labels["error"] = error[:100]  # Truncate long errors
            
        # Record latency
        self.record_latency("request", latency_ms, labels)
        
        # Increment counters
        self.increment_counter("request_total", labels=labels)
        if status_code >= 400:
            self.increment_counter("request_errors", labels=labels)
            
    def record_token_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        model: str
    ):
        """Record token usage for an agent.
        
        Args:
            agent_name: Name of the agent
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens used
            model: Model that processed the tokens
        """
        labels = {"agent": agent_name, "model": model}
        
        self._record_value("tokens_input", input_tokens, labels)
        self._record_value("tokens_output", output_tokens, labels)
        self._record_value("tokens_total", total_tokens, labels)
        
        # Update counters
        self.increment_counter("tokens_total_cumulative", total_tokens, labels)
        
    def record_workflow_execution(
        self,
        agent_name: str,
        workflow_name: str,
        duration_ms: float,
        steps_completed: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Record workflow execution metrics.
        
        Args:
            agent_name: Name of the agent
            workflow_name: Name of the workflow
            duration_ms: Total execution time in milliseconds
            steps_completed: Number of steps completed
            success: Whether workflow completed successfully
            error: Error message if workflow failed
        """
        labels = {
            "agent": agent_name,
            "workflow": workflow_name,
            "success": str(success).lower()
        }
        
        if error:
            labels["error_type"] = error.split(":")[0][:50]
            
        self._record_value("workflow_duration_ms", duration_ms, labels)
        self._record_value("workflow_steps", steps_completed, labels)
        self.increment_counter("workflow_executions", labels=labels)
        
    def record_cache_operation(
        self,
        cache_name: str,
        operation: str,  # "hit", "miss", "set", "evict"
        latency_ms: Optional[float] = None
    ):
        """Record cache operation metrics.
        
        Args:
            cache_name: Name of the cache
            operation: Type of operation
            latency_ms: Operation latency if applicable
        """
        labels = {"cache": cache_name, "operation": operation}
        
        self.increment_counter(f"cache_{operation}", labels=labels)
        if latency_ms is not None:
            self.record_latency("cache_operation", latency_ms, labels)
            
    def increment_counter(
        self, 
        counter_name: str, 
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric.
        
        Args:
            counter_name: Name of the counter
            value: Value to increment by
            labels: Optional labels
        """
        key = self._make_key(counter_name, labels)
        with self._lock:
            self._counters[key] += value
            
        # Log to MLflow if enabled
        if self._mlflow_enabled:
            self._log_to_mlflow(counter_name, self._counters[key], labels)
            
    def set_gauge(
        self, 
        gauge_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value.
        
        Args:
            gauge_name: Name of the gauge
            value: Current value
            labels: Optional labels
        """
        key = self._make_key(gauge_name, labels)
        with self._lock:
            self._gauges[key] = value
            
        # Log to MLflow if enabled
        if self._mlflow_enabled:
            self._log_to_mlflow(gauge_name, value, labels)
    
    def get_metric_summary(
        self, 
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        window_seconds: Optional[int] = None
    ) -> Optional[MetricSummary]:
        """Get summary statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            labels: Optional label filter
            window_seconds: Time window for aggregation (uses default if not specified)
            
        Returns:
            MetricSummary or None if no data
        """
        window_seconds = window_seconds or self.aggregation_window_seconds
        cutoff_time = datetime.now(timezone.utc).timestamp() - window_seconds
        
        # Get matching metrics
        matching_points = []
        with self._lock:
            for key, points in self._metrics.items():
                if metric_name in key:
                    if labels:
                        # Check if labels match
                        key_labels = self._parse_key_labels(key)
                        if all(key_labels.get(k) == v for k, v in labels.items()):
                            matching_points.extend([
                                p for p in points 
                                if p.timestamp.timestamp() > cutoff_time
                            ])
                    else:
                        matching_points.extend([
                            p for p in points 
                            if p.timestamp.timestamp() > cutoff_time
                        ])
        
        if not matching_points:
            return None
            
        # Calculate statistics
        values = [p.value for p in matching_points]
        values.sort()
        
        return MetricSummary(
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=sum(values) / len(values),
            p50=self._percentile(values, 50),
            p95=self._percentile(values, 95),
            p99=self._percentile(values, 99)
        )
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics.
        
        Returns:
            Dictionary with counters, gauges, and recent metric summaries
        """
        with self._lock:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "summaries": {}
            }
            
        # Get summaries for all metrics
        metric_names = set()
        with self._lock:
            for key in self._metrics.keys():
                metric_name = key.split("{")[0]
                metric_names.add(metric_name)
                
        for metric_name in metric_names:
            summary = self.get_metric_summary(metric_name)
            if summary:
                result["summaries"][metric_name] = summary.to_dict()
                
        return result
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        lines = []
        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Export counters
        with self._lock:
            for key, value in self._counters.items():
                metric_name, labels = self._parse_key(key)
                label_str = self._format_prometheus_labels(labels)
                lines.append(f"{metric_name}_total{label_str} {value} {timestamp}")
                
            # Export gauges
            for key, value in self._gauges.items():
                metric_name, labels = self._parse_key(key)
                label_str = self._format_prometheus_labels(labels)
                lines.append(f"{metric_name}{label_str} {value} {timestamp}")
        
        # Export metric summaries
        metric_names = set()
        with self._lock:
            for key in self._metrics.keys():
                metric_name = key.split("{")[0]
                metric_names.add(metric_name)
                
        for metric_name in metric_names:
            summary = self.get_metric_summary(metric_name)
            if summary:
                lines.append(f"{metric_name}_count {summary.count} {timestamp}")
                lines.append(f"{metric_name}_sum {summary.sum} {timestamp}")
                lines.append(f"{metric_name}_min {summary.min} {timestamp}")
                lines.append(f"{metric_name}_max {summary.max} {timestamp}")
                lines.append(f"{metric_name}_avg {summary.avg} {timestamp}")
                lines.append(f'{metric_name}{{quantile="0.5"}} {summary.p50} {timestamp}')
                lines.append(f'{metric_name}{{quantile="0.95"}} {summary.p95} {timestamp}')
                lines.append(f'{metric_name}{{quantile="0.99"}} {summary.p99} {timestamp}')
                
        return "\n".join(lines)
    
    def _record_value(
        self, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels
        """
        key = self._make_key(metric_name, labels)
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        
        with self._lock:
            self._metrics[key].append(point)
            
    def _make_key(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a metric key from name and labels.
        
        Args:
            metric_name: Name of the metric
            labels: Optional labels
            
        Returns:
            Metric key string
        """
        if not labels:
            return metric_name
            
        label_parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return f"{metric_name}{{{','.join(label_parts)}}}"
    
    def _parse_key(self, key: str) -> tuple[str, Dict[str, str]]:
        """Parse metric key into name and labels.
        
        Args:
            key: Metric key string
            
        Returns:
            Tuple of (metric_name, labels)
        """
        if "{" not in key:
            return key, {}
            
        metric_name = key.split("{")[0]
        labels = self._parse_key_labels(key)
        return metric_name, labels
    
    def _parse_key_labels(self, key: str) -> Dict[str, str]:
        """Parse labels from metric key.
        
        Args:
            key: Metric key string
            
        Returns:
            Dictionary of labels
        """
        if "{" not in key:
            return {}
            
        label_str = key.split("{")[1].rstrip("}")
        labels = {}
        
        for part in label_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k] = v.strip('"')
                
        return labels
    
    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus export.
        
        Args:
            labels: Label dictionary
            
        Returns:
            Formatted label string
        """
        if not labels:
            return ""
            
        label_parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return f"{{{','.join(label_parts)}}}"
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of sorted values.
        
        Args:
            values: Sorted list of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
            
        index = int(len(values) * percentile / 100)
        if index >= len(values):
            return values[-1]
        return values[index]
    
    def _log_to_mlflow(
        self, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Log metric to MLflow.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels
        """
        if not self._mlflow_enabled:
            return
            
        try:
            import mlflow
            
            # Format metric name with labels
            if labels:
                label_str = "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))
                full_metric_name = f"{metric_name}_{label_str}"
            else:
                full_metric_name = metric_name
                
            # Log to MLflow
            if self._mlflow_run_id:
                with mlflow.start_run(run_id=self._mlflow_run_id):
                    mlflow.log_metric(full_metric_name, value)
            else:
                mlflow.log_metric(full_metric_name, value)
                
        except Exception:
            # Silently ignore MLflow logging errors
            pass