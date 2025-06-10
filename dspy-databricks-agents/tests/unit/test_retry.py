"""Tests for retry logic implementation."""

import pytest
import time
from unittest.mock import Mock, patch, call
from dspy_databricks_agents.resilience import (
    RetryConfig,
    retry_with_backoff,
    RetryExhausted,
    RetryableOperation
)
from dspy_databricks_agents.resilience.retry import calculate_backoff_delay


class TestRetryLogic:
    """Test retry with exponential backoff."""
    
    def test_retry_config_defaults(self):
        """Test RetryConfig has sensible defaults."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retry_on == (Exception,)
    
    def test_calculate_backoff_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False
        )
        
        # Test exponential growth
        assert calculate_backoff_delay(1, config) == 1.0  # 1 * 2^0
        assert calculate_backoff_delay(2, config) == 2.0  # 1 * 2^1
        assert calculate_backoff_delay(3, config) == 4.0  # 1 * 2^2
        assert calculate_backoff_delay(4, config) == 8.0  # 1 * 2^3
        
        # Test max delay cap
        assert calculate_backoff_delay(10, config) == 60.0  # Would be 512, capped at 60
    
    def test_calculate_backoff_delay_with_jitter(self):
        """Test backoff calculation with jitter."""
        config = RetryConfig(
            initial_delay=10.0,
            jitter=True
        )
        
        # With jitter, delay should be between 50% and 100% of base delay
        delays = [calculate_backoff_delay(1, config) for _ in range(100)]
        
        assert all(5.0 <= d <= 10.0 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 50
    
    def test_successful_call_no_retry(self):
        """Test successful call doesn't retry."""
        mock_func = Mock(return_value="success")
        
        @retry_with_backoff(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_failure_then_success(self):
        """Test retry succeeds after initial failures."""
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_exhausted(self):
        """Test RetryExhausted raised after max attempts."""
        mock_func = Mock(side_effect=ValueError("persistent error"))
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(RetryExhausted) as exc_info:
            test_func()
        
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert mock_func.call_count == 3
    
    def test_retry_specific_exceptions(self):
        """Test retry only on specified exceptions."""
        mock_func = Mock(side_effect=[ValueError("retry this"), TypeError("don't retry")])
        
        @retry_with_backoff(retry_on=(ValueError,), initial_delay=0.01)
        def test_func():
            return mock_func()
        
        # Should retry ValueError but not TypeError
        with pytest.raises(TypeError):
            test_func()
        
        assert mock_func.call_count == 2
    
    def test_retry_with_callback(self):
        """Test retry with on_retry callback."""
        retry_calls = []
        
        def on_retry(exception, attempt):
            retry_calls.append((str(exception), attempt))
        
        mock_func = Mock(side_effect=[ValueError("error1"), ValueError("error2"), "success"])
        
        @retry_with_backoff(
            max_attempts=3,
            initial_delay=0.01,
            on_retry=on_retry
        )
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert retry_calls == [("error1", 1), ("error2", 2)]
    
    def test_retry_decorator_with_args(self):
        """Test retry decorator with function arguments."""
        mock_func = Mock(side_effect=[ValueError(), "result"])
        
        @retry_with_backoff(initial_delay=0.01)
        def test_func(a, b, c=None):
            return mock_func(a, b, c)
        
        result = test_func(1, 2, c=3)
        assert result == "result"
        assert mock_func.call_args_list == [call(1, 2, 3), call(1, 2, 3)]
    
    def test_retry_decorator_without_parentheses(self):
        """Test retry decorator can be used without parentheses."""
        mock_func = Mock(side_effect=[ValueError(), "success"])
        
        @retry_with_backoff
        def test_func():
            return mock_func()
        
        # Should use defaults
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_preserves_function_metadata(self):
        """Test retry decorator preserves function metadata."""
        @retry_with_backoff(max_attempts=5)
        def documented_function():
            """This is a documented function."""
            return "result"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."
        # Can access retry config
        assert documented_function._retry_config.max_attempts == 5
    
    @patch('time.sleep')
    def test_retry_timing(self, mock_sleep):
        """Test retry delays are applied correctly."""
        mock_func = Mock(side_effect=[ValueError(), ValueError(), "success"])
        
        @retry_with_backoff(
            max_attempts=3,
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        
        # Check sleep was called with correct delays
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry
        mock_sleep.assert_any_call(2.0)  # Second retry
    
    def test_retryable_operation_context_manager(self):
        """Test RetryableOperation context manager."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])
        
        with RetryableOperation(config) as retry_op:
            result = retry_op.execute(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 2
        assert retry_op.attempts == 2
    
    def test_retryable_operation_exhausted(self):
        """Test RetryableOperation raises when exhausted."""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        mock_func = Mock(side_effect=ValueError("persistent"))
        
        with pytest.raises(RetryExhausted) as exc_info:
            with RetryableOperation(config) as retry_op:
                retry_op.execute(mock_func)
        
        assert exc_info.value.attempts == 2
        assert mock_func.call_count == 2