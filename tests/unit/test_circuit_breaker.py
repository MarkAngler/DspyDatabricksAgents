"""Tests for circuit breaker implementation."""

import pytest
import time
from unittest.mock import Mock, patch
from dspy_databricks_agents.resilience import (
    CircuitBreaker, 
    CircuitBreakerError, 
    CircuitState,
    circuit_breaker
)
from dspy_databricks_agents.resilience.circuit_breaker import CircuitBreakerConfig


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes in closed state."""
        cb = CircuitBreaker("test")
        
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0
        assert cb._last_failure_time is None
    
    def test_successful_calls_in_closed_state(self):
        """Test successful calls don't open the circuit."""
        cb = CircuitBreaker("test")
        mock_func = Mock(return_value="success")
        
        # Multiple successful calls
        for _ in range(10):
            result = cb.call(mock_func)
            assert result == "success"
        
        assert cb.state == CircuitState.CLOSED
        assert mock_func.call_count == 10
    
    def test_circuit_opens_after_threshold_failures(self):
        """Test circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        mock_func = Mock(side_effect=ValueError("error"))
        
        # First two failures don't open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(mock_func)
            assert cb.state == CircuitState.CLOSED
        
        # Third failure opens circuit
        with pytest.raises(ValueError):
            cb.call(mock_func)
        assert cb.state == CircuitState.OPEN
        
        # Subsequent calls are rejected
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.call(mock_func)
        assert "is OPEN" in str(exc_info.value)
        assert mock_func.call_count == 3  # No additional calls
    
    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1  # 100ms for testing
        )
        cb = CircuitBreaker("test", config)
        mock_func = Mock(side_effect=ValueError("error"))
        
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(mock_func)
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Check state transitions to half-open
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_half_open_to_closed_on_success(self):
        """Test circuit closes from half-open after successful calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=2
        )
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(Mock(side_effect=ValueError()))
        
        # Wait for half-open
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        
        # First success doesn't close
        cb.call(Mock(return_value="success"))
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second success closes circuit
        cb.call(Mock(return_value="success"))
        assert cb.state == CircuitState.CLOSED
    
    def test_half_open_to_open_on_failure(self):
        """Test circuit reopens from half-open on failure."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1
        )
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(Mock(side_effect=ValueError()))
        
        # Wait for half-open
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Failure in half-open reopens circuit
        with pytest.raises(ValueError):
            cb.call(Mock(side_effect=ValueError()))
        assert cb.state == CircuitState.OPEN
    
    def test_only_configured_exceptions_trigger_breaker(self):
        """Test only configured exception types trigger the breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            error_types=(ValueError,)
        )
        cb = CircuitBreaker("test", config)
        
        # TypeError doesn't trigger breaker
        with pytest.raises(TypeError):
            cb.call(Mock(side_effect=TypeError()))
        assert cb.state == CircuitState.CLOSED
        
        # ValueError does trigger breaker
        with pytest.raises(ValueError):
            cb.call(Mock(side_effect=ValueError()))
        assert cb.state == CircuitState.OPEN
    
    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(Mock(side_effect=ValueError()))
        assert cb.state == CircuitState.OPEN
        
        # Manual reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0
        assert cb._last_failure_time is None
    
    def test_get_status(self):
        """Test getting circuit breaker status."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            success_threshold=2
        )
        cb = CircuitBreaker("test", config)
        
        status = cb.get_status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["last_failure"] is None
        assert status["config"]["failure_threshold"] == 3
        assert status["config"]["recovery_timeout"] == 60
        assert status["config"]["success_threshold"] == 2
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as a decorator."""
        call_count = 0
        
        @circuit_breaker("test", failure_threshold=2)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Flaky error")
            return "success"
        
        # First two calls fail and open circuit
        with pytest.raises(ValueError):
            flaky_function()
        with pytest.raises(ValueError):
            flaky_function()
        
        # Circuit is now open
        with pytest.raises(CircuitBreakerError):
            flaky_function()
        
        # Access the circuit breaker instance
        assert hasattr(flaky_function, "_circuit_breaker")
        assert flaky_function._circuit_breaker.state == CircuitState.OPEN
    
    def test_thread_safety(self):
        """Test circuit breaker is thread-safe."""
        import threading
        
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)
        errors = []
        
        def make_failing_calls():
            try:
                for _ in range(10):
                    try:
                        cb.call(Mock(side_effect=ValueError()))
                    except (ValueError, CircuitBreakerError):
                        pass
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=make_failing_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No unexpected errors
        assert len(errors) == 0
        # Circuit should be open
        assert cb.state == CircuitState.OPEN