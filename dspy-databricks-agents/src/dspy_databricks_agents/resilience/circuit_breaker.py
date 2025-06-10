"""Circuit breaker implementation for fault tolerance."""

import time
from enum import Enum
from typing import Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    error_types: tuple = (Exception,)  # Errors that trigger the breaker


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation."""
    
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        logger.info(f"Circuit breaker '{self.name}' initialized with config: {self.config}")
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        return (
            self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.config.recovery_timeout
        )
    
    def _transition_to(self, state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = state
        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to {state.value}")
        
        # Reset counters on state change
        if state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif state == CircuitState.HALF_OPEN:
            self._success_count = 0
    
    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable, will retry after {self.config.recovery_timeout}s"
                )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.error_types as e:
                self._on_failure()
                raise
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(f"Circuit breaker '{self.name}' success in HALF_OPEN: {self._success_count}/{self.config.success_threshold}")
                
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                logger.warning(f"Circuit breaker '{self.name}' failure: {self._failure_count}/{self.config.failure_threshold}")
                
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_status(self) -> dict:
        """Get current status of circuit breaker."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": datetime.fromtimestamp(self._last_failure_time).isoformat() if self._last_failure_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                }
            }


def circuit_breaker(name: str, **config_kwargs) -> Callable:
    """Decorator to apply circuit breaker to a function."""
    breaker_config = CircuitBreakerConfig(**config_kwargs)
    breaker = CircuitBreaker(name, breaker_config)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        wrapper._circuit_breaker = breaker  # Expose breaker for testing/monitoring
        return wrapper
    return decorator