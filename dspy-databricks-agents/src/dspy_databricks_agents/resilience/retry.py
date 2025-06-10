"""Retry logic with exponential backoff."""

import time
import random
import logging
from typing import Callable, Optional, Tuple, Type, Union, Any
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomization to prevent thundering herd
    retry_on: Tuple[Type[Exception], ...] = (Exception,)  # Exceptions to retry on
    

class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    # Exponential backoff
    delay = min(
        config.initial_delay * (config.exponential_base ** (attempt - 1)),
        config.max_delay
    )
    
    # Add jitter if configured
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


def retry_with_backoff(
    func: Optional[Callable] = None,
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Union[Callable, Any]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        func: Function to retry (when used as @retry_with_backoff)
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        retry_on: Tuple of exceptions to retry on
        on_retry: Optional callback called on each retry with (exception, attempt)
    
    Returns:
        Decorated function or decorator
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on
    )
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return f(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Retry exhausted for {f.__name__} after {attempt} attempts. "
                            f"Last error: {e}"
                        )
                        raise RetryExhausted(e, attempt)
                    
                    delay = calculate_backoff_delay(attempt, config)
                    logger.warning(
                        f"Retry {attempt}/{config.max_attempts} for {f.__name__} "
                        f"after error: {e}. Waiting {delay:.2f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
                
        # Expose config for testing/monitoring
        wrapper._retry_config = config
        return wrapper
    
    # Support both @retry_with_backoff and @retry_with_backoff()
    if func is None:
        return decorator
    else:
        return decorator(func)


class RetryableOperation:
    """Context manager for retryable operations."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.attempts = 0
        self.last_exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
        
        if not issubclass(exc_type, self.config.retry_on):
            return False
        
        self.attempts += 1
        self.last_exception = exc_val
        
        if self.attempts >= self.config.max_attempts:
            logger.error(f"Retry exhausted after {self.attempts} attempts")
            return False
        
        delay = calculate_backoff_delay(self.attempts, self.config)
        logger.warning(
            f"Retry {self.attempts}/{self.config.max_attempts} "
            f"after error: {exc_val}. Waiting {delay:.2f}s..."
        )
        time.sleep(delay)
        return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.attempts = attempt
                return func(*args, **kwargs)
            except self.config.retry_on as e:
                self.last_exception = e
                
                if attempt == self.config.max_attempts:
                    raise RetryExhausted(e, attempt)
                
                delay = calculate_backoff_delay(attempt, self.config)
                logger.warning(
                    f"Retry {attempt}/{self.config.max_attempts} "
                    f"after error: {e}. Waiting {delay:.2f}s..."
                )
                time.sleep(delay)