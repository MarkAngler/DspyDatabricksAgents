"""LLM response caching implementation."""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for LLM cache."""
    max_size_mb: int = 100  # Maximum cache size in MB
    max_entries: int = 1000  # Maximum number of entries
    ttl_seconds: int = 3600  # Time to live in seconds (1 hour default)
    eviction_policy: str = "lru"  # lru, lfu, fifo
    persistent: bool = False  # Whether to persist cache to disk
    cache_dir: str = ".cache/llm"  # Directory for persistent cache


class LLMCache:
    """Thread-safe LLM response cache with multiple eviction policies."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._total_size_bytes = 0
        self._hits = 0
        self._misses = 0
        
        # Load persistent cache if configured
        if self.config.persistent:
            self._load_persistent_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Update access metadata
            entry.touch()
            
            # Move to end for LRU
            if self.config.eviction_policy == "lru":
                self._cache.move_to_end(key)
            
            self._hits += 1
            logger.debug(f"Cache hit for key: {key[:16]}...")
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache with automatic eviction if needed."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if single item exceeds max size
            max_size_bytes = self.config.max_size_mb * 1024 * 1024
            if size_bytes > max_size_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return
            
            # Evict if necessary
            while (
                (self._total_size_bytes + size_bytes > max_size_bytes or
                 len(self._cache) >= self.config.max_entries) and
                len(self._cache) > 0
            ):
                self._evict_one()
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._total_size_bytes += size_bytes
            
            logger.debug(f"Cached response for key: {key[:16]}... ({size_bytes} bytes)")
            
            # Persist if configured
            if self.config.persistent:
                self._persist_entry(key, entry)
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
            self._hits = 0
            self._misses = 0
            
            if self.config.persistent:
                self._clear_persistent_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "entries": len(self._cache),
                "size_mb": self._total_size_bytes / (1024 * 1024),
                "max_size_mb": self.config.max_size_mb,
                "eviction_policy": self.config.eviction_policy,
                "ttl_seconds": self.config.ttl_seconds,
            }
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        age = time.time() - entry.created_at
        return age > self.config.ttl_seconds
    
    def _evict_one(self) -> None:
        """Evict one entry based on eviction policy."""
        if not self._cache:
            return
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used (first item)
            key = next(iter(self._cache))
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.config.eviction_policy == "fifo":
            # Remove oldest (first item)
            key = next(iter(self._cache))
        else:
            # Default to LRU
            key = next(iter(self._cache))
        
        self._remove_entry(key)
        logger.debug(f"Evicted cache entry: {key[:16]}...")
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update size."""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size_bytes -= entry.size_bytes
            del self._cache[key]
            
            if self.config.persistent:
                self._remove_persistent_entry(key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback for non-picklable objects
            return len(str(obj).encode())
    
    # Persistent cache methods
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache entry."""
        os.makedirs(self.config.cache_dir, exist_ok=True)
        return os.path.join(self.config.cache_dir, f"{key}.pkl")
    
    def _persist_entry(self, key: str, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        try:
            path = self._get_cache_path(key)
            with open(path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Load cache from disk on startup."""
        if not os.path.exists(self.config.cache_dir):
            return
        
        try:
            for filename in os.listdir(self.config.cache_dir):
                if filename.endswith('.pkl'):
                    key = filename[:-4]
                    path = os.path.join(self.config.cache_dir, filename)
                    
                    with open(path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    # Skip expired entries
                    if not self._is_expired(entry):
                        self._cache[key] = entry
                        self._total_size_bytes += entry.size_bytes
                    else:
                        os.remove(path)
            
            logger.info(f"Loaded {len(self._cache)} entries from persistent cache")
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    def _remove_persistent_entry(self, key: str) -> None:
        """Remove persistent cache entry."""
        try:
            path = self._get_cache_path(key)
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to remove persistent cache entry: {e}")
    
    def _clear_persistent_cache(self) -> None:
        """Clear all persistent cache files."""
        try:
            if os.path.exists(self.config.cache_dir):
                for filename in os.listdir(self.config.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.config.cache_dir, filename))
        except Exception as e:
            logger.warning(f"Failed to clear persistent cache: {e}")


def cached_llm_call(
    cache: LLMCache,
    cache_key_func: Callable[..., str]
) -> Callable:
    """
    Decorator to cache LLM function calls.
    
    Args:
        cache: LLMCache instance
        cache_key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_key_func(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        wrapper._cache = cache  # Expose cache for testing
        return wrapper
    return decorator