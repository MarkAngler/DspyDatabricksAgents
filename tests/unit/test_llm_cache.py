"""Tests for LLM caching implementation."""

import pytest
import time
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from cache import (
    LLMCache,
    CacheConfig,
    CacheEntry,
    generate_cache_key
)
from cache.llm_cache import cached_llm_call


class TestCacheKeyGeneration:
    """Test cache key generation."""
    
    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        key1 = generate_cache_key(
            module_name="test_module",
            inputs={"query": "Hello world"}
        )
        
        key2 = generate_cache_key(
            module_name="test_module",
            inputs={"query": "Hello world"}
        )
        
        # Same inputs should generate same key
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length
    
    def test_generate_cache_key_different_inputs(self):
        """Test different inputs generate different keys."""
        key1 = generate_cache_key("module", {"query": "Hello"})
        key2 = generate_cache_key("module", {"query": "World"})
        
        assert key1 != key2
    
    def test_generate_cache_key_order_independent(self):
        """Test dict key order doesn't affect cache key."""
        key1 = generate_cache_key(
            "module",
            {"a": 1, "b": 2, "c": 3}
        )
        
        key2 = generate_cache_key(
            "module", 
            {"c": 3, "a": 1, "b": 2}
        )
        
        assert key1 == key2
    
    def test_generate_cache_key_with_optional_params(self):
        """Test cache key with optional parameters."""
        key1 = generate_cache_key(
            "module",
            {"query": "test"},
            model_name="gpt-4",
            temperature=0.7
        )
        
        key2 = generate_cache_key(
            "module",
            {"query": "test"},
            model_name="gpt-4",
            temperature=0.8
        )
        
        # Different temperature should generate different keys
        assert key1 != key2


class TestLLMCache:
    """Test LLM cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initializes with default config."""
        cache = LLMCache()
        
        assert cache.config.max_size_mb == 100
        assert cache.config.max_entries == 1000
        assert cache.config.ttl_seconds == 3600
        assert cache.config.eviction_policy == "lru"
        assert len(cache._cache) == 0
    
    def test_basic_get_put(self):
        """Test basic cache get and put operations."""
        cache = LLMCache()
        
        # Miss on empty cache
        assert cache.get("key1") is None
        
        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Stats reflect hit/miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
    
    def test_ttl_expiration(self):
        """Test entries expire after TTL."""
        config = CacheConfig(ttl_seconds=0.1)  # 100ms TTL
        cache = LLMCache(config)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None
        assert len(cache._cache) == 0
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        config = CacheConfig(max_entries=3, eviction_policy="lru")
        cache = LLMCache(config)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 and key3 to make key2 least recently used
        cache.get("key1")
        cache.get("key3")
        
        # Add new entry should evict key2
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_lfu_eviction(self):
        """Test LFU eviction policy."""
        config = CacheConfig(max_entries=3, eviction_policy="lfu")
        cache = LLMCache(config)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 and key3 multiple times
        for _ in range(3):
            cache.get("key1")
            cache.get("key3")
        cache.get("key2")  # Access key2 only once
        
        # Add new entry should evict key2 (least frequently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_size_based_eviction(self):
        """Test eviction based on size limit."""
        config = CacheConfig(max_size_mb=0.0001)  # 100 bytes limit
        cache = LLMCache(config)
        
        # Add large entries
        large_value = "x" * 50  # ~50 bytes when pickled
        cache.put("key1", large_value)
        cache.put("key2", large_value)
        cache.put("key3", large_value)
        
        # Should have evicted some entries to stay under limit
        remaining_entries = [k for k in ["key1", "key2", "key3"] if cache.get(k) is not None]
        assert len(remaining_entries) < 3  # At least one should be evicted
        assert cache._total_size_bytes <= 100  # Should respect size limit
    
    def test_invalidate(self):
        """Test manual cache invalidation."""
        cache = LLMCache()
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Invalidate specific key
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        # Invalidate non-existent key
        assert cache.invalidate("key3") is False
    
    def test_clear(self):
        """Test clearing entire cache."""
        cache = LLMCache()
        
        # Add multiple entries
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")
        
        stats = cache.get_stats()
        assert stats["entries"] == 5
        
        # Clear cache
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_thread_safety(self):
        """Test cache is thread-safe."""
        import threading
        
        cache = LLMCache()
        errors = []
        
        def cache_operations():
            try:
                for i in range(100):
                    cache.put(f"key{i}", f"value{i}")
                    cache.get(f"key{i}")
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=cache_operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0
    
    def test_persistent_cache(self):
        """Test persistent cache functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(persistent=True, cache_dir=temp_dir)
            
            # Create cache and add entries
            cache1 = LLMCache(config)
            cache1.put("key1", "value1")
            cache1.put("key2", {"data": "complex"})
            
            # Create new cache instance - should load from disk
            cache2 = LLMCache(config)
            assert cache2.get("key1") == "value1"
            assert cache2.get("key2") == {"data": "complex"}
            
            # Clear should remove files
            cache2.clear()
            assert len(os.listdir(temp_dir)) == 0
    
    def test_cached_llm_call_decorator(self):
        """Test cached LLM call decorator."""
        cache = LLMCache()
        call_count = 0
        
        def cache_key_func(query: str, **kwargs) -> str:
            return generate_cache_key("test", {"query": query})
        
        @cached_llm_call(cache, cache_key_func)
        def expensive_llm_call(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response to: {query}"
        
        # First call - cache miss
        result1 = expensive_llm_call("Hello")
        assert result1 == "Response to: Hello"
        assert call_count == 1
        
        # Second call - cache hit
        result2 = expensive_llm_call("Hello")
        assert result2 == "Response to: Hello"
        assert call_count == 1  # No additional call
        
        # Different query - cache miss
        result3 = expensive_llm_call("World")
        assert result3 == "Response to: World"
        assert call_count == 2
    
    def test_cache_entry_metadata(self):
        """Test cache entry metadata is tracked correctly."""
        cache = LLMCache()
        
        cache.put("key1", "value1")
        entry = cache._cache["key1"]
        
        # Check initial metadata
        assert entry.key == "key1"
        assert entry.value == "value1"
        assert entry.access_count == 0
        assert entry.size_bytes > 0
        
        # Access should update metadata
        initial_accessed = entry.accessed_at
        time.sleep(0.01)
        cache.get("key1")
        
        assert entry.access_count == 1
        assert entry.accessed_at > initial_accessed