"""Redis-based cache manager for enterprise features."""

import json
import pickle
import redis
from typing import Any, Optional, Union
from datetime import timedelta
import os
from src.utils.logger import logger


class CacheManager:
    """Redis-based cache manager with fallback to memory cache."""
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        """Initialize cache manager."""
        self.default_ttl = default_ttl
        self.redis_client = None
        self.memory_cache = {}  # Fallback cache
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using memory cache.")
                self.redis_client = None
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return json.dumps(data).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            return pickle.loads(data)
        except Exception:
            try:
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Deserialization error: {e}")
                return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return self._deserialize(data)
            else:
                # Fallback to memory cache
                if key in self.memory_cache:
                    return self.memory_cache[key]
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            serialized_data = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            if self.redis_client:
                return self.redis_client.setex(key, ttl, serialized_data)
            else:
                # Fallback to memory cache
                self.memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                # Fallback to memory cache
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
        
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return key in self.memory_cache
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self.redis_client:
                return self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
                return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'type': 'redis',
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            else:
                return {
                    'type': 'memory',
                    'total_keys': len(self.memory_cache),
                    'memory_usage': f"{len(str(self.memory_cache))} bytes"
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {'type': 'error', 'error': str(e)}
    
    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment a numeric value in cache."""
        try:
            if self.redis_client:
                result = self.redis_client.incrby(key, amount)
                if ttl:
                    self.redis_client.expire(key, ttl)
                return result
            else:
                # Fallback to memory cache
                current = self.memory_cache.get(key, 0)
                new_value = current + amount
                self.memory_cache[key] = new_value
                return new_value
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return 0
    
    def set_hash(self, key: str, mapping: dict, ttl: Optional[int] = None) -> bool:
        """Set hash in cache."""
        try:
            if self.redis_client:
                result = self.redis_client.hset(key, mapping=mapping)
                if ttl:
                    self.redis_client.expire(key, ttl)
                return bool(result)
            else:
                # Fallback to memory cache
                self.memory_cache[key] = mapping
                return True
        except Exception as e:
            logger.error(f"Cache set_hash error: {e}")
            return False
    
    def get_hash(self, key: str) -> Optional[dict]:
        """Get hash from cache."""
        try:
            if self.redis_client:
                return self.redis_client.hgetall(key)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get_hash error: {e}")
            return None


# Global cache instance
cache_manager = CacheManager(
    redis_url=os.getenv('REDIS_URL'),
    default_ttl=int(os.getenv('CACHE_TTL', '3600'))
)