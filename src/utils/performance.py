"""Performance optimization utilities for Microsoft standards."""

import asyncio
import time
import functools
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from src.config.settings import settings
from src.utils.logger import logger

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    operation: str
    duration: float
    memory_usage: float
    cache_hit: bool
    timestamp: datetime

class PerformanceOptimizer:
    """Performance optimization manager."""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url) if hasattr(settings, 'redis_url') else None
        self.metrics: List[PerformanceMetrics] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
    def cache_with_ttl(self, ttl_seconds: int = 3600):
        """Decorator for caching function results with TTL."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                if self.redis_client:
                    try:
                        cached_result = self.redis_client.get(cache_key)
                        if cached_result:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return json.loads(cached_result)
                    except Exception as e:
                        logger.warning(f"Cache read error: {e}")
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Store in cache
                if self.redis_client:
                    try:
                        self.redis_client.setex(
                            cache_key, 
                            ttl_seconds, 
                            json.dumps(result, default=str)
                        )
                    except Exception as e:
                        logger.warning(f"Cache write error: {e}")
                
                # Record metrics
                self._record_metrics(func.__name__, duration, cache_hit=False)
                
                return result
            return wrapper
        return decorator
    
    def async_execution(self, func: Callable) -> Callable:
        """Decorator for async execution of CPU-intensive tasks."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
        return async_wrapper
    
    def rate_limit(self, max_calls: int = 100, time_window: int = 3600):
        """Decorator for rate limiting function calls."""
        def decorator(func: Callable) -> Callable:
            call_counts = {}
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                now = time.time()
                key = f"{func.__name__}:{now // time_window}"
                
                # Clean old entries
                for k in list(call_counts.keys()):
                    if now - float(k.split(':')[1]) * time_window > time_window:
                        del call_counts[k]
                
                # Check rate limit
                if call_counts.get(key, 0) >= max_calls:
                    raise Exception(f"Rate limit exceeded for {func.__name__}")
                
                call_counts[key] = call_counts.get(key, 0) + 1
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _record_metrics(self, operation: str, duration: float, cache_hit: bool):
        """Record performance metrics."""
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        metric = PerformanceMetrics(
            operation=operation,
            duration=duration,
            memory_usage=memory_usage,
            cache_hit=cache_hit,
            timestamp=datetime.utcnow()
        )
        
        self.metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}
        
        recent_metrics = [m for m in self.metrics if m.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        if not recent_metrics:
            return {}
        
        avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
        
        return {
            'total_operations': len(recent_metrics),
            'average_duration': avg_duration,
            'average_memory_mb': avg_memory,
            'cache_hit_rate': cache_hit_rate,
            'operations_per_minute': len(recent_metrics) / 60
        }

class ResponseOptimizer:
    """Optimize API responses for faster delivery."""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
    
    def compress_response(self, data: Any) -> bytes:
        """Compress response data."""
        import gzip
        json_data = json.dumps(data, default=str)
        return gzip.compress(json_data.encode('utf-8'))
    
    def paginate_results(self, data: List[Any], page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """Paginate large result sets."""
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_data = data[start_idx:end_idx]
        
        return {
            'data': paginated_data,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_items': len(data),
                'total_pages': (len(data) + page_size - 1) // page_size,
                'has_next': end_idx < len(data),
                'has_prev': page > 1
            }
        }
    
    def optimize_query_response(self, query: str, results: List[Any]) -> Dict[str, Any]:
        """Optimize query response with smart caching and formatting."""
        # Use caching for similar queries
        @self.optimizer.cache_with_ttl(ttl_seconds=1800)  # 30 minutes
        def process_query_results(query: str, results: List[Any]) -> Dict[str, Any]:
            return {
                'query': query,
                'results': results,
                'count': len(results),
                'timestamp': datetime.utcnow().isoformat(),
                'optimized': True
            }
        
        return process_query_results(query, results)

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
response_optimizer = ResponseOptimizer()