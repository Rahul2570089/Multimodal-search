"""
Performance optimization and caching service
"""
import os
import logging
import time
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    hit_count: int = 0
    size_bytes: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    execution_time_ms: float
    cache_hit: bool
    result_size: int
    timestamp: datetime

class PerformanceOptimizer:
    """Performance optimization and caching service"""
    
    def __init__(self):
        # Cache configuration
        self.cache_enabled = True
        self.cache_max_size = 1000  # Maximum number of entries
        self.cache_default_ttl = 3600  # 1 hour default TTL
        
        # In-memory cache
        self._cache = {}
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self.metrics = []
        self.operation_stats = defaultdict(list)
        
        # Optimization settings
        self.optimization_thresholds = {
            'slow_operation_ms': 100,
            'large_result_size': 1000,
            'low_cache_hit_rate': 0.3
        }
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._cleanup_running = False
        self._start_cleanup_thread()
    
    def cache_result(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Cache a result with optional TTL"""
        try:
            if not self.cache_enabled:
                return False
            
            with self._cache_lock:
                # Check cache size limit
                if len(self._cache) >= self.cache_max_size:
                    self._evict_oldest_entries()
                
                # Calculate size (approximate)
                try:
                    size_bytes = len(pickle.dumps(value))
                except:
                    size_bytes = len(str(value))
                
                # Create cache entry
                ttl = ttl_seconds or self.cache_default_ttl
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=datetime.now(),
                    ttl_seconds=ttl,
                    size_bytes=size_bytes
                )
                
                self._cache[key] = entry
                return True
                
        except Exception as e:
            logger.error(f"Error caching result: {e}")
            return False
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        try:
            if not self.cache_enabled:
                return None
            
            with self._cache_lock:
                if key not in self._cache:
                    return None
                
                entry = self._cache[key]
                
                # Check TTL
                if datetime.now() - entry.timestamp > timedelta(seconds=entry.ttl_seconds):
                    del self._cache[key]
                    return None
                
                # Update hit count
                entry.hit_count += 1
                return entry.value
                
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    def cached_operation(self, ttl_seconds: Optional[int] = None, key_func=None):
        """Decorator for caching operation results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                start_time = time.time()
                cached_result = self.get_cached_result(cache_key)
                
                if cached_result is not None:
                    execution_time = (time.time() - start_time) * 1000
                    self._record_metric(func.__name__, execution_time, True, len(str(cached_result)))
                    return cached_result
                
                # Execute operation
                result = func(*args, **kwargs)
                
                # Cache result
                execution_time = (time.time() - start_time) * 1000
                self.cache_result(cache_key, result, ttl_seconds)
                self._record_metric(func.__name__, execution_time, False, len(str(result)))
                
                return result
            
            return wrapper
        return decorator
    
    def optimize_search_query(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize search query parameters"""
        try:
            optimized = {
                'original_query': query,
                'optimized_query': query.strip(),
                'filters': filters.copy(),
                'optimizations_applied': []
            }
            
            # Query optimization
            if len(query.strip()) < 3:
                optimized['optimized_query'] = ""
                optimized['optimizations_applied'].append('query_too_short')
            
            # Filter optimization
            if 'price' in filters:
                price_filter = filters['price']
                if isinstance(price_filter, dict):
                    # Normalize price range
                    min_price = price_filter.get('min', 0)
                    max_price = price_filter.get('max', float('inf'))
                    
                    if min_price < 0:
                        min_price = 0
                        optimized['optimizations_applied'].append('negative_min_price')
                    
                    if max_price < min_price:
                        max_price = min_price * 2
                        optimized['optimizations_applied'].append('invalid_price_range')
                    
                    optimized['filters']['price'] = {'min': min_price, 'max': max_price}
            
            # Category optimization
            if 'category' in filters:
                category = filters['category']
                if isinstance(category, str):
                    optimized['filters']['category'] = category.lower().strip()
                    optimized['optimizations_applied'].append('category_normalization')
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing search query: {e}")
            return {'original_query': query, 'optimized_query': query, 'filters': filters, 'optimizations_applied': []}
    
    def batch_process_results(self, results: List[Any], batch_size: int = 50) -> List[List[Any]]:
        """Process results in batches for better performance"""
        try:
            if not results:
                return []
            
            batches = []
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                batches.append(batch)
            
            return batches
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [results] if results else []
    
    def parallel_process(self, items: List[Any], process_func, max_workers: int = 4) -> List[Any]:
        """Process items in parallel (simplified implementation)"""
        try:
            # This is a simplified version - in production, use ThreadPoolExecutor
            results = []
            
            # Process in chunks to simulate parallelism
            chunk_size = max(1, len(items) // max_workers)
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            
            for chunk in chunks:
                chunk_results = [process_func(item) for item in chunk]
                results.extend(chunk_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            return [process_func(item) for item in items]
    
    def optimize_vector_search(self, query_embedding: List[float], 
                             collection_size: int) -> Dict[str, Any]:
        """Optimize vector search parameters"""
        try:
            optimization = {
                'original_embedding_size': len(query_embedding),
                'collection_size': collection_size,
                'optimizations': []
            }
            
            # Embedding optimization
            if len(query_embedding) > 1000:
                optimization['optimizations'].append('large_embedding_dimension')
                # Suggest dimensionality reduction
            
            # Collection size optimization
            if collection_size > 100000:
                optimization['optimizations'].append('large_collection')
                # Suggest indexing strategies
            
            # Search parameter optimization
            optimization['suggested_k'] = min(100, max(10, collection_size // 1000))
            optimization['suggested_ef'] = min(200, max(50, collection_size // 5000))
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing vector search: {e}")
            return {}
    
    def monitor_performance(self, operation_name: str, execution_time_ms: float, 
                         result_size: int, cache_hit: bool = False):
        """Monitor operation performance"""
        try:
            metric = PerformanceMetrics(
                operation_name=operation_name,
                execution_time_ms=execution_time_ms,
                cache_hit=cache_hit,
                result_size=result_size,
                timestamp=datetime.now()
            )
            
            self.metrics.append(metric)
            self.operation_stats[operation_name].append(metric)
            
            # Keep only recent metrics (last 1000)
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
            
            # Alert on performance issues
            if execution_time_ms > self.optimization_thresholds['slow_operation_ms']:
                logger.warning(f"Slow operation detected: {operation_name} took {execution_time_ms:.2f}ms")
            
            if result_size > self.optimization_thresholds['large_result_size']:
                logger.info(f"Large result set: {operation_name} returned {result_size} items")
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
    
    def _record_metric(self, operation_name: str, execution_time_ms: float, 
                      cache_hit: bool, result_size: int):
        """Record performance metric"""
        self.monitor_performance(operation_name, execution_time_ms, result_size, cache_hit)
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function arguments"""
        try:
            # Create a hash from function name and arguments
            key_data = {
                'func': func_name,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"{func_name}_{hash(str(args) + str(kwargs))}"
    
    def _evict_oldest_entries(self, count: int = 10):
        """Evict oldest cache entries"""
        try:
            with self._cache_lock:
                if not self._cache:
                    return
                
                # Sort by timestamp and remove oldest
                sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
                
                for i in range(min(count, len(sorted_entries))):
                    key = sorted_entries[i][0]
                    del self._cache[key]
                    
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        try:
            with self._cache_lock:
                current_time = datetime.now()
                expired_keys = []
                
                for key, entry in self._cache.items():
                    if current_time - entry.timestamp > timedelta(seconds=entry.ttl_seconds):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        try:
            self._cleanup_running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting cleanup thread: {e}")
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while self._cleanup_running:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._cleanup_expired_entries()
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._cache_lock:
                total_entries = len(self._cache)
                total_size = sum(entry.size_bytes for entry in self._cache.values())
                avg_hit_count = sum(entry.hit_count for entry in self._cache.values()) / total_entries if total_entries > 0 else 0
                
                # Calculate hit rate
                recent_metrics = [m for m in self.metrics if m.timestamp > datetime.now() - timedelta(hours=1)]
                cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
                hit_rate = cache_hits / len(recent_metrics) if recent_metrics else 0
                
                return {
                    'total_entries': total_entries,
                    'total_size_bytes': total_size,
                    'average_hit_count': round(avg_hit_count, 2),
                    'recent_hit_rate': round(hit_rate, 3),
                    'cache_enabled': self.cache_enabled,
                    'max_size': self.cache_max_size,
                    'utilization': round(total_entries / self.cache_max_size, 3)
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = {
                'total_operations': len(self.metrics),
                'operation_stats': {},
                'slow_operations': [],
                'cache_performance': {}
            }
            
            # Operation-specific stats
            for operation, metrics in self.operation_stats.items():
                if metrics:
                    execution_times = [m.execution_time_ms for m in metrics]
                    cache_hits = sum(1 for m in metrics if m.cache_hit)
                    
                    stats['operation_stats'][operation] = {
                        'total_calls': len(metrics),
                        'avg_execution_time_ms': round(sum(execution_times) / len(execution_times), 2),
                        'min_execution_time_ms': round(min(execution_times), 2),
                        'max_execution_time_ms': round(max(execution_times), 2),
                        'cache_hit_rate': round(cache_hits / len(metrics), 3),
                        'avg_result_size': round(sum(m.result_size for m in metrics) / len(metrics), 1)
                    }
            
            # Slow operations
            slow_threshold = self.optimization_thresholds['slow_operation_ms']
            stats['slow_operations'] = [
                {
                    'operation': m.operation_name,
                    'execution_time_ms': m.execution_time_ms,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.metrics if m.execution_time_ms > slow_threshold
            ]
            
            # Cache performance
            recent_metrics = [m for m in self.metrics if m.timestamp > datetime.now() - timedelta(hours=1)]
            if recent_metrics:
                cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
                stats['cache_performance'] = {
                    'recent_hit_rate': round(cache_hits / len(recent_metrics), 3),
                    'total_recent_calls': len(recent_metrics),
                    'avg_execution_time_cached': round(
                        sum(m.execution_time_ms for m in recent_metrics if m.cache_hit) / cache_hits if cache_hits > 0 else 0, 2
                    ),
                    'avg_execution_time_uncached': round(
                        sum(m.execution_time_ms for m in recent_metrics if not m.cache_hit) / (len(recent_metrics) - cache_hits) if len(recent_metrics) - cache_hits > 0 else 0, 2
                    )
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cache entries"""
        try:
            with self._cache_lock:
                self._cache.clear()
                logger.info("Cache cleared")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def enable_cache(self):
        """Enable caching"""
        self.cache_enabled = True
        logger.info("Cache enabled")
    
    def disable_cache(self):
        """Disable caching"""
        self.cache_enabled = False
        logger.info("Cache disabled")
    
    def shutdown(self):
        """Shutdown the performance optimizer"""
        try:
            self._cleanup_running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)
            logger.info("Performance optimizer shutdown")
            
        except Exception as e:
            logger.error(f"Error shutting down performance optimizer: {e}")

# Global instance
performance_optimizer = PerformanceOptimizer()

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    return performance_optimizer
