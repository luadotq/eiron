import psutil
import os
import gc
import threading
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref
import sys
from contextlib import contextmanager

@dataclass
class MemoryStats:
    
    process_memory_mb: float = 0.0
    system_memory_percent: float = 0.0
    available_memory_mb: float = 0.0
    swap_usage_percent: float = 0.0
    cpu_percent: float = 0.0
    cache_size_mb: float = 0.0
    cache_items: int = 0

@dataclass
class CacheItem:
    
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    priority: float = 1.0

class OptimizedMemoryManager:
    
    
    def __init__(self, 
                 max_memory_percent: float = 80.0,
                 max_cache_size_mb: float = 512.0,
                 emergency_threshold: float = 95.0,
                 optimization_interval: float = 30.0):
        
        self.max_memory_percent = max_memory_percent / 100.0
        self.emergency_threshold = emergency_threshold / 100.0
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.optimization_interval = optimization_interval
        

        self.start_time = time.time()
        

        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.cache_lock = threading.RLock()
        

        self.monitor_thread = None
        self.stop_monitor = threading.Event()
        self.last_optimization = 0
        self.last_emergency_clear = 0
        self.emergency_clear_cooldown = 120.0
        

        self.stats = MemoryStats()
        self.optimization_count = 0
        self.emergency_clear_count = 0
        

        self.start_monitoring()
        
        logging.info(f"Optimized memory manager initialized")
        logging.info(f"Limits: memory {max_memory_percent}%, cache {max_cache_size_mb}MB, emergency {emergency_threshold}%")
    
    def start_monitoring(self):
        
        def monitor_loop():
            while not self.stop_monitor.is_set():
                try:
                    self._update_stats()
                    

                    if self._should_optimize():
                        self.optimize_memory()
                    

                    if self._is_emergency_condition():
                        self.emergency_clear()
                    
                    time.sleep(self.optimization_interval)
                    
                except Exception as e:
                    logging.error(f"Memory monitor error: {e}")
                    time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Memory monitoring started")
    
    def stop_monitoring(self):
        
        self.stop_monitor.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("Memory monitoring stopped")
    
    def _update_stats(self):
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            with self.cache_lock:
                cache_size = sum(item.size_bytes for item in self.cache.values())
                cache_items = len(self.cache)
            
            self.stats = MemoryStats(
                process_memory_mb=memory_info.rss / (1024 * 1024),
                system_memory_percent=system_memory.percent,
                available_memory_mb=system_memory.available / (1024 * 1024),
                swap_usage_percent=psutil.swap_memory().percent,
                cpu_percent=process.cpu_percent(),
                cache_size_mb=cache_size / (1024 * 1024),
                cache_items=cache_items
            )
            
        except Exception as e:
            logging.error(f"Error updating statistics: {e}")
    
    def _should_optimize(self) -> bool:
        
        current_time = time.time()
        

        if current_time - self.last_optimization < self.optimization_interval:
            return False
        

        process_memory_mb = self.stats.process_memory_mb
        cache_size_mb = self.stats.cache_size_mb
        

        should_optimize = (
            process_memory_mb > 1500 or
            cache_size_mb > self.max_cache_size_bytes / (1024 * 1024) * 0.9
        )
        

        if should_optimize:
            logging.debug(f"Memory optimization triggered: "
                         f"process {process_memory_mb:.1f}MB, "
                         f"cache {cache_size_mb:.1f}MB")
        
        return should_optimize
    
    def _is_emergency_condition(self) -> bool:
        
        current_time = time.time()
        

        if current_time - self.last_emergency_clear < self.emergency_clear_cooldown:
            return False
        

        process_memory_mb = self.stats.process_memory_mb
        cache_size_mb = self.stats.cache_size_mb
        

        is_emergency = (
            process_memory_mb > 3000 or
            cache_size_mb > self.max_cache_size_bytes / (1024 * 1024) * 0.98
        )
        

        if is_emergency:
            logging.warning(f"Emergency condition detected: "
                           f"process {process_memory_mb:.1f}MB, "
                           f"cache {cache_size_mb:.1f}MB")
        
        return is_emergency
    
    def optimize_memory(self):
        
        self.optimization_count += 1
        self.last_optimization = time.time()
        
        logging.info(f"Memory optimization #{self.optimization_count}")
        logging.info(f"Before optimization: {self.stats.process_memory_mb:.1f}MB, "
                    f"cache {self.stats.cache_size_mb:.1f}MB")
        
        try:

            self._cleanup_cache()
            

            collected = gc.collect()
            

            gc.collect(2)
            

            self._update_stats()
            
            logging.info(f"After optimization: {self.stats.process_memory_mb:.1f}MB, "
                        f"cache {self.stats.cache_size_mb:.1f}MB, "
                        f"collected {collected} objects")
            
        except Exception as e:
            logging.error(f"Error optimizing memory: {e}")
    
    def emergency_clear(self):
        
        self.emergency_clear_count += 1
        self.last_emergency_clear = time.time()
        
        logging.warning(f"EMERGENCY CLEAR #{self.emergency_clear_count}")
        logging.warning(f"Process memory: {self.stats.process_memory_mb:.1f}MB, "
                       f"Cache: {self.stats.cache_size_mb:.1f}MB")
        
        try:

            with self.cache_lock:
                self.cache.clear()
            

            for _ in range(3):
                gc.collect()
            



            if hasattr(gc, 'garbage'):
                gc.garbage.clear()
            

            self._clear_external_caches()
            

            self._update_stats()
            
            logging.warning(f"Emergency clear completed: "
                           f"{self.stats.process_memory_mb:.1f}MB, "
                           f"cache {self.stats.cache_size_mb:.1f}MB")
            
        except Exception as e:
            logging.error(f"Error during emergency clear: {e}")
    
    def _clear_external_caches(self):
        
        try:

            import numpy as np

        except ImportError:
            pass
        
        try:

            import sklearn

        except ImportError:
            pass
        
        try:

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def _cleanup_cache(self):
        
        with self.cache_lock:
            if not self.cache:
                return
            

            items_to_remove = []
            current_size = sum(item.size_bytes for item in self.cache.values())
            

            if current_size > self.max_cache_size_bytes:
                target_size = self.max_cache_size_bytes * 0.8
                

                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: (x[1].priority, x[1].last_accessed)
                )
                
                for key, item in sorted_items:
                    if current_size <= target_size:
                        break
                    
                    items_to_remove.append(key)
                    current_size -= item.size_bytes
            

            for key in items_to_remove:
                del self.cache[key]
            
            if items_to_remove:
                logging.info(f"Removed {len(items_to_remove)} cache items")
    
    def cache_set(self, key: str, value: Any, size_bytes: Optional[int] = None, 
                  priority: float = 1.0) -> bool:
        
        try:
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            

            if size_bytes > self.max_cache_size_bytes:
                logging.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            with self.cache_lock:

                current_size = sum(item.size_bytes for item in self.cache.values())
                if current_size + size_bytes > self.max_cache_size_bytes:
                    self._cleanup_cache()
                

                self.cache[key] = CacheItem(
                    value=value,
                    size_bytes=size_bytes,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    priority=priority
                )
                

                self.cache.move_to_end(key)
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding to cache: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[Any]:
        
        try:
            with self.cache_lock:
                if key in self.cache:
                    item = self.cache[key]
                    item.last_accessed = time.time()
                    item.access_count += 1
                    

                    self.cache.move_to_end(key)
                    
                    return item.value
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting from cache: {e}")
            return None
    
    def cache_remove(self, key: str) -> bool:
        
        try:
            with self.cache_lock:
                if key in self.cache:
                    del self.cache[key]
                    return True
            return False
            
        except Exception as e:
            logging.error(f"Error removing from cache: {e}")
            return False
    
    def cache_clear(self):
        
        try:
            with self.cache_lock:
                size = sum(item.size_bytes for item in self.cache.values())
                self.cache.clear()
            
            logging.info(f"Cache cleared, freed {size / (1024 * 1024):.1f}MB")
            
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
    
    def _estimate_size(self, obj: Any) -> int:
        
        try:
            if isinstance(obj, (str, bytes, bytearray)):
                return sys.getsizeof(obj)
            elif isinstance(obj, (list, tuple, set)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in obj.items())
            else:
                return sys.getsizeof(obj)
        except Exception:
            return 1024
    
    def get_stats(self) -> MemoryStats:
        
        self._update_stats()
        return self.stats
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        
        stats = self.get_stats()
        
        return {
            'memory': {
                'process_mb': stats.process_memory_mb,
                'system_percent': stats.system_memory_percent,
                'available_mb': stats.available_memory_mb,
                'swap_percent': stats.swap_usage_percent
            },
            'performance': {
                'cpu_percent': stats.cpu_percent,
                'optimization_count': self.optimization_count,
                'emergency_clear_count': self.emergency_clear_count,
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            },
            'cache': {
                'size_mb': stats.cache_size_mb,
                'items': stats.cache_items,
                'max_size_mb': self.max_cache_size_bytes / (1024 * 1024)
            },
            'limits': {
                'max_memory_percent': self.max_memory_percent * 100,
                'emergency_threshold': self.emergency_threshold * 100
            }
        }
    
    @contextmanager
    def memory_context(self, max_memory_mb: Optional[float] = None):
        
        initial_stats = self.get_stats()
        initial_cache_size = len(self.cache)
        
        try:
            yield self
        finally:

            current_stats = self.get_stats()
            memory_increase = current_stats.process_memory_mb - initial_stats.process_memory_mb
            
            if max_memory_mb and memory_increase > max_memory_mb:
                logging.warning(f"Memory limit exceeded: +{memory_increase:.1f}MB")

                self.cache_clear()
                self.optimize_memory()
    
    def __del__(self):
        
        self.stop_monitoring() 