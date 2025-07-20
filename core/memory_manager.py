import psutil
import os
import gc
from typing import Optional, Dict, Any, Tuple
import json
from pathlib import Path
import logging
import time
from collections import OrderedDict
import sys
import weakref
import threading
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CacheEntry:
    value: Any
    size: int
    last_accessed: float
    access_count: int = 0

class MemoryManager:
    def __init__(self, config_path: str = "config/memory_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.memory_threshold = self.config.get("memory_threshold", 0.8)
        self.emergency_threshold = self.config.get("emergency_threshold", 0.4)
        self.chunk_size = self.config.get("chunk_size", 1024 * 1024)
        self.max_memory = self.config.get("max_memory", 0)
        self.max_cache_size = self.config.get("max_cache_size", 100 * 1024 * 1024)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.last_emergency_clear = 0
        self.emergency_cooldown = 60
        self._cache_lock = threading.Lock()
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        self._torch_available = False
        self._check_torch()
        self.start_memory_monitor()

    def _check_torch(self):
        
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False

    def start_memory_monitor(self):
        
        def monitor():
            while not self._stop_monitor.is_set():
                try:
                    if self.is_emergency_condition():
                        self.emergency_clear()
                    elif self.should_optimize():
                        self.optimize_memory()
                except Exception as e:
                    logging.error(f"Error in memory monitor: {e}")
                time.sleep(5)

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def stop_memory_monitor(self):
        
        if self._monitor_thread:
            self._stop_monitor.set()
            self._monitor_thread.join()
            self._monitor_thread = None

    def estimate_size(self, obj: Any) -> int:
        
        try:
            if isinstance(obj, (str, bytes, bytearray)):
                return sys.getsizeof(obj)
            elif isinstance(obj, (list, tuple, set)):
                return sum(self.estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self.estimate_size(k) + self.estimate_size(v) for k, v in obj.items())
            else:
                return sys.getsizeof(obj)
        except Exception:
            return 0

    def get_current_cache_size(self) -> int:
        
        return sum(entry.size for entry in self.cache.values())

    def evict_oldest(self, required_size: int):
        
        with self._cache_lock:
            while self.cache and self.get_current_cache_size() + required_size > self.max_cache_size:
                _, entry = self.cache.popitem(last=False)
                logging.debug(f"Evicted cache entry of size {entry.size} bytes")

    def cache_item(self, key: str, value: Any):
        
        size = self.estimate_size(value)
        

        if self.get_current_cache_size() + size > self.max_cache_size:
            self.evict_oldest(size)
            
        with self._cache_lock:
            self.cache[key] = CacheEntry(
                value=value,
                size=size,
                last_accessed=time.time()
            )

    def get_cached_item(self, key: str) -> Optional[Any]:
        
        with self._cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1

                self.cache.move_to_end(key)
                return entry.value
        return None

    def clear_cache(self):
        
        with self._cache_lock:
            cache_size = self.get_current_cache_size()
            self.cache.clear()
            gc.collect()
            logging.info(f"Cleared cache of {cache_size} bytes")

    def optimize_memory(self):
        
        if self.is_emergency_condition():
            self.emergency_clear()
            return

        if self.should_optimize():
            logging.info("Performing memory optimization")
            logging.info(f"Before optimization - Memory: {self.get_memory_usage():.1%}")
            

            gc.collect()
            

            if self.get_memory_usage() > self.memory_threshold:
                self.clear_cache()
                

            gc.collect()
            
            logging.info(f"After optimization - Memory: {self.get_memory_usage():.1%}")

    def emergency_clear(self):
        
        current_time = time.time()
        if current_time - self.last_emergency_clear < self.emergency_cooldown:
            return

        logging.warning("EMERGENCY: Performing emergency memory clear")
        logging.warning(f"Before clear - System Memory: {self.get_system_memory_usage():.1%}, "
                       f"Swap: {self.get_swap_usage():.1%}, "
                       f"Process: {self.get_memory_usage():.1%}")

        try:

            self.clear_cache()
            

            for _ in range(3):
                gc.collect()
            

            if self._torch_available:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"Error clearing torch cache: {e}")

            self.last_emergency_clear = current_time
            
            logging.warning(f"After clear - System Memory: {self.get_system_memory_usage():.1%}, "
                           f"Swap: {self.get_swap_usage():.1%}, "
                           f"Process: {self.get_memory_usage():.1%}")
        except Exception as e:
            logging.error(f"Error during emergency clear: {e}")

    def get_detailed_memory_stats(self) -> dict:
        
        return {
            "process_memory": self.get_memory_usage(),
            "system_memory": self.get_system_memory_usage(),
            "swap_usage": self.get_swap_usage(),
            "available_memory": self.get_available_memory(),
            "cache_size": self.get_current_cache_size(),
            "cache_items": len(self.cache),
            "chunk_size": self.chunk_size,
            "memory_threshold": self.memory_threshold,
            "emergency_threshold": self.emergency_threshold,
            "max_cache_size": self.max_cache_size
        }

    def __del__(self):
        
        self.stop_memory_monitor()

    def load_config(self) -> dict:
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading memory config: {e}")
        return {}

    def save_config(self):
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving memory config: {e}")

    def get_memory_usage(self) -> float:
        
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()

            return max(0.0, min(100.0, memory_percent)) / 100.0
        except Exception as e:
            logging.error(f"Error getting memory usage: {e}")
            return 0.0

    def get_system_memory_usage(self) -> float:
        
        return psutil.virtual_memory().percent / 100.0

    def get_available_memory(self) -> int:
        
        return psutil.virtual_memory().available

    def get_swap_usage(self) -> float:
        
        return psutil.swap_memory().percent / 100.0

    def should_optimize(self) -> bool:
        
        return self.get_memory_usage() > self.memory_threshold

    def is_emergency_condition(self) -> bool:
        
        current_time = time.time()
        if current_time - self.last_emergency_clear < self.emergency_cooldown:
            return False
            
        system_memory = self.get_system_memory_usage()
        swap_usage = self.get_swap_usage()
        process_memory = self.get_memory_usage()
        
        return (system_memory > self.emergency_threshold or 
                swap_usage > 0.3 or
                process_memory > 40.0)

    def set_memory_threshold(self, threshold: float):
        
        self.memory_threshold = max(0.1, min(0.9, threshold))
        self.config["memory_threshold"] = self.memory_threshold
        self.save_config()

    def set_emergency_threshold(self, threshold: float):
        
        self.emergency_threshold = max(0.1, min(0.9, threshold))
        self.config["emergency_threshold"] = self.emergency_threshold
        self.save_config()

    def set_chunk_size(self, size: int):
        
        self.chunk_size = max(1024, size)
        self.config["chunk_size"] = self.chunk_size
        self.save_config()

    def set_max_memory(self, max_memory: int):
        
        self.max_memory = max_memory
        self.config["max_memory"] = self.max_memory
        self.save_config()

    def get_optimal_chunk_size(self) -> int:
        
        available = self.get_available_memory()
        if self.max_memory > 0:
            available = min(available, self.max_memory)
        return min(self.chunk_size, available // 4)

    def remove_cached_item(self, key: str):
        
        self.cache.pop(key, None)

    def get_memory_stats(self) -> dict:
        
        return {
            "process_memory": self.get_memory_usage(),
            "system_memory": self.get_system_memory_usage(),
            "swap_usage": self.get_swap_usage(),
            "available_memory": self.get_available_memory(),
            "cache_size": self.get_current_cache_size(),
            "cache_items": len(self.cache),
            "chunk_size": self.chunk_size,
            "memory_threshold": self.memory_threshold,
            "emergency_threshold": self.emergency_threshold,
            "max_cache_size": self.max_cache_size
        } 