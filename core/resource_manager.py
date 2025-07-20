import os
import sys
import logging
import psutil
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import platform

@dataclass
class ResourceLimits:
    
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_file_size_mb: int = 100
    max_open_files: int = 1024
    max_processes: int = 1000
    enable_rlimit: bool = True

class ResourceManager:
    
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.is_unix = platform.system() in ['Linux', 'Darwin', 'FreeBSD']
        self._original_limits = {}
        self._setup_logging()
        
    def _setup_logging(self):
        
        self.logger = logging.getLogger(__name__)
        
    def set_rlimit(self, resource_type: str, soft_limit: int, hard_limit: int) -> bool:
        
        if not self.is_unix or not self.limits.enable_rlimit:
            return False
            
        try:
            import resource
            

            resource_map = {
                'RLIMIT_AS': resource.RLIMIT_AS,
                'RLIMIT_CPU': resource.RLIMIT_CPU,
                'RLIMIT_DATA': resource.RLIMIT_DATA,
                'RLIMIT_FSIZE': resource.RLIMIT_FSIZE,
                'RLIMIT_NOFILE': resource.RLIMIT_NOFILE,
                'RLIMIT_NPROC': resource.RLIMIT_NPROC,
                'RLIMIT_RSS': resource.RLIMIT_RSS,
                'RLIMIT_STACK': resource.RLIMIT_STACK,
            }
            
            if resource_type not in resource_map:
                self.logger.error(f"Unknown resource type: {resource_type}")
                return False
                

            original_soft, original_hard = resource.getrlimit(resource_map[resource_type])
            self._original_limits[resource_type] = (original_soft, original_hard)
            

            resource.setrlimit(resource_map[resource_type], (soft_limit, hard_limit))
            
            self.logger.info(f"Set {resource_type}: soft={soft_limit}, hard={hard_limit}")
            return True
            
        except ImportError:
            self.logger.warning("Resource module not available")
            return False
        except Exception as e:
            self.logger.error(f"Error setting resource limit {resource_type}: {e}")
            return False
    
    def setup_default_limits(self) -> bool:
        
        if not self.is_unix or not self.limits.enable_rlimit:
            self.logger.info("RLIMIT not enabled or not on Unix system")
            return False
            
        try:
            import resource
            

            memory_limit = self.limits.max_memory_mb * 1024 * 1024
            self.set_rlimit('RLIMIT_AS', memory_limit, memory_limit)
            

            file_size_limit = self.limits.max_file_size_mb * 1024 * 1024
            self.set_rlimit('RLIMIT_FSIZE', file_size_limit, file_size_limit)
            

            self.set_rlimit('RLIMIT_NOFILE', self.limits.max_open_files, self.limits.max_open_files)
            

            self.set_rlimit('RLIMIT_NPROC', self.limits.max_processes, self.limits.max_processes)
            
            self.logger.info("Default resource limits set successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting default limits: {e}")
            return False
    
    def restore_original_limits(self) -> bool:
        
        if not self.is_unix or not self.limits.enable_rlimit:
            return False
            
        try:
            import resource
            
            resource_map = {
                'RLIMIT_AS': resource.RLIMIT_AS,
                'RLIMIT_CPU': resource.RLIMIT_CPU,
                'RLIMIT_DATA': resource.RLIMIT_DATA,
                'RLIMIT_FSIZE': resource.RLIMIT_FSIZE,
                'RLIMIT_NOFILE': resource.RLIMIT_NOFILE,
                'RLIMIT_NPROC': resource.RLIMIT_NPROC,
                'RLIMIT_RSS': resource.RLIMIT_RSS,
                'RLIMIT_STACK': resource.RLIMIT_STACK,
            }
            
            for resource_type, (soft, hard) in self._original_limits.items():
                if resource_type in resource_map:
                    resource.setrlimit(resource_map[resource_type], (soft, hard))
                    self.logger.info(f"Restored {resource_type}: soft={soft}, hard={hard}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring original limits: {e}")
            return False
    
    def get_current_limits(self) -> Dict[str, Tuple[int, int]]:
        
        if not self.is_unix:
            return {}
            
        try:
            import resource
            
            limits = {}
            resource_map = {
                'RLIMIT_AS': resource.RLIMIT_AS,
                'RLIMIT_CPU': resource.RLIMIT_CPU,
                'RLIMIT_DATA': resource.RLIMIT_DATA,
                'RLIMIT_FSIZE': resource.RLIMIT_FSIZE,
                'RLIMIT_NOFILE': resource.RLIMIT_NOFILE,
                'RLIMIT_NPROC': resource.RLIMIT_NPROC,
                'RLIMIT_RSS': resource.RLIMIT_RSS,
                'RLIMIT_STACK': resource.RLIMIT_STACK,
            }
            
            for name, res in resource_map.items():
                try:
                    soft, hard = resource.getrlimit(res)
                    limits[name] = (soft, hard)
                except Exception as e:
                    self.logger.warning(f"Could not get limit for {name}: {e}")
            
            return limits
            
        except ImportError:
            return {}
        except Exception as e:
            self.logger.error(f"Error getting current limits: {e}")
            return {}
    
    def check_resource_usage(self) -> Dict[str, Any]:
        
        try:
            process = psutil.Process()
            

            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            memory_percent = max(0.0, min(100.0, memory_percent))
            

            cpu_percent = process.cpu_percent()
            cpu_count = psutil.cpu_count()
            

            num_fds = 0
            if self.is_unix:
                try:
                    num_fds = len(process.open_files())
                except:
                    pass
            
            return {
                'memory': {
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'percent': memory_percent
                },
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'files': {
                    'open_files': num_fds
                },
                'limits': self.get_current_limits()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking resource usage: {e}")
            return {}
    
    def is_within_limits(self) -> bool:
        
        usage = self.check_resource_usage()
        
        if not usage:
            return True
            

        if 'memory' in usage:
            memory_mb = usage['memory']['rss_mb']
            if memory_mb > self.limits.max_memory_mb:
                self.logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.limits.max_memory_mb}MB")
                return False
        

        if 'cpu' in usage:
            cpu_percent = usage['cpu']['percent']
            if cpu_percent > self.limits.max_cpu_percent:
                self.logger.warning(f"CPU usage {cpu_percent:.1f}% exceeds limit {self.limits.max_cpu_percent}%")
                return False
        

        if 'files' in usage and self.is_unix:
            open_files = usage['files']['open_files']
            if open_files > self.limits.max_open_files:
                self.logger.warning(f"Open files {open_files} exceeds limit {self.limits.max_open_files}")
                return False
        
        return True
    
    def __enter__(self):
        
        if self.limits.enable_rlimit:
            self.setup_default_limits()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        
        if self.limits.enable_rlimit:
            self.restore_original_limits() 