import psutil
import time
import logging
from typing import Dict, Any
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._last_net_io = psutil.net_io_counters()
        self._last_net_time = time.time()
        
    def get_cpu_stats(self) -> Dict[str, Any]:
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            cpu_stats = {
                'usage_percent': cpu_percent,
                'frequency': {
                    'current': cpu_freq.current if cpu_freq else 0,
                    'min': cpu_freq.min if cpu_freq else 0,
                    'max': cpu_freq.max if cpu_freq else 0
                },
                'cores': {
                    'physical': cpu_count,
                    'logical': psutil.cpu_count(logical=True)
                }
            }
            return cpu_stats
        except Exception as e:
            self.logger.error(f"Error getting CPU stats: {e}")
            return {}

    def get_memory_stats(self) -> Dict[str, Any]:
        
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                'virtual': {
                    'total': mem.total,
                    'available': mem.available,
                    'used': mem.used,
                    'percent': mem.percent
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {}

    def get_disk_stats(self) -> Dict[str, Any]:
        
        try:
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')
            return {
                'io': {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                },
                'usage': {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'percent': disk_usage.percent
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting disk stats: {e}")
            return {}

    def get_network_stats(self) -> Dict[str, Any]:
        
        try:
            current_net_io = psutil.net_io_counters()
            current_time = time.time()
            

            time_diff = current_time - self._last_net_time
            bytes_sent_rate = (current_net_io.bytes_sent - self._last_net_io.bytes_sent) / time_diff
            bytes_recv_rate = (current_net_io.bytes_recv - self._last_net_io.bytes_recv) / time_diff
            

            self._last_net_io = current_net_io
            self._last_net_time = current_time
            
            return {
                'bytes_sent': current_net_io.bytes_sent,
                'bytes_recv': current_net_io.bytes_recv,
                'packets_sent': current_net_io.packets_sent,
                'packets_recv': current_net_io.packets_recv,
                'send_rate': bytes_sent_rate,
                'recv_rate': bytes_recv_rate
            }
        except Exception as e:
            self.logger.error(f"Error getting network stats: {e}")
            return {}

    def get_process_stats(self) -> Dict[str, Any]:
        
        try:
            process = psutil.Process()
            with process.oneshot():
                return {
                                    'cpu_percent': process.cpu_percent(),
                'memory_percent': max(0.0, min(100.0, process.memory_percent())),
                    'memory_info': {
                        'rss': process.memory_info().rss,
                        'vms': process.memory_info().vms
                    },
                    'num_threads': process.num_threads(),
                    'num_handles': process.num_handles() if hasattr(process, 'num_handles') else 0,
                    'create_time': datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            self.logger.error(f"Error getting process stats: {e}")
            return {}

    def get_all_stats(self) -> Dict[str, Any]:
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu': self.get_cpu_stats(),
            'memory': self.get_memory_stats(),
            'disk': self.get_disk_stats(),
            'network': self.get_network_stats(),
            'process': self.get_process_stats()
        } 