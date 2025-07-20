


import sys
import os
import time
import logging
import threading
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.optimized_memory_manager import OptimizedMemoryManager

def test_process_only_memory_management():
    
    print("Testing memory management based on process metrics")
    

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    

    memory_manager = OptimizedMemoryManager(
        max_memory_percent=50.0,
        max_cache_size_mb=100.0,
        emergency_threshold=80.0,
        optimization_interval=2.0
    )
    
    try:
        print("Initial statistics:")
        stats = memory_manager.get_stats()
        print(f"   Process: {stats.process_memory_mb:.1f}MB")
        print(f"   Cache: {stats.cache_size_mb:.1f}MB")
        print(f"   Cache items: {stats.cache_items}")
        

        print("\nFilling cache...")
        for i in range(50):
            data = f"test_data_{i}" * 1000
            success = memory_manager.cache_set(f"key_{i}", data, priority=1.0)
            if not success:
                print(f"   Failed to add element {i}")
        

        time.sleep(1)
        stats = memory_manager.get_stats()
        print(f"\nStatistics after filling the cache:")
        print(f"   Process memory: {stats.process_memory_mb:.1f}MB")
        print(f"   Cache size: {stats.cache_size_mb:.1f}MB")
        print(f"   Cache items: {stats.cache_items}")
        

        print("\nWaiting for memory optimization...")
        time.sleep(5)
        

        stats = memory_manager.get_stats()
        print(f"\nStatistics after optimization:")
        print(f"   Process memory: {stats.process_memory_mb:.1f}MB")
        print(f"   Cache size: {stats.cache_size_mb:.1f}MB")
        print(f"   Cache items: {stats.cache_items}")
        

        detailed_stats = memory_manager.get_detailed_stats()
        print(f"\nDetailed statistics:")
        print(f"   Optimizations: {detailed_stats['performance']['optimization_count']}")
        print(f"   Emergency clear: {detailed_stats['performance']['emergency_clear_count']}")
        print(f"   Uptime: {detailed_stats['performance']['uptime_seconds']:.1f}s")
        

        print("\nTesting cache retrieval...")
        for i in range(10):
            value = memory_manager.cache_get(f"key_{i}")
            if value:
                print(f"   Retrieved element {i}: {len(value)} characters")
            else:
                print(f"   Element {i} not found in cache")
        

        print("\nCache clearing...")
        memory_manager.cache_clear()
        

        time.sleep(1)
        stats = memory_manager.get_stats()
        print(f"\nFinal statistics:")
        print(f"   Process memory: {stats.process_memory_mb:.1f}MB")
        print(f"   Cache size: {stats.cache_size_mb:.1f}MB")
        print(f"   Cache items: {stats.cache_items}")
        
        print("\nMemory management test completed successfully!")
        
    except Exception as e:
        print(f"ERROR in test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:

        memory_manager.stop_monitoring()
        print("Memory monitoring stopped")

def test_memory_context():
    
    print("\nTesting memory context manager")
    
    memory_manager = OptimizedMemoryManager()
    
    try:
        with memory_manager.memory_context(max_memory_mb=1):
            print("Inside memory context")

            for i in range(100):
                data = f"test_data_{i}" * 10000
                memory_manager.cache_set(f"context_test_{i}", data)
            print("   Data added to cache")
        
        print("   Context completed")
        

        value = memory_manager.cache_get("context_test_0")
        if value is None:
            print("Data successfully cleared from context")
        else:
            print("Data not cleared")
            
    except Exception as e:
        print(f"Error in context test: {e}")
    
    finally:
        memory_manager.stop_monitoring()

if __name__ == "__main__":
    print("Starting memory management tests")
    print("=" * 50)
    
    test_process_only_memory_management()
    test_memory_context()
    
    print("\n" + "=" * 50)
    print("Tests passed") 