


import os
import sys
import time
import logging
import tempfile
import shutil
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_test_files(directory: str, num_files: int = 100):
    
    logging.info(f"Creating {num_files} test files in {directory}")
    os.makedirs(directory, exist_ok=True)

    sample_texts = [
        "Artificial Intelligence (AI) is a field of computer science that deals with creating systems capable of performing tasks that require human intelligence.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
        "Deep learning uses neural networks to process large amounts of data and extract complex patterns.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Robotics combines AI with mechanical engineering to create intelligent machines that can perform physical tasks.",
        "AI ethics considers the moral aspects of developing and using AI systems.",
        "Big data analytics uses AI algorithms to process and analyze large datasets for insights and predictions.",
        "Autonomous vehicles use AI for real-time navigation and decision making.",
        "Healthcare AI applications include disease diagnosis, drug discovery, and personalized medicine."
    ]
    for i in range(num_files):

        if i % 4 == 0:
            ext = ".txt"
            content = f"Document {i+1}\n\n" + "\n\n".join(sample_texts) * 3
        elif i % 4 == 1:
            ext = ".md"
            content = f"# Document {i+1}\n\n" + "\n\n".join([f"## Section {j+1}\n{text}" for j, text in enumerate(sample_texts)])
        elif i % 4 == 2:
            ext = ".py"
            content = f"# File {i+1}\n\n" + "\n".join([f'print("{text}")' for text in sample_texts[:5]])
        else:
            ext = ".json"
            content = '{\n' + '\n'.join([f'  "section_{j+1}": "{text}"' for j, text in enumerate(sample_texts[:3])]) + '\n}'
        filename = f"test_file_{i+1:03d}{ext}"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    logging.info(f"Created {num_files} test files")

def test_standard_indexer(test_dir: str):
    
    logging.info("Testing standard indexer")
    try:
        from core.indexer import Indexer
        start_time = time.time()
        indexer = Indexer()
        processed_count = indexer.index_directory(test_dir)
        end_time = time.time()
        indexing_time = end_time - start_time
        logging.info(f"Standard indexer:")
        logging.info(f"   Indexing time: {indexing_time:.2f} seconds")
        logging.info(f"   Files processed: {processed_count}")
        logging.info(f"   Speed: {processed_count/indexing_time:.1f} files/sec")

        search_start = time.time()
        results = indexer.search("artificial intelligence", search_mode="hybrid", max_results=10)
        search_time = time.time() - search_start
        logging.info(f"   Search time: {search_time:.3f} seconds")
        logging.info(f"   Results found: {len(results)}")
        return {
            'indexing_time': indexing_time,
            'processed_count': processed_count,
            'search_time': search_time,
            'results_count': len(results)
        }
    except Exception as e:
        logging.error(f"Error in standard indexer: {e}")
        return None

def test_optimized_indexer(test_dir: str):
    
    logging.info("Testing optimized indexer")
    try:
        from core.optimized_indexer import OptimizedIndexer
        start_time = time.time()
        indexer = OptimizedIndexer(
            index_file='optimized_index.bin',
            max_workers=8,
            batch_size=50
        )
        processed_count = indexer.index_directory(test_dir)
        end_time = time.time()
        indexing_time = end_time - start_time
        logging.info(f"Optimized indexer:")
        logging.info(f"   Indexing time: {indexing_time:.2f} seconds")
        logging.info(f"   Files processed: {processed_count}")
        logging.info(f"   Speed: {processed_count/indexing_time:.1f} files/sec")

        stats = indexer.get_statistics()
        logging.info(f"   Index statistics:")
        logging.info(f"      Documents: {stats['total_documents']}")
        logging.info(f"      Terms: {stats['total_terms']}")
        logging.info(f"      Size: {stats['index_size_mb']:.1f}MB")

        search_start = time.time()
        results = indexer.search("artificial intelligence", search_mode="hybrid", max_results=10)
        search_time = time.time() - search_start
        logging.info(f"   Search time: {search_time:.3f} seconds")
        logging.info(f"   Results found: {len(results)}")
        return {
            'indexing_time': indexing_time,
            'processed_count': processed_count,
            'search_time': search_time,
            'results_count': len(results),
            'stats': stats
        }
    except Exception as e:
        logging.error(f"Error in optimized indexer: {e}")
        return None

def test_memory_manager():
    
    logging.info("Testing memory manager")
    try:
        from core.optimized_memory_manager import OptimizedMemoryManager
        manager = OptimizedMemoryManager(
            max_memory_percent=70.0,
            max_cache_size_mb=100.0
        )

        logging.info("Testing caching...")

        for i in range(50):
            key = f"test_key_{i}"
            value = f"test_value_{i}" * 1000
            success = manager.cache_set(key, value, priority=1.0 - (i / 50))
            if not success:
                logging.warning(f"Failed to add element {key}")

        stats = manager.get_detailed_stats()
        logging.info(f"Cache statistics:")
        logging.info(f"   Size: {stats['cache']['size_mb']:.1f}MB")
        logging.info(f"   Items: {stats['cache']['items']}")
        logging.info(f"   Maximum: {stats['cache']['max_size_mb']:.1f}MB")

        hit_count = 0
        for i in range(20):
            key = f"test_key_{i}"
            value = manager.cache_get(key)
            if value:
                hit_count += 1
        logging.info(f"   Hit rate: {hit_count/20*100:.1f}%")

        manager.cache_clear()
        manager.stop_monitoring()
        return True
    except Exception as e:
        logging.error(f"Error in memory manager: {e}")
        return None

def compare_performance(standard_results, optimized_results):
    
    if not standard_results or not optimized_results:
        logging.warning("Could not get results for comparison")
        return
    
    logging.info("PERFORMANCE COMPARISON")
    logging.info("=" * 50)
    

    indexing_improvement = (standard_results['indexing_time'] - optimized_results['indexing_time']) / standard_results['indexing_time'] * 100
    logging.info(f"Indexing:")
    logging.info(f"   Standard: {standard_results['indexing_time']:.2f}s")
    logging.info(f"   Optimized: {optimized_results['indexing_time']:.2f}s")
    logging.info(f"   Improvement: {indexing_improvement:+.1f}%")
    

    standard_speed = standard_results['processed_count'] / standard_results['indexing_time']
    optimized_speed = optimized_results['processed_count'] / optimized_results['indexing_time']
    speed_improvement = (optimized_speed - standard_speed) / standard_speed * 100
    
    logging.info(f"Processing speed:")
    logging.info(f"   Standard: {standard_speed:.1f} files/sec")
    logging.info(f"   Optimized: {optimized_speed:.1f} files/sec")
    logging.info(f"   Improvement: {speed_improvement:+.1f}%")
    

    search_improvement = (standard_results['search_time'] - optimized_results['search_time']) / standard_results['search_time'] * 100
    logging.info(f"Search:")
    logging.info(f"   Standard: {standard_results['search_time']:.3f}s")
    logging.info(f"   Optimized: {optimized_results['search_time']:.3f}s")
    logging.info(f"   Improvement: {search_improvement:+.1f}%")
    

    if indexing_improvement > 0 and speed_improvement > 0:
        logging.info("OPTIMIZATION SUCCESSFUL!")
    else:
        logging.info("Additional optimization required")

def main():
    
    logging.info("STARTING OPTIMIZED INDEXER TESTING")
    logging.info("=" * 60)
    

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_files")
        

        create_test_files(test_dir, num_files=200)
        

        standard_results = test_standard_indexer(test_dir)
        

        time.sleep(2)
        

        optimized_results = test_optimized_indexer(test_dir)
        

        compare_performance(standard_results, optimized_results)
        

        memory_success = test_memory_manager()
        

        logging.info("=" * 60)
        logging.info("FINAL REPORT")
        logging.info("=" * 60)
        
        if standard_results and optimized_results:
            logging.info("Indexing testing completed successfully")
        else:
            logging.error("Errors in indexing testing")
        
        if memory_success:
            logging.info("Memory manager testing completed successfully")
        else:
            logging.error("Errors in memory manager testing")
        
        logging.info("Recommendations:")
        logging.info("   - Use optimized indexer for large data volumes")
        logging.info("   - Configure memory manager parameters for your system")
        logging.info("   - Monitor resource usage during indexing")

if __name__ == "__main__":
    main() 