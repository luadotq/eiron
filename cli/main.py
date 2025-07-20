import argparse
import logging
import os
from pathlib import Path
from core.indexer import Indexer
from core.optimized_indexer import OptimizedIndexer
from core.optimized_memory_manager import OptimizedMemoryManager
from semantic.search import EnhancedSemanticSearcher, SemanticSearchConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="EIRON CLI Search")
    parser.add_argument('--path', required=True, help="Path to directory with documents")
    parser.add_argument('--query', help="Search query")
    parser.add_argument('--search-mode', type=str, default='hybrid', 
                       choices=['keyword', 'semantic', 'hybrid', 'exact'],
                       help="Search mode: keyword, semantic, hybrid, or exact")
    parser.add_argument('--ext', default="txt,md,py,js,html,css,json,xml,pdf,csv", help="File extensions to index, comma-separated")
    parser.add_argument('--index-only', action='store_true', help="Only create/update index without searching")
    parser.add_argument('--max-results', type=int, default=10, help="Maximum number of results to show")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    

    parser.add_argument('--indexer', type=str, default='auto', 
                       choices=['auto', 'standard', 'optimized'],
                       help="Indexer type: auto (automatic detection), standard, or optimized")
    parser.add_argument('--index-file', type=str, default='eiron_index.bin',
                       help="Index file path")
    

    parser.add_argument('--max-memory-percent', type=float, default=80.0,
                       help="Maximum memory usage percentage (10-95)")
    parser.add_argument('--max-cache-size-mb', type=int, default=512,
                       help="Maximum cache size in MB (64-8192)")
    parser.add_argument('--emergency-threshold', type=float, default=95.0,
                       help="Emergency memory threshold percentage (80-99)")
    parser.add_argument('--optimization-interval', type=int, default=30,
                       help="Memory optimization interval in seconds (5-300)")
    

    parser.add_argument('--hide-paths', action='store_true', help="Hide file paths in output (use document IDs)")
    parser.add_argument('--dynamic-output', action='store_true', default=True, help="Enable dynamic output length based on relevance")
    parser.add_argument('--no-dynamic-output', dest='dynamic_output', action='store_false', help="Disable dynamic output length")
    parser.add_argument('--enable-rlimit', action='store_true', help="Enable RLIMIT resource management (Unix only)")
    parser.add_argument('--max-memory-mb', type=int, default=1024, help="Maximum memory usage in MB")
    parser.add_argument('--max-cpu-percent', type=float, default=80.0, help="Maximum CPU usage percentage")
    parser.add_argument('--max-file-size-mb', type=int, default=100, help="Maximum file size to process in MB")
    parser.add_argument('--max-open-files', type=int, default=1024, help="Maximum number of open files")
    
    args = parser.parse_args()


    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    

    try:
        memory_manager = OptimizedMemoryManager(
            max_memory_percent=args.max_memory_percent,
            max_cache_size_mb=args.max_cache_size_mb,
            emergency_threshold=args.emergency_threshold,
            optimization_interval=args.optimization_interval
        )
        logging.info(f"Memory manager initialized: {args.max_memory_percent}% max, {args.max_cache_size_mb}MB cache")
    except Exception as e:
        logging.error(f"Failed to create memory manager: {e}")
        return
    

    resource_limits = None
    if args.enable_rlimit:
        resource_limits = {
            'max_memory_mb': args.max_memory_mb,
            'max_cpu_percent': args.max_cpu_percent,
            'max_file_size_mb': args.max_file_size_mb,
            'max_open_files': args.max_open_files,
            'enable_rlimit': True
        }
    

    try:
        from core.indexer_factory import create_auto_indexer, get_auto_indexer_info
        

        force_type = None if args.indexer == 'auto' else args.indexer
        

        indexer_params = {}
        if args.indexer == 'standard':
            indexer_params.update({
                'hide_paths': args.hide_paths,
                'dynamic_output': args.dynamic_output,
                'resource_limits': resource_limits
            })
        elif args.indexer == 'optimized':
            indexer_params.update({
                'max_workers': 8,
                'batch_size': 50
            })
        
        indexer = create_auto_indexer(
            index_file=args.index_file,
            force_type=force_type,
            **indexer_params
        )
        

        index_info = get_auto_indexer_info(args.index_file)
        indexer_type = index_info.get('recommended_indexer', args.indexer)
        confidence = index_info.get('confidence', 0.0)
        
        logging.info(f"Using {indexer_type} indexer: {args.index_file} (confidence: {confidence:.2f})")
        
    except Exception as e:
        logging.error(f"Failed to create indexer: {e}")
        return
    

    exts = [f'.{e.strip().lower()}' for e in args.ext.split(',')]
    logging.info(f"Will index files with extensions: {exts}")
    

    if not os.path.exists(args.path):
        logging.error(f"Directory does not exist: {args.path}")
        return
    

    logging.info(f"Scanning directory: {args.path}")
    for root, dirs, files in os.walk(args.path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in exts:
                logging.debug(f"Found file: {file_path}")
    

    logging.info(f"Starting indexing in {args.path}")
    count = indexer.index_directory(args.path, extensions=exts)
    logging.info(f"Indexed {count} files")
    
    if args.index_only:
        return
    
    if not args.query:
        logging.error("Please provide a search query")
        return
    

    print(f"\nEnhanced Context Search (Mode: {args.search_mode})")
    print("=" * 50)
    
    results = indexer.search(
        args.query, 
        max_results=args.max_results,
        dynamic_context=args.dynamic_output,
        search_mode=args.search_mode
    )
    
    if not results:
        print("No results found.")
        return
    

    for i, (doc_id, score, context) in enumerate(results, 1):

        if args.hide_paths:
            original_path = indexer.index.documents.get(doc_id, {}).get('metadata', {}).get('original_path', doc_id)
            display_path = f"{doc_id} (original: {original_path})"
        else:
            display_path = doc_id
        
        print(f"\nResult {i}: {display_path}")
        print(f"Relevance: {score:.3f}")
        
        if context:
            print(f"Context: {context}")
        else:

            text = indexer.index.get_document_text(doc_id)
            if text:

                query_pos = text.lower().find(args.query.lower())
                if query_pos != -1:

                    start = max(0, query_pos - 100)
                    end = min(len(text), query_pos + len(args.query) + 100)
                    context = text[start:end]
                    

                    if start > 0:
                        context = "..." + context
                    if end < len(text):
                        context = context + "..."
                    
                    print(f"Context: {context}")
        
        print("-" * 50)
    

    try:
        stats = memory_manager.get_stats()
        detailed_stats = memory_manager.get_detailed_stats()
        print(f"\nMemory Statistics:")
        print(f"   Process Memory: {stats.process_memory_mb:.1f}MB")
        print(f"   Cache Size: {stats.cache_size_mb:.1f}MB")
        print(f"   Cache Items: {stats.cache_items}")
        print(f"   Optimizations: {detailed_stats['performance']['optimization_count']}")
        print(f"   Emergency Clears: {detailed_stats['performance']['emergency_clear_count']}")
    except Exception as e:
        logging.error(f"Error getting memory stats: {e}")

if __name__ == "__main__":
    main()
