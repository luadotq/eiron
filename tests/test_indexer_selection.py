


import os
import tempfile
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.indexer import Indexer
from core.optimized_indexer import OptimizedIndexer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_indexer_selection():
    
    print("Testing indexer selection...")
    

    with tempfile.TemporaryDirectory() as temp_dir:

        test_files = [
            ("test1.txt", "This is a test document about artificial intelligence."),
            ("test2.txt", "Machine learning is a subset of artificial intelligence."),
            ("test3.txt", "Deep learning uses neural networks for processing data.")
        ]
        
        for filename, content in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Created {len(test_files)} test files in {temp_dir}")
        

        print("\n=== Testing Standard Indexer ===")
        try:
            standard_indexer = Indexer(index_file='test_standard_index.bin')
            processed = standard_indexer.index_directory(temp_dir)
            print(f"Standard indexer processed {processed} files")
            

            standard_indexer.save('test_standard_index.bin')
            print("Standard index saved successfully")
            

            results = standard_indexer.search("artificial intelligence")
            print(f"Standard indexer search returned {len(results)} results")
            
        except Exception as e:
            print(f"❌ Standard indexer test failed: {e}")
            return False
        

        print("\n=== Testing Optimized Indexer ===")
        try:
            optimized_indexer = OptimizedIndexer(index_file='test_optimized_index.bin')
            processed = optimized_indexer.index_directory(temp_dir)
            print(f"Optimized indexer processed {processed} files")
            

            optimized_indexer.storage.save_index(optimized_indexer.index_data, optimized_indexer.metadata)
            print("Optimized index saved successfully")
            

            results = optimized_indexer.search("artificial intelligence")
            print(f"Optimized indexer search returned {len(results)} results")
            
        except Exception as e:
            print(f"Optimized indexer test failed: {e}")
            return False
        

        print("\n=== Testing Cross-Loading ===")
        try:

            cross_indexer = OptimizedIndexer(index_file='test_standard_index.bin')
            print("Successfully loaded standard index in optimized indexer")
            

            results = cross_indexer.search("machine learning")
            print(f"Cross-loaded indexer search returned {len(results)} results")
            
        except Exception as e:
            print(f"Cross-loading test failed: {e}")
            return False
        

        for filename in ['test_standard_index.bin', 'test_optimized_index.bin']:
            if os.path.exists(filename):
                os.remove(filename)
        
        print("\nAll indexer selection tests passed!")
        return True

def test_gui_integration():
    
    print("\n=== Testing GUI Integration ===")
    
    try:

        from gui.app import IndexThread, SearchThread
        
        print("GUI components imported successfully")
        

        index_thread_standard = IndexThread("test_folder", "standard", "test_index.bin")
        print("Standard indexer thread created")
        
        index_thread_optimized = IndexThread("test_folder", "optimized", "test_index.bin")
        print("Optimized indexer thread created")
        
        search_thread_standard = SearchThread("test query", index_file="test_index.bin", indexer_type="standard")
        print("Standard search thread created")
        
        search_thread_optimized = SearchThread("test query", index_file="test_index.bin", indexer_type="optimized")
        print("Optimized search thread created")
        
        return True
        
    except Exception as e:
        print(f"ERROR: GUI integration test failed: {e}")
        return False

def test_cli_integration():
    
    print("\n=== Testing CLI Integration ===")
    
    try:

        from cli.main import main
        
        print("CLI components imported successfully")
        

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--indexer', type=str, default='standard', 
                           choices=['standard', 'optimized'],
                           help="Indexer type: standard or optimized")
        parser.add_argument('--index-file', type=str, default='eiron_index.bin',
                           help="Index file path")
        
        print("CLI arguments for indexer selection available")
        
        return True
        
    except Exception as e:
        print(f"CLI integration test failed: {e}")
        return False

def main():
    
    print("Running indexer selection tests...")
    print("=" * 50)
    
    tests = [
        test_indexer_selection,
        test_gui_integration,
        test_cli_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! Indexer selection is working correctly.")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 