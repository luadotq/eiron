


import os
import sys
import tempfile
import logging


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimized_indexer import OptimizedIndexer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_save_method():
    
    print("Testing OptimizedIndexer save method...")
    

    with tempfile.TemporaryDirectory() as temp_dir:

        test_files = [
            ("test1.txt", "This is a test document about artificial intelligence."),
            ("test2.txt", "Machine learning is a subset of artificial intelligence."),
        ]
        
        for filename, content in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Created {len(test_files)} test files")
        
        try:

            indexer = OptimizedIndexer(index_file='test_save_index.bin')
            print("OptimizedIndexer created successfully")
            

            processed = indexer.index_directory(temp_dir)
            print(f"Indexed {processed} files")
            

            indexer.save('test_save_index.bin')
            print("Save method works correctly")
            

            if os.path.exists('test_save_index.bin'):
                print("Index file created successfully")
                file_size = os.path.getsize('test_save_index.bin')
                print(f"Index file size: {file_size} bytes")
            else:
                print("ERROR: Index file not created")
                return False
            

            new_indexer = OptimizedIndexer(index_file='test_save_index.bin')
            new_indexer.load('test_save_index.bin')
            print("Load method works correctly")
            

            stats = new_indexer.get_statistics()
            print(f"Loaded index statistics: {stats['total_documents']} documents")
            

            results = new_indexer.search("artificial intelligence")
            print(f"Search returned {len(results)} results")
            

            if os.path.exists('test_save_index.bin'):
                os.remove('test_save_index.bin')
            
            print("\nAll save/load tests passed!")
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False

def test_gui_compatibility():
    
    print("\n=== Testing GUI Compatibility ===")
    
    try:

        from gui.app import IndexThread
        
        print("GUI components imported successfully")
        

        index_thread = IndexThread("test_folder", "optimized", "test_gui_index.bin")
        print("OptimizedIndexer thread created successfully")
        

        if hasattr(index_thread.indexer, 'save'):
            print("OptimizedIndexer has save method")
        else:
            print("OptimizedIndexer missing save method")
            return False
        

        if hasattr(index_thread.indexer, 'load'):
            print("OptimizedIndexer has load method")
        else:
            print("OptimizedIndexer missing load method")
            return False
        
        return True
        
    except Exception as e:
        print(f"GUI compatibility test failed: {e}")
        return False

def main():
    
    print("Running save method tests...")
    print("=" * 50)
    
    tests = [
        test_save_method,
        test_gui_compatibility
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
        print("All tests passed!")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 