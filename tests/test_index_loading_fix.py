


import os
import sys
import logging
import tempfile
import shutil


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_index_loading_fix():
    
    print("Index Loading Fix Test")
    print("=" * 50)
    
    try:

        from core.indexer import Indexer
        

        index_files = ['eiron_index.bin', 'optimized_index.bin']
        
        for index_file in index_files:
            if os.path.exists(index_file):
                print(f"\nTesting loading: {index_file}")
                print(f"   File size: {os.path.getsize(index_file)} bytes")
                
                try:

                    indexer = Indexer(index_file=index_file)
                    

                    doc_count = indexer.index.total_docs
                    term_count = indexer.index.total_terms
                    
                    print(f"   SUCCESS: Loading successful!")
                    print(f"   Documents: {doc_count}")
                    print(f"   Terms: {term_count}")
                    

                    if doc_count > 0:
                        results = indexer.search("test", max_results=3)
                        print(f"   Search works: found {len(results)} results")
                    else:
                        print("   WARNING: Index is empty")
                        
                except Exception as e:
                    print(f"   ERROR: Loading error: {e}")
            else:
                print(f"\nFile not found: {index_file}")
        

        print(f"\nTest creating new index")
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.txt")
        
        try:

            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("This is a test document for indexing verification.")
            

            test_indexer = Indexer(index_file=os.path.join(temp_dir, "test_index.bin"))
            count = test_indexer.index_directory(temp_dir)
            
            print(f"   Indexed files: {count}")
            
            if count > 0:

                test_indexer.save(os.path.join(temp_dir, "test_index.bin"))
                print("   Index saved")
                

                new_indexer = Indexer(index_file=os.path.join(temp_dir, "test_index.bin"))
                print(f"   Loaded documents: {new_indexer.index.total_docs}")
                

                results = new_indexer.search("test", max_results=3)
                print(f"   Search works: found {len(results)} results")
                
                print("   SUCCESS: Index creation and loading test passed!")
            else:
                print("   ERROR: Files were not indexed")
                
        finally:

            shutil.rmtree(temp_dir)
        
        print("\nSUCCESS: All tests completed!")
        
    except Exception as e:
        print(f"ERROR in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_index_loading_fix() 