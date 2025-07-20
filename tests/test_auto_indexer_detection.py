


import os
import sys
import logging
import tempfile
import shutil
import json
from datetime import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_auto_indexer_detection():
    
    print("Auto Index Type Detection Test")
    print("=" * 60)
    
    try:

        from core.index_detector import auto_detect_index_type, get_index_info, IndexType
        from core.indexer_factory import create_auto_indexer, get_auto_indexer_info
        
        print("SUCCESS: Modules imported successfully")
        

        print("\nTest 1: Analysis of existing indexes")
        
        index_files = ['eiron_index.bin', 'optimized_index.bin']
        
        for index_file in index_files:
            if os.path.exists(index_file):
                print(f"\n   Analyzing: {index_file}")
                

                index_info = get_index_info(index_file)
                auto_type = auto_detect_index_type(index_file)
                summary = get_auto_indexer_info(index_file)
                
                print(f"   File size: {index_info.file_size / 1024 / 1024:.1f} MB")
                print(f"   Detected type: {index_info.index_type.value}")
                print(f"   Recommended indexer: {auto_type}")
                print(f"   Confidence: {index_info.confidence:.2f}")
                print(f"   Documents: {index_info.total_docs}")
                print(f"   Terms: {index_info.total_terms}")
                print(f"   Version: {index_info.version}")
                print(f"   Compression: {index_info.compression}")
                print(f"   Metadata: {'Yes' if index_info.has_metadata else 'No'}")
                
                if index_info.confidence > 0.7:
                    print("   SUCCESS: High confidence in detection")
                elif index_info.confidence > 0.5:
                    print("   WARNING: Medium confidence in detection")
                else:
                    print("   ERROR: Low confidence in detection")
            else:
                print(f"\n   File not found: {index_file}")
        

        print("\nTest 2: Creation and testing of new indexes")
        
        with tempfile.TemporaryDirectory() as temp_dir:

            test_files = [
                ("test1.txt", "This is a test document about artificial intelligence."),
                ("test2.txt", "Machine learning is a subset of AI."),
                ("test3.txt", "Deep learning uses neural networks.")
            ]
            
            for filename, content in test_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            print(f"   Created {len(test_files)} test files")
            

            print("\n   Test 2.1: Standard indexer")
            standard_index_file = os.path.join(temp_dir, "standard_index.bin")
            
            try:
                from core.indexer import Indexer
                standard_indexer = Indexer(index_file=standard_index_file)
                count = standard_indexer.index_directory(temp_dir)
                standard_indexer.save(standard_index_file)
                
                print(f"   Indexed files: {count}")
                

                standard_info = get_index_info(standard_index_file)
                standard_auto_type = auto_detect_index_type(standard_index_file)
                
                print(f"   Detected type: {standard_info.index_type.value}")
                print(f"   Recommended indexer: {standard_auto_type}")
                print(f"   Confidence: {standard_info.confidence:.2f}")
                

                auto_standard = create_auto_indexer(standard_index_file)
                print(f"   SUCCESS: Auto indexer created: {type(auto_standard).__name__}")
                
            except Exception as e:
                print(f"   ERROR: Standard indexer error: {e}")
            

            print("\n   Test 2.2: Optimized indexer")
            optimized_index_file = os.path.join(temp_dir, "optimized_index.bin")
            
            try:
                from core.optimized_indexer import OptimizedIndexer
                optimized_indexer = OptimizedIndexer(index_file=optimized_index_file)
                count = optimized_indexer.index_directory(temp_dir)
                optimized_indexer.save(optimized_index_file)
                
                print(f"   Indexed files: {count}")
                

                optimized_info = get_index_info(optimized_index_file)
                optimized_auto_type = auto_detect_index_type(optimized_index_file)
                
                print(f"   Detected type: {optimized_info.index_type.value}")
                print(f"   Recommended indexer: {optimized_auto_type}")
                print(f"   Confidence: {optimized_info.confidence:.2f}")
                

                auto_optimized = create_auto_indexer(optimized_index_file)
                print(f"   SUCCESS: Auto indexer created: {type(auto_optimized).__name__}")
                
            except Exception as e:
                print(f"   ERROR: Optimized indexer error: {e}")
        

        print("\nTest 3: Factory testing with forced type")
        
        test_index_file = "eiron_index.bin"
        if os.path.exists(test_index_file):
            try:

                forced_standard = create_auto_indexer(test_index_file, force_type='standard')
                print(f"   SUCCESS: Forced standard indexer: {type(forced_standard).__name__}")
                

                forced_optimized = create_auto_indexer(test_index_file, force_type='optimized')
                print(f"   SUCCESS: Forced optimized indexer: {type(forced_optimized).__name__}")
                

                auto_indexer = create_auto_indexer(test_index_file)
                print(f"   SUCCESS: Auto indexer: {type(auto_indexer).__name__}")
                
            except Exception as e:
                print(f"   ERROR: Factory error: {e}")
        else:
            print(f"   File not found: {test_index_file}")
        

        print("\nTest 4: Testing non-existent files")
        
        non_existent_file = "non_existent_index.bin"
        try:
            info = get_index_info(non_existent_file)
            auto_type = auto_detect_index_type(non_existent_file)
            
            print(f"   Non-existent file type: {info.index_type.value}")
            print(f"   Recommended indexer: {auto_type}")
            print(f"   Confidence: {info.confidence:.2f}")
            

            new_indexer = create_auto_indexer(non_existent_file)
            print(f"   SUCCESS: Created indexer for new file: {type(new_indexer).__name__}")
            
        except Exception as e:
            print(f"   ERROR: Non-existent file error: {e}")
        
        print("\nSUCCESS: All auto detection tests completed successfully!")
        
    except Exception as e:
        print(f"ERROR in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_auto_indexer_detection() 