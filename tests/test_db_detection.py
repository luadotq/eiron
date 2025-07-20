


import os
import sys
import logging
import tempfile
import shutil


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_db_detection():
    
    print("Database Detection Test")
    print("=" * 60)
    
    try:

        from core.index_detector import auto_detect_index_type, get_index_info, IndexType
        from core.indexer_factory import create_auto_indexer, get_auto_indexer_info
        
        print("SUCCESS: Modules imported successfully")
        

        print("\nTest 1: Analysis of existing indexes with database")
        
        index_files = ['eiron_index.bin', 'optimized_index.bin']
        
        for index_file in index_files:
            if os.path.exists(index_file):
                print(f"\n   Analyzing: {index_file}")
                

                metadata_file = f"{index_file}.meta"
                db_file = f"{index_file}.db"
                
                has_metadata = os.path.exists(metadata_file)
                has_database = os.path.exists(db_file)
                
                print(f"   Main file: {os.path.getsize(index_file) / 1024 / 1024:.1f} MB")
                print(f"   Metadata: {'Yes' if has_metadata else 'No'}")
                if has_metadata:
                    print(f"   Metadata size: {os.path.getsize(metadata_file)} bytes")
                print(f"   Database: {'Yes' if has_database else 'No'}")
                if has_database:
                    print(f"   Database size: {os.path.getsize(db_file) / 1024 / 1024:.1f} MB")
                

                index_info = get_index_info(index_file)
                auto_type = auto_detect_index_type(index_file)
                summary = get_auto_indexer_info(index_file)
                
                print(f"   Detected type: {index_info.index_type.value}")
                print(f"   Recommended indexer: {auto_type}")
                print(f"   Confidence: {index_info.confidence:.2f}")
                print(f"   Documents: {index_info.total_docs}")
                print(f"   Terms: {index_info.total_terms}")
                print(f"   Version: {index_info.version}")
                print(f"   Compression: {index_info.compression}")
                

                print(f"   Total size: {summary.get('total_size', 0) / 1024 / 1024:.1f} MB")
                print(f"   Has DB: {summary.get('has_database', False)}")
                
                if index_info.confidence > 0.7:
                    print("   SUCCESS: High confidence in detection")
                elif index_info.confidence > 0.5:
                    print("   WARNING: Medium confidence in detection")
                else:
                    print("   ERROR: Low confidence in detection")
                    

                try:
                    indexer = create_auto_indexer(index_file)
                    print(f"   SUCCESS: Indexer created: {type(indexer).__name__}")
                except Exception as e:
                    print(f"   ERROR: Failed to create indexer: {e}")
            else:
                print(f"\n   File not found: {index_file}")
        

        print("\nTest 2: Check main file only without DB")
        

        with tempfile.TemporaryDirectory() as temp_dir:
            test_index = os.path.join(temp_dir, "test_index.bin")
            

            if os.path.exists('eiron_index.bin'):
                shutil.copy2('eiron_index.bin', test_index)
                
                print(f"   File copied: {test_index}")
                

                index_info = get_index_info(test_index)
                auto_type = auto_detect_index_type(test_index)
                
                print(f"   Detected type: {index_info.index_type.value}")
                print(f"   Recommended indexer: {auto_type}")
                print(f"   Confidence: {index_info.confidence:.2f}")
                print(f"   Has DB: {os.path.exists(test_index + '.db')}")
                

                try:
                    indexer = create_auto_indexer(test_index)
                    print(f"   SUCCESS: Indexer created: {type(indexer).__name__}")
                except Exception as e:
                    print(f"   ERROR: Failed to create indexer: {e}")
            else:
                print("   Source file not found")
        

        print("\nTest 3: Check DB only without main file")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db = os.path.join(temp_dir, "test_index.bin.db")
            

            if os.path.exists('eiron_index.bin.db'):
                shutil.copy2('eiron_index.bin.db', test_db)
                
                print(f"   Database copied: {test_db}")
                

                test_index = os.path.join(temp_dir, "test_index.bin")
                index_info = get_index_info(test_index)
                auto_type = auto_detect_index_type(test_index)
                
                print(f"   Detected type: {index_info.index_type.value}")
                print(f"   Recommended indexer: {auto_type}")
                print(f"   Confidence: {index_info.confidence:.2f}")
                print(f"   Has DB: {os.path.exists(test_db)}")
                

                try:
                    indexer = create_auto_indexer(test_index)
                    print(f"   SUCCESS: Indexer created: {type(indexer).__name__}")
                except Exception as e:
                    print(f"   ERROR: Failed to create indexer: {e}")
            else:
                print("   Source database not found")
        
        print("\nSUCCESS: All database detection tests completed!")
        
    except Exception as e:
        print(f"ERROR in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_db_detection() 