import os
import json
import pickle
import logging
import sqlite3
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class IndexType(Enum):
    
    STANDARD = "standard"
    OPTIMIZED = "optimized"
    UNKNOWN = "unknown"
    CORRUPTED = "corrupted"

@dataclass
class IndexInfo:
    
    index_type: IndexType
    version: str = ""
    total_docs: int = 0
    total_terms: int = 0
    compression: str = ""
    created_at: str = ""
    last_updated: str = ""
    file_size: int = 0
    has_metadata: bool = False
    confidence: float = 0.0

class IndexDetector:
    
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _get_db_stats(self, db_file: str):
        try:
            with sqlite3.connect(db_file) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM documents")
                total_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT term) FROM terms")
                total_terms = cur.fetchone()[0]
                return total_docs, total_terms
        except Exception as e:
            self.logger.warning(f"Error reading DB stats: {e}")
            return 0, 0
    
    def detect_index_type(self, index_file: str) -> IndexInfo:
        
        if not os.path.exists(index_file):
            return IndexInfo(
                index_type=IndexType.UNKNOWN,
                confidence=0.0
            )
        
        file_size = os.path.getsize(index_file)
        if file_size == 0:
            return IndexInfo(
                index_type=IndexType.CORRUPTED,
                file_size=0,
                confidence=0.0
            )
        

        metadata_file = f"{index_file}.meta"
        db_file = f"{index_file}.db"
        has_metadata = os.path.exists(metadata_file)
        has_database = os.path.exists(db_file)
        

        if has_database:
            self.logger.info(f"Found database file: {db_file}")
            total_docs, total_terms = self._get_db_stats(db_file)
            if has_metadata:
                metadata_info = self._analyze_metadata(metadata_file)
                metadata_info.file_size = file_size
                metadata_info.has_metadata = True
                metadata_info.confidence = max(metadata_info.confidence, 0.95)
                metadata_info.total_docs = total_docs
                metadata_info.total_terms = total_terms
                return metadata_info
            else:
                return IndexInfo(
                    index_type=IndexType.OPTIMIZED,
                    file_size=file_size,
                    has_metadata=False,
                    confidence=0.9,
                    total_docs=total_docs,
                    total_terms=total_terms
                )
        

        if has_metadata:
            metadata_info = self._analyze_metadata(metadata_file)
            if metadata_info.confidence > 0.7:
                metadata_info.file_size = file_size
                metadata_info.has_metadata = True
                return metadata_info
        

        content_info = self._analyze_content(index_file)
        content_info.file_size = file_size
        content_info.has_metadata = has_metadata
        
        return content_info
    
    def _analyze_metadata(self, metadata_file: str) -> IndexInfo:
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            

            version = metadata.get('version', '')
            compression = metadata.get('compression', '')
            
            if version == '2.0' and compression == 'lz4':
                return IndexInfo(
                    index_type=IndexType.OPTIMIZED,
                    version=version,
                    total_docs=metadata.get('total_docs', 0),
                    total_terms=metadata.get('total_terms', 0),
                    compression=compression,
                    created_at=metadata.get('created_at', ''),
                    last_updated=metadata.get('last_updated', ''),
                    confidence=0.9
                )
            elif version == '1.0' or not version:
                return IndexInfo(
                    index_type=IndexType.STANDARD,
                    version=version or '1.0',
                    total_docs=metadata.get('total_docs', 0),
                    total_terms=metadata.get('total_terms', 0),
                    compression=compression or 'none',
                    created_at=metadata.get('created_at', ''),
                    last_updated=metadata.get('last_updated', ''),
                    confidence=0.8
                )
            else:
                return IndexInfo(
                    index_type=IndexType.UNKNOWN,
                    version=version,
                    confidence=0.5
                )
                
        except Exception as e:
            self.logger.debug(f"Error analyzing metadata: {e}")
            return IndexInfo(
                index_type=IndexType.UNKNOWN,
                confidence=0.3
            )
    
    def _analyze_content(self, index_file: str) -> IndexInfo:
        
        try:

            try:
                import lz4.frame
                with open(index_file, 'rb') as f:
                    compressed_data = f.read()
                

                if len(compressed_data) >= 4:
                    magic = compressed_data[:4]
                    if magic == b'\x04\x22\x4d\x18':
                        data = pickle.loads(lz4.frame.decompress(compressed_data))
                        return self._analyze_pickle_data(data, IndexType.OPTIMIZED, 0.85)
            except ImportError:
                pass
            except Exception:
                pass
            

            try:
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                return self._analyze_pickle_data(data, IndexType.STANDARD, 0.8)
            except Exception:
                pass
            

            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._analyze_json_data(data, 0.7)
            except Exception:
                pass
            

            return IndexInfo(
                index_type=IndexType.UNKNOWN,
                confidence=0.2
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing content: {e}")
            return IndexInfo(
                index_type=IndexType.CORRUPTED,
                confidence=0.1
            )
    
    def _analyze_pickle_data(self, data: Dict, expected_type: IndexType, base_confidence: float) -> IndexInfo:
        
        try:

            required_keys = {'index', 'documents', 'total_docs', 'total_terms'}
            has_required_keys = all(key in data for key in required_keys)
            
            if not has_required_keys:
                return IndexInfo(
                    index_type=IndexType.UNKNOWN,
                    confidence=base_confidence * 0.5
                )
            

            index_type = expected_type
            confidence = base_confidence
            

            if expected_type == IndexType.OPTIMIZED:

                if 'symbol_counts' in data or 'metadata' in data:
                    confidence += 0.1
            

            total_docs = data.get('total_docs', 0)
            documents_count = len(data.get('documents', {}))
            

            if abs(total_docs - documents_count) <= 5:
                confidence += 0.05
            
            return IndexInfo(
                index_type=index_type,
                total_docs=documents_count,
                total_terms=data.get('total_terms', 0),
                confidence=min(confidence, 1.0)
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing pickle data: {e}")
            return IndexInfo(
                index_type=IndexType.UNKNOWN,
                confidence=base_confidence * 0.3
            )
    
    def _analyze_json_data(self, data: Dict, base_confidence: float) -> IndexInfo:
        
        try:

            required_keys = {'index', 'documents', 'total_docs', 'total_terms'}
            has_required_keys = all(key in data for key in required_keys)
            
            if not has_required_keys:
                return IndexInfo(
                    index_type=IndexType.UNKNOWN,
                    confidence=base_confidence * 0.5
                )
            
            return IndexInfo(
                index_type=IndexType.STANDARD,
                total_docs=len(data.get('documents', {})),
                total_terms=data.get('total_terms', 0),
                confidence=base_confidence
            )
            
        except Exception as e:
            self.logger.debug(f"Error analyzing JSON data: {e}")
            return IndexInfo(
                index_type=IndexType.UNKNOWN,
                confidence=base_confidence * 0.3
            )
    
    def get_recommended_indexer(self, index_info: IndexInfo) -> str:
        

        if index_info.confidence < 0.3:

            return 'standard'
        
        if index_info.index_type == IndexType.OPTIMIZED:
            return 'optimized'
        elif index_info.index_type == IndexType.STANDARD:
            return 'standard'
        else:

            return 'optimized'
    
    def get_index_summary(self, index_file: str) -> Dict[str, Any]:
        
        index_info = self.detect_index_type(index_file)
        recommended_indexer = self.get_recommended_indexer(index_info)
        

        metadata_file = f"{index_file}.meta"
        db_file = f"{index_file}.db"
        has_metadata = os.path.exists(metadata_file)
        has_database = os.path.exists(db_file)
        

        file_size = index_info.file_size
        metadata_size = os.path.getsize(metadata_file) if has_metadata else 0
        db_size = os.path.getsize(db_file) if has_database else 0
        total_size = file_size + metadata_size + db_size
        
        if has_database:
            total_docs, total_terms = self._get_db_stats(db_file)
        else:
            total_docs, total_terms = index_info.total_docs, index_info.total_terms
        
        return {
            'file_path': index_file,
            'exists': os.path.exists(index_file),
            'index_type': index_info.index_type.value,
            'recommended_indexer': recommended_indexer,
            'confidence': index_info.confidence,
            'total_docs': total_docs,
            'total_terms': total_terms,
            'version': index_info.version,
            'compression': index_info.compression,
            'file_size': file_size,
            'metadata_size': metadata_size,
            'db_size': db_size,
            'total_size': total_size,
            'has_metadata': has_metadata,
            'has_database': has_database,
            'created_at': index_info.created_at,
            'last_updated': index_info.last_updated
        }


_detector = None

def get_index_detector() -> IndexDetector:
    
    global _detector
    if _detector is None:
        _detector = IndexDetector()
    return _detector

def auto_detect_index_type(index_file: str) -> str:
    
    detector = get_index_detector()
    index_info = detector.detect_index_type(index_file)
    return detector.get_recommended_indexer(index_info)

def get_index_info(index_file: str) -> IndexInfo:
    
    detector = get_index_detector()
    return detector.detect_index_type(index_file) 