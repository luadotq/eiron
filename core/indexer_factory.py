


import os
import logging
from typing import Optional, Dict, Any
from .index_detector import auto_detect_index_type, get_index_info, IndexInfo

class IndexerFactory:
    
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_indexer(self, index_file: str = 'eiron_index.bin', 
                      force_type: Optional[str] = None,
                      **kwargs) -> Any:
        
        try:

            if force_type:
                self.logger.info(f"Using forced indexer type: {force_type}")
                return self._create_indexer_by_type(force_type, index_file, **kwargs)
            

            if os.path.exists(index_file):
                detected_type = auto_detect_index_type(index_file)
                index_info = get_index_info(index_file)
                
                self.logger.info(f"Auto-detected index type: {detected_type} "
                               f"(confidence: {index_info.confidence:.2f})")
                
                return self._create_indexer_by_type(detected_type, index_file, **kwargs)
            else:

                self.logger.info("Index file not found, using standard indexer")
                return self._create_indexer_by_type('standard', index_file, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error creating indexer: {e}")

            self.logger.info("Falling back to standard indexer")
            return self._create_indexer_by_type('standard', index_file, **kwargs)
    
    def _create_indexer_by_type(self, indexer_type: str, index_file: str, **kwargs) -> Any:
        
        try:
            if indexer_type == 'optimized':
                from .optimized_indexer import OptimizedIndexer
                

                default_params = {
                    'max_workers': min(32, (os.cpu_count() or 1) + 4),
                    'batch_size': 100
                }
                default_params.update(kwargs)
                
                indexer = OptimizedIndexer(
                    index_file=index_file,
                    **default_params
                )
                self.logger.info(f"Created optimized indexer: {index_file}")
                return indexer
                
            else:
                from .indexer import Indexer
                

                default_params = {
                    'hide_paths': False,
                    'dynamic_output': True
                }
                default_params.update(kwargs)
                
                indexer = Indexer(
                    index_file=index_file,
                    **default_params
                )
                self.logger.info(f"Created standard indexer: {index_file}")
                return indexer
                
        except ImportError as e:
            self.logger.error(f"Failed to import {indexer_type} indexer: {e}")

            if indexer_type != 'standard':
                self.logger.info("Falling back to standard indexer")
                return self._create_indexer_by_type('standard', index_file, **kwargs)
            else:
                raise
    
    def get_indexer_info(self, index_file: str) -> Dict[str, Any]:
        
        try:
            index_info = get_index_info(index_file)
            recommended_type = auto_detect_index_type(index_file)
            
            return {
                'file_path': index_file,
                'exists': os.path.exists(index_file),
                'detected_type': index_info.index_type.value,
                'recommended_indexer': recommended_type,
                'confidence': index_info.confidence,
                'total_docs': index_info.total_docs,
                'total_terms': index_info.total_terms,
                'version': index_info.version,
                'compression': index_info.compression,
                'file_size': index_info.file_size,
                'has_metadata': index_info.has_metadata,
                'created_at': index_info.created_at,
                'last_updated': index_info.last_updated
            }
        except Exception as e:
            self.logger.error(f"Error getting indexer info: {e}")
            return {
                'file_path': index_file,
                'exists': os.path.exists(index_file),
                'detected_type': 'unknown',
                'recommended_indexer': 'standard',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def validate_indexer_compatibility(self, index_file: str, indexer_type: str) -> Dict[str, Any]:
        
        try:
            index_info = get_index_info(index_file)
            recommended_type = auto_detect_index_type(index_file)
            
            is_compatible = (
                indexer_type == recommended_type or
                index_info.confidence < 0.5
            )
            
            return {
                'file_path': index_file,
                'requested_type': indexer_type,
                'recommended_type': recommended_type,
                'is_compatible': is_compatible,
                'confidence': index_info.confidence,
                'warning': None if is_compatible else f"Recommended type: {recommended_type}"
            }
            
        except Exception as e:
            return {
                'file_path': index_file,
                'requested_type': indexer_type,
                'is_compatible': False,
                'error': str(e)
            }


_factory = None

def get_indexer_factory() -> IndexerFactory:
    
    global _factory
    if _factory is None:
        _factory = IndexerFactory()
    return _factory

def create_auto_indexer(index_file: str = 'eiron_index.bin', 
                       force_type: Optional[str] = None,
                       **kwargs) -> Any:
    
    factory = get_indexer_factory()
    return factory.create_indexer(index_file, force_type, **kwargs)

def get_auto_indexer_info(index_file: str) -> Dict[str, Any]:
    
    factory = get_indexer_factory()
    return factory.get_indexer_info(index_file) 