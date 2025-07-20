import os
import textwrap
import sys
import json
import mmap
import platform
from typing import Generator, Optional, List, Dict, Set, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from .file_loader import load_text_from_file, stream_file_content
import logging
import re
from collections import defaultdict, OrderedDict
import unicodedata
import pickle
import zlib
from pathlib import Path
import csv
import io
import PyPDF2
import pandas as pd
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import threading
import time
import hashlib
import struct
import lz4.frame
import psutil
import gc
from dataclasses import dataclass, field
from contextlib import contextmanager


CHUNK_SIZE = 64 * 1024
MAX_MEMORY_USAGE = 0.8
BATCH_SIZE = 100
COMPRESSION_LEVEL = 1

@dataclass
class IndexMetadata:
    
    total_docs: int = 0
    total_terms: int = 0
    index_size: int = 0
    created_at: str = ""
    last_updated: str = ""
    version: str = "2.0"
    compression: str = "lz4"
    
@dataclass
class DocumentInfo:
    
    doc_id: str
    file_path: str
    file_size: int
    text_length: int
    term_count: int
    indexed_at: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class OptimizedTokenizer:
    
    

    STOP_WORDS = {

        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
        'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can',
        'just', 'should', 'now', 'then', 'here', 'there', 'up', 'down', 'out',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'her', 'him',
        'his', 'hers', 'itself', 'themselves', 'themself', 'ourselves', 'myself',
        'yourself', 'yourselves', 'himself', 'herself', 'ours', 'yours', 'mine',
        'yours', 'theirs', 'ours', 'yours', 'mine', 'theirs', 'ours', 'yours',
        

        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
        'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
        'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
        'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
        'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
        'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
        'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
        'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь',
        'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем',
        'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после',
        'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
        'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
        'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю'
    }
    
    def __init__(self):
        self._cache = {}
        self._cache_size = 1000
        self._lock = threading.Lock()
    
    def normalize_text(self, text: str) -> str:
        
        if not text:
            return ""
        

        text = unicodedata.normalize('NFKD', text.lower())
        

        text = re.sub(r'[^\w\s\u0400-\u04FF.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        

        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        with self._lock:
            if text_hash in self._cache:
                return self._cache[text_hash]
        

        normalized = self.normalize_text(text)
        

        words = normalized.split()
        

        filtered_words = [
            word for word in words 
            if word not in self.STOP_WORDS and len(word) > 2
        ]
        

        with self._lock:
            if len(self._cache) >= self._cache_size:

                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[text_hash] = filtered_words
        
        return filtered_words

class OptimizedMemoryManager:
    
    
    def __init__(self, max_memory_usage: float = MAX_MEMORY_USAGE):
        self.max_memory_usage = max_memory_usage
        self.process = psutil.Process()
        self._lock = threading.Lock()
        self._last_check = 0
        self._check_interval = 1.0
        
    def get_memory_usage(self) -> float:
        
        return self.process.memory_percent() / 100.0
    
    def get_system_memory_usage(self) -> float:
        
        return psutil.virtual_memory().percent / 100.0
    
    def should_optimize(self) -> bool:
        
        current_time = time.time()
        

        if current_time - self._last_check < self._check_interval:
            return False
        
        self._last_check = current_time
        
        with self._lock:
            memory_usage = self.get_memory_usage()
            system_memory = self.get_system_memory_usage()
            
            return memory_usage > self.max_memory_usage or system_memory > 0.9
    
    def optimize_memory(self):
        
        if not self.should_optimize():
            return
        
        logging.info("Memory optimization...")
        

        gc.collect()
        


        if hasattr(gc, 'garbage'):
            gc.garbage.clear()
        
        logging.info("Memory optimized")

class OptimizedStorage:
    
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.metadata_path = f"{index_path}.meta"
        self.db_path = f"{index_path}.db"
        self._lock = threading.Lock()
        
        if not os.path.exists(self.db_path):
            logging.warning(f"[OptimizedStorage] Database file '{self.db_path}' not found. Context search and some features may be inaccurate or unavailable.")

        self._init_database()
    
    def _init_database(self):
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, file_path TEXT, file_size INTEGER, text_length INTEGER, term_count INTEGER, indexed_at TEXT, checksum TEXT, metadata TEXT)")
            
            conn.execute("CREATE TABLE IF NOT EXISTS terms (term TEXT, doc_id TEXT, positions TEXT, count INTEGER)")
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_terms ON terms(term)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON terms(doc_id)")
            conn.commit()
    
    def save_index(self, index_data: Dict, metadata: IndexMetadata):
        
        with self._lock:

            compressed_data = lz4.frame.compress(
                pickle.dumps(index_data), 
                compression_level=COMPRESSION_LEVEL
            )
            

            with open(self.index_path, 'wb') as f:
                f.write(compressed_data)
            

            with open(self.metadata_path, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2)
            

            self._update_database(index_data)
    
    def load_index(self) -> Tuple[Dict, IndexMetadata]:
        logging.info(f"[DEBUG] load_index called for {self.index_path}")
        with self._lock:

            if os.path.exists(self.metadata_path):
                try:
                    with open(self.metadata_path, 'r') as f:
                        meta_dict = json.load(f)
                        metadata = IndexMetadata(**meta_dict)
                except Exception as e:
                    logging.warning(f"Error loading metadata: {e}, using default")
                    metadata = IndexMetadata()
            else:
                metadata = IndexMetadata()
            
            # First try to load from database
            db_data = self._load_from_database()
            if db_data:
                logging.info("Successfully loaded index from database")
                return db_data, metadata
            
            # If database is empty or contains no data, load from main file
            if os.path.exists(self.index_path):
                index_data = None
                
                file_size = os.path.getsize(self.index_path)
                
                if file_size == 0:
                    logging.warning("Index file is empty, creating new index")
                    index_data = self._create_empty_index()
                else:
                    index_data = self._try_load_index_file()
                
                if index_data is None:
                    logging.error("Failed to load index, creating new empty index")
                    index_data = self._create_empty_index()
            else:
                logging.info("Index file not found, creating new index")
                index_data = self._create_empty_index()
            
            return index_data, metadata
    
    def _create_empty_index(self) -> Dict:
        
        return {
            'index': {},
            'documents': {},
            'total_docs': 0,
            'total_terms': 0
        }
    
    def _try_load_index_file(self) -> Optional[Dict]:
        
        try:

            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            logging.info("Loaded standard pickle index")
            return index_data
        except Exception as e1:
            logging.debug(f"Failed to load as standard pickle: {e1}")
            
            try:

                with open(self.index_path, 'rb') as f:
                    data = f.read()
                index_data = pickle.loads(lz4.frame.decompress(data))
                logging.info("Loaded LZ4 compressed index")
                return index_data
            except Exception as e2:
                logging.debug(f"Failed to load as LZ4 compressed: {e2}")
                
                try:

                    with open(self.index_path, 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    logging.info("Loaded JSON index")
                    return index_data
                except Exception as e3:
                    logging.debug(f"Failed to load as JSON: {e3}")
                    

                    logging.error(f"All loading attempts failed. File may be corrupted.")
                    return None
    
    def _update_database(self, index_data: Dict):
        
        with sqlite3.connect(self.db_path) as conn:

            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM terms")
            

            for doc_id, doc_info in index_data['documents'].items():
                conn.execute("INSERT INTO documents (doc_id, file_path, file_size, text_length, term_count, indexed_at, checksum, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                    doc_id,
                    doc_info.get('file_path', ''),
                    doc_info.get('file_size', 0),
                    doc_info.get('text_length', 0),
                    doc_info.get('term_count', 0),
                    doc_info.get('indexed_at', ''),
                    doc_info.get('checksum', ''),
                    json.dumps(doc_info.get('metadata', {}))
                ))
            

            for term, doc_positions in index_data['index'].items():
                for doc_id, positions in doc_positions.items():
                    conn.execute("INSERT INTO terms (term, doc_id, positions, count) VALUES (?, ?, ?, ?)", (
                        term,
                        doc_id,
                        json.dumps(positions),
                        len(positions)
                    ))
            
            conn.commit()
    
    def _load_from_database(self) -> Optional[Dict]:
        logging.info(f"[DEBUG] Attempting to load from database: {self.db_path}")
        if not os.path.exists(self.db_path):
            logging.warning(f"[DEBUG] Database file does not exist: {self.db_path}")
            return None
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load documents
                documents = {}
                cursor = conn.execute("SELECT doc_id, file_path, file_size, text_length, term_count, indexed_at, checksum, metadata FROM documents")
                doc_rows = cursor.fetchall()
                logging.info(f"[DEBUG] Loaded {len(doc_rows)} document rows from DB")
                for row in doc_rows:
                    doc_id, file_path, file_size, text_length, term_count, indexed_at, checksum, metadata_json = row
                    documents[doc_id] = {
                        'file_path': file_path,
                        'file_size': file_size,
                        'text_length': text_length,
                        'term_count': term_count,
                        'indexed_at': indexed_at,
                        'checksum': checksum,
                        'metadata': json.loads(metadata_json) if metadata_json else {}
                    }
                
                # Load index
                index = {}
                cursor = conn.execute("SELECT term, doc_id, positions, count FROM terms")
                term_rows = cursor.fetchall()
                logging.info(f"[DEBUG] Loaded {len(term_rows)} term rows from DB")
                for row in term_rows:
                    term, doc_id, positions_json, count = row
                    if term not in index:
                        index[term] = {}
                    positions = json.loads(positions_json) if positions_json else []
                    index[term][doc_id] = positions
                
                if documents and index:
                    # If we have data in the DB, but no text in documents, 
                    # try to load text from main file
                    if os.path.exists(self.index_path):
                        try:
                            main_data = self._try_load_index_file()
                            if main_data and 'documents' in main_data:
                                # Combine data from DB with text from main file
                                for doc_id in documents:
                                    if doc_id in main_data['documents'] and 'text' in main_data['documents'][doc_id]:
                                        documents[doc_id]['text'] = main_data['documents'][doc_id]['text']
                                    else:
                                        documents[doc_id]['text'] = ""
                                logging.info("Combined database data with text from main file")
                            else:
                                for doc_id in documents:
                                    documents[doc_id]['text'] = ""
                                logging.warning("Could not load text from main file, using empty texts")
                        except Exception as e:
                            logging.warning(f"Error loading text from main file: {e}")
                            for doc_id in documents:
                                documents[doc_id]['text'] = ""
                    else:
                        for doc_id in documents:
                            documents[doc_id]['text'] = ""
                    
                    logging.info(f"Loaded {len(documents)} documents and {len(index)} terms from database")
                    return {
                        'index': index,
                        'documents': documents,
                        'total_docs': len(documents),
                        'total_terms': len(index)
                    }
                else:
                    logging.warning("Database exists but contains no data")
                    return None
                    
        except Exception as e:
            logging.error(f"Error loading from database: {e}")
            return None

class OptimizedIndexer:
    
    
    def __init__(self, index_file: str = 'eiron_index.bin', 
                 max_workers: int = None,
                 batch_size: int = BATCH_SIZE):
        self.index_file = index_file
        self.storage = OptimizedStorage(index_file)
        self.tokenizer = OptimizedTokenizer()
        self.memory_manager = OptimizedMemoryManager()
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.batch_size = batch_size
        
        # Preload index into memory
        self.index_data, self.metadata = self.storage.load_index()
        self._index_loaded = True
        
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            stop_words='english',
            ngram_range=(1, 2),
            analyzer='word',
            dtype=np.float64
        )
        
        if self.index_data['documents']:
            self._prepare_vectorizer()
    
    def _prepare_vectorizer(self):
        
        try:
            texts = [doc['text'] for doc in self.index_data['documents'].values()]
            if texts:
                self.vectorizer.fit_transform(texts)
        except Exception as e:
            logging.error(f"Error preparing vectorizer: {e}")
    
    def index_directory(self, directory: str, extensions: List[str] = None) -> int:
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        directory = os.path.normpath(directory)
        logging.info(f"Starting indexing: {directory}")
        

        files_to_process = self._collect_files(directory, extensions)
        
        if not files_to_process:
            logging.warning(f"Files not found in {directory}")
            return 0
        
        logging.info(f"Found {len(files_to_process)} files to process")
        

        processed_count = 0
        total_batches = (len(files_to_process) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(files_to_process))
            batch_files = files_to_process[start_idx:end_idx]
            
            logging.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_files)} files)")
            

            batch_results = self._process_batch(batch_files)
            

            for doc_id, doc_data in batch_results:
                self.index_data['documents'][doc_id] = doc_data
                processed_count += 1
            

            if self.memory_manager.should_optimize():
                self.memory_manager.optimize_memory()
            

            if batch_num % 5 == 0:
                self._save_progress()
        

        self._finalize_indexing(processed_count)
        
        return processed_count
    
    def _collect_files(self, directory: str, extensions: List[str] = None) -> List[str]:
        
        files = []
        supported_extensions = extensions or ['.txt', '.md', '.docx', '.pdf', '.csv', '.py', '.js', '.html']
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if any(filename.lower().endswith(ext.lower()) for ext in supported_extensions):
                    files.append(file_path)
        
        return files
    
    def _process_batch(self, files: List[str]) -> List[Tuple[str, Dict]]:
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_single_file, file_path): file_path 
                      for file_path in files}
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    file_path = futures[future]
                    logging.error(f"Error processing file {file_path}: {e}")
        
        return results
    
    def _process_single_file(self, file_path: str) -> Optional[Tuple[str, Dict]]:
        
        try:

            if not os.access(file_path, os.R_OK):
                logging.warning(f"File not accessible: {file_path}")
                return None
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.warning(f"Empty file: {file_path}")
                return None
            

            text = self._read_file_content(file_path)
            if not text:
                return None
            

            tokens = self.tokenizer.tokenize(text)
            if not tokens:
                logging.warning(f"No tokens in file: {file_path}")
                return None
            

            doc_id = self._generate_doc_id(file_path)
            doc_data = self._create_document_data(file_path, text, tokens, file_size)
            

            self._add_to_index(doc_id, tokens)
            
            logging.info(f"Successfully processed file: {file_path} ({len(tokens)} tokens)")
            return doc_id, doc_data
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return None
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdf':
                return self._read_pdf_file(file_path)
            elif ext == '.csv':
                return self._read_csv_file(file_path)
            else:
                return self._read_text_file(file_path)
                
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _read_text_file(self, file_path: str) -> Optional[str]:
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except UnicodeDecodeError:

            try:
                with open(file_path, 'r', encoding='latin1', errors='ignore') as f:
                    return f.read()
            except Exception:
                return None
    
    def _read_pdf_file(self, file_path: str) -> Optional[str]:
        
        try:
            text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            logging.error(f"Error reading PDF {file_path}: {e}")
            return None
    
    def _read_csv_file(self, file_path: str) -> Optional[str]:
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            return df.fillna('').to_string(index=False)
        except Exception:
            try:
                df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
                return df.fillna('').to_string(index=False)
            except Exception as e:
                logging.error(f"Error reading CSV {file_path}: {e}")
                return None
    
    def _generate_doc_id(self, file_path: str) -> str:
        
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _create_document_data(self, file_path: str, text: str, tokens: List[str], file_size: int) -> Dict:
        
        return {
            'file_path': file_path,
            'text': text,
            'file_size': file_size,
            'text_length': len(text),
            'term_count': len(tokens),
            'indexed_at': datetime.now().isoformat(),
            'checksum': hashlib.md5(text.encode()).hexdigest(),
            'metadata': {
                'extension': os.path.splitext(file_path)[1],
                'filename': os.path.basename(file_path)
            }
        }
    
    def _add_to_index(self, doc_id: str, tokens: List[str]):
        
        for position, token in enumerate(tokens):
            if token not in self.index_data['index']:
                self.index_data['index'][token] = {}
            
            if doc_id not in self.index_data['index'][token]:
                self.index_data['index'][token][doc_id] = []
            
            self.index_data['index'][token][doc_id].append(position)
    
    def _save_progress(self):
        
        try:
            self.metadata.total_docs = len(self.index_data['documents'])
            self.metadata.total_terms = len(self.index_data['index'])
            self.metadata.last_updated = datetime.now().isoformat()
            
            self.storage.save_index(self.index_data, self.metadata)
            logging.info(f"Progress saved: {self.metadata.total_docs} documents")
        except Exception as e:
            logging.error(f"Error saving progress: {e}")
    
    def _finalize_indexing(self, processed_count: int):
        
        try:

            self.metadata.total_docs = len(self.index_data['documents'])
            self.metadata.total_terms = len(self.index_data['index'])
            self.metadata.last_updated = datetime.now().isoformat()
            self.metadata.index_size = os.path.getsize(self.index_file) if os.path.exists(self.index_file) else 0
            

            self.storage.save_index(self.index_data, self.metadata)
            

            self._prepare_vectorizer()
            
            logging.info(f"Indexing completed: {processed_count} files processed")
            logging.info(f"Statistics: {self.metadata.total_docs} documents, {self.metadata.total_terms} terms")
            
        except Exception as e:
            logging.error(f"Error finalizing indexing: {e}")
    
    def search(self, query: str, search_mode: str = 'hybrid', 
               max_results: int = 20, min_similarity: float = 0.1) -> List[Tuple[str, float, Optional[str]]]:
        
        try:
            query_tokens = self.tokenizer.tokenize(query)
            if not query_tokens:
                return []
            
            results = []
            
            if search_mode in ['keyword', 'hybrid']:

                keyword_results = self._keyword_search(query_tokens)
                results.extend(keyword_results)
            
            if search_mode in ['semantic', 'hybrid']:

                semantic_results = self._semantic_search(query, max_results)
                results.extend(semantic_results)
            

            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(unique_results, key=lambda x: x[1], reverse=True)
            

            return sorted_results[:max_results]
            
        except Exception as e:
            logging.error(f"Error searching: {e}")
            return []
    
    def _keyword_search(self, query_tokens: List[str]) -> List[Tuple[str, float, Optional[str]]]:
        
        results = []
        
        for token in query_tokens:
            if token in self.index_data['index']:
                for doc_id, positions in self.index_data['index'][token].items():
                    # Check if document exists
                    if doc_id not in self.index_data['documents']:
                        continue
                        
                    score = self._calculate_relevance_score(token, positions, doc_id)
                    
                    # Get context if there is text
                    context = None
                    doc_text = self.index_data['documents'][doc_id].get('text', '')
                    if doc_text:
                        # Search for token position in text for context
                        try:
                            token_pos = doc_text.lower().find(token.lower())
                            if token_pos != -1:
                                start = max(0, token_pos - 100)
                                end = min(len(doc_text), token_pos + len(token) + 100)
                                context = doc_text[start:end]
                                if start > 0:
                                    context = "..." + context
                                if end < len(doc_text):
                                    context = context + "..."
                        except Exception:
                            context = None
                    
                    results.append((doc_id, score, context))
        
        return results
    
    def _semantic_search(self, query: str, max_results: int) -> List[Tuple[str, float, Optional[str]]]:
        
        try:
            if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
                return []
            
            valid_docs = []
            valid_doc_ids = []
            for doc_id, doc_data in self.index_data['documents'].items():
                doc_text = doc_data.get('text', '')
                if doc_text and doc_text.strip():
                    valid_docs.append(doc_text)
                    valid_doc_ids.append(doc_id)
            
            if not valid_docs:
                return []
            
            query_vector = self.vectorizer.transform([query])
            doc_vectors = self.vectorizer.transform(valid_docs)
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            results = []
            for i, similarity in enumerate(similarities):
                if similarity > 0.1:
                    doc_id = valid_doc_ids[i]
                    doc_text = self.index_data['documents'][doc_id].get('text', '')
                    
                    # Create context
                    context = None
                    if doc_text:
                        try:
                            # Search for query position in text
                            query_pos = doc_text.lower().find(query.lower())
                            if query_pos != -1:
                                start = max(0, query_pos - 100)
                                end = min(len(doc_text), query_pos + len(query) + 100)
                                context = doc_text[start:end]
                                if start > 0:
                                    context = "..." + context
                                if end < len(doc_text):
                                    context = context + "..."
                        except Exception:
                            context = None
                    
                    results.append((doc_id, float(similarity), context))
            
            return results
            
        except Exception as e:
            logging.error(f"Error semantic search: {e}")
            return []
    
    def _calculate_relevance_score(self, token: str, positions: List[int], doc_id: str) -> float:
        
        frequency = len(positions)
        
        position_bonus = sum(1.0 / (pos + 1) for pos in positions[:10])
        
        # Safe document text retrieval
        doc_text = self.index_data['documents'].get(doc_id, {}).get('text', '')
        doc_length = len(doc_text) if doc_text else 1
        length_penalty = 1.0 / (1.0 + doc_length / 10000)
        
        return (frequency + position_bonus) * length_penalty
    
    def _deduplicate_results(self, results: List[Tuple[str, float, Optional[str]]]) -> List[Tuple[str, float, Optional[str]]]:
        
        seen = set()
        unique_results = []
        
        for doc_id, score, context in results:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc_id, score, context))
        
        return unique_results
    
    def get_statistics(self) -> Dict[str, Any]:
        
        return {
            'total_documents': self.metadata.total_docs,
            'total_terms': self.metadata.total_terms,
            'index_size_mb': self.metadata.index_size / (1024 * 1024),
            'created_at': self.metadata.created_at,
            'last_updated': self.metadata.last_updated,
            'version': self.metadata.version,
            'compression': self.metadata.compression
        }
    
    def save(self, filename: str = None):
        
        try:

            self.metadata.total_docs = len(self.index_data['documents'])
            self.metadata.total_terms = len(self.index_data['index'])
            self.metadata.last_updated = datetime.now().isoformat()
            

            save_file = filename or self.index_file
            

            temp_storage = OptimizedStorage(save_file)
            temp_storage.save_index(self.index_data, self.metadata)
            
            logging.info(f"Index saved to: {save_file}")
            
        except Exception as e:
            logging.error(f"Error saving index: {e}")
            raise
    
    def load(self, filename: str = None):
        # Now reloading the index is prohibited if it is already in memory
        if getattr(self, '_index_loaded', False):
            logging.warning("[OptimizedIndexer] Index already loaded in memory, skipping reload.")
            return
        try:
            load_file = filename or self.index_file
            temp_storage = OptimizedStorage(load_file)
            self.index_data, self.metadata = temp_storage.load_index()
            self._index_loaded = True
            if self.index_data['documents']:
                self._prepare_vectorizer()
            logging.info(f"Index loaded from: {load_file}")
        except Exception as e:
            logging.error(f"Error loading index: {e}")
            raise 