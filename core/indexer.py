import os
import textwrap
import sys
import json
import mmap
import platform
from typing import Generator, Optional, List, Dict, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .file_loader import load_text_from_file, stream_file_content
import logging
import re
from collections import defaultdict
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



SUPPORTED_EXTENSIONS = ['.txt', '.md', '.docx', '.pdf', '.csv', '.xlsx', '.html', '.htm', '.xml', '.json', '.py', '.js', '.css']

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.replace('\0', '')
    
    return text.strip()

class CustomTokenizer:
    
    ENGLISH_STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
        'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can', 'will',
        'just', 'should', 'now'
    }
    
    RUSSIAN_STOP_WORDS = {
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
    
    STOP_WORDS = ENGLISH_STOP_WORDS | RUSSIAN_STOP_WORDS
    
    @staticmethod
    def normalize_text(text: str) -> str:
        

        text = text.lower()
        

        text = unicodedata.normalize('NFKD', text)
        

        text = re.sub(r'[^\w\s\u0400-\u04FF.,!?-]', ' ', text)
        

        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def is_russian_word(word: str) -> bool:
        
        return any('\u0400' <= char <= '\u04FF' for char in word)
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        

        text = CustomTokenizer.normalize_text(text)
        

        words = text.split()
        



        words = [word for word in words 
                if word not in CustomTokenizer.STOP_WORDS 
                and (len(word) > 2 or (len(word) > 1 and CustomTokenizer.is_russian_word(word)))]
        
        return words

class ContextEngine:
    
    
    def __init__(self, context_size: int = 100):
        self.context_size = context_size
    
    def extract_context(self, text: str, position: int) -> str:
        
        start = max(0, position - self.context_size)
        end = min(len(text), position + self.context_size)
        

        if start > 0:

            for i in range(start, max(0, start - 50), -1):
                if text[i] in '.!?':
                    start = i + 1
                    break
        
        if end < len(text):

            for i in range(end, min(len(text), end + 50)):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        context = text[start:end].strip()
        

        if start > 0:
            context = '...' + context
        if end < len(text):
            context = context + '...'
            
        return context

class Vectorizer:
    
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            analyzer='word'
        )
        self.document_vectors = None
        self.document_texts = []
    
    def fit_transform(self, texts: List[str]):
        
        self.document_texts = texts
        self.document_vectors = self.vectorizer.fit_transform(texts)
    
    def transform_query(self, query: str) -> np.ndarray:
        
        return self.vectorizer.transform([query])
    
    def find_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.documents: Dict[str, Dict] = {}
        self.total_docs = 0
        self.total_terms = 0
        self.tokenizer = CustomTokenizer()
        self.context_engine = ContextEngine()
        self.vectorizer = Vectorizer()
        self.symbol_counts = defaultdict(int)

    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        
        if not text:
            logging.warning(f"Empty text provided for document {doc_id}")
            return

        logging.info(f"Adding document {doc_id} to index")
        

        words = self.tokenizer.tokenize(text)
        logging.info(f"Tokenized {len(words)} words from document {doc_id}")
        
        if not words:
            logging.warning(f"No valid words found in document {doc_id}")
            return


        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata or {},
            'word_count': len(words),
            'symbol_count': len(text)
        }
        

        for char in text:
            self.symbol_counts[char] += 1
        

        for position, word in enumerate(words):
            self.index[word][doc_id].append(position)
            self.total_terms += 1
        
        self.total_docs += 1
        logging.info(f"Successfully indexed document {doc_id} with {len(words)} words")

    def get_document_text(self, doc_id: str) -> Optional[str]:
        
        return self.documents.get(doc_id, {}).get('text')

    def get_context(self, doc_id: str, position: int) -> Optional[str]:
        
        text = self.get_document_text(doc_id)
        if text:
            return self.context_engine.extract_context(text, position)
        return None

    def search(self, query: str, use_semantic: bool = False, min_similarity: float = 0.1) -> List[Tuple[str, float, Optional[str]]]:
        
        if not query:
            logging.warning("Empty query provided")
            return []

        logging.info(f"Searching for query: {query}")
        logging.info(f"Total documents in index: {self.total_docs}")
        logging.info(f"Total terms in index: {self.total_terms}")


        query_words = self.tokenizer.tokenize(query)
        logging.info(f"Tokenized query words: {query_words}")

        if not query_words:
            logging.warning("No valid words found in query after tokenization")
            return []

        results = []
        
        if use_semantic:

            if not self.vectorizer.document_vectors is None:
                query_vector = self.vectorizer.transform_query(query)
                similar_docs = self.vectorizer.find_similar(query_vector, top_k=10)
                
                for doc_idx, score in similar_docs:
                    doc_id = list(self.documents.keys())[doc_idx]
                    text = self.get_document_text(doc_id)
                    if text:

                        context = None
                        for word in query_words:
                            pos = text.lower().find(word.lower())
                            if pos != -1:
                                context = self.get_context(doc_id, pos)
                                break
                        results.append((doc_id, score, context))
        else:

            scores = defaultdict(float)
            matched_docs = set()
            doc_metadata = {}

            for word in query_words:
                if word in self.index:
                    logging.info(f"Found word '{word}' in {len(self.index[word])} documents")

                    doc_freq = len(self.index[word])
                    idf = math.log((self.total_docs + 1) / (doc_freq + 1)) + 1
                    
                    for doc_id, positions in self.index[word].items():
                        text = self.documents[doc_id]['text']
                        word_count = self.documents[doc_id]['word_count']
                        

                        tf = len(positions) / word_count
                        

                        if word.lower() in text.lower():
                            tf *= 1.5
                        

                        if word.lower() in text[:200].lower():
                            tf *= 1.3
                        

                        proximity_bonus = self._calculate_proximity_bonus(positions, word_count)
                        

                        position_bonus = self._calculate_position_bonus(positions, word_count)
                        

                        phrase_bonus = self._calculate_phrase_bonus(text, query_words)
                        

                        base_score = tf * idf
                        enhanced_score = base_score * (1 + proximity_bonus + position_bonus + phrase_bonus)
                        
                        scores[doc_id] += enhanced_score
                        matched_docs.add(doc_id)
                        

                        if doc_id not in doc_metadata:
                            doc_metadata[doc_id] = {
                                'best_positions': positions,
                                'text': text
                            }


            query_length = len(query_words)
            

            for doc_id in matched_docs:
                normalized_score = scores[doc_id] / query_length
                

                context = self._get_optimal_context(doc_id, query_words, doc_metadata.get(doc_id, {}))
                
                results.append((doc_id, normalized_score, context))


        results.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"Returning {len(results)} results")
        return results
    
    def _calculate_proximity_bonus(self, positions: List[int], word_count: int) -> float:
        
        if len(positions) < 2:
            return 0.0
        

        distances = []
        for i in range(len(positions) - 1):
            distance = positions[i + 1] - positions[i]
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances)
        

        if avg_distance <= 5:
            return 0.5
        elif avg_distance <= 20:
            return 0.3
        elif avg_distance <= 50:
            return 0.1
        else:
            return 0.0
    
    def _calculate_position_bonus(self, positions: List[int], word_count: int) -> float:
        
        if not positions:
            return 0.0
        

        avg_position = sum(positions) / len(positions)
        position_percentage = avg_position / word_count
        

        if position_percentage <= 0.2:
            return 0.3
        elif position_percentage <= 0.4:
            return 0.2
        elif position_percentage <= 0.6:
            return 0.1
        else:
            return 0.0
    
    def _calculate_phrase_bonus(self, text: str, query_words: List[str]) -> float:
        
        if len(query_words) < 2:
            return 0.0
        
        text_lower = text.lower()
        

        phrase = ' '.join(query_words)
        if phrase in text_lower:
            return 0.4
        

        consecutive_bonus = 0.0
        for i in range(len(query_words) - 1):
            two_word_phrase = f"{query_words[i]} {query_words[i+1]}"
            if two_word_phrase in text_lower:
                consecutive_bonus += 0.1
        
        return min(0.3, consecutive_bonus)
    
    def _get_optimal_context(self, doc_id: str, query_words: List[str], metadata: Dict) -> Optional[str]:
        
        text = self.get_document_text(doc_id)
        if not text:
            return None
        

        best_position = None
        best_score = 0
        
        for word in query_words:
            positions = self.index.get(word, {}).get(doc_id, [])
            for pos in positions:

                position_score = 1.0 - (pos / len(text))
                

                nearby_bonus = 0
                for other_word in query_words:
                    if other_word != word:
                        other_positions = self.index.get(other_word, {}).get(doc_id, [])
                        for other_pos in other_positions:
                            if abs(other_pos - pos) <= 50:
                                nearby_bonus += 0.2
                
                total_score = position_score + nearby_bonus
                if total_score > best_score:
                    best_score = total_score
                    best_position = pos
        
        if best_position is not None:
            return self.get_context(doc_id, best_position)
        

        for word in query_words:
            pos = text.lower().find(word.lower())
            if pos != -1:
                return self.get_context(doc_id, pos)
        
        return None

    def get_symbol_stats(self) -> Dict[str, int]:
        
        return dict(self.symbol_counts)

    def save(self, filename: str):
        
        try:

            index_dict = {}
            for term, doc_positions in self.index.items():
                index_dict[term] = dict(doc_positions)
            
            data = {
                'index': index_dict,
                'documents': self.documents,
                'total_docs': self.total_docs,
                'total_terms': self.total_terms,
                'symbol_counts': dict(self.symbol_counts)
            }
            

            try:
                import lz4.frame
                compressed_data = lz4.frame.compress(pickle.dumps(data))
                with open(filename, 'wb') as f:
                    f.write(compressed_data)
                logging.info(f"Index saved to {filename} with LZ4 compression")
            except ImportError:

                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
                logging.info(f"Index saved to {filename} with standard pickle")
            
        except Exception as e:
            logging.error(f"Error saving index: {e}")
            raise

    def load(self, filename: str):
        
        try:
            if not os.path.exists(filename):
                logging.warning(f"Index file not found: {filename}")
                return
                
            if os.path.getsize(filename) == 0:
                logging.warning(f"Index file is empty: {filename}")
                return


            data = self._try_load_index_data(filename)
            
            if data is None:
                logging.error("Failed to load index data from file")
                return
            

            required_keys = {'index', 'documents', 'total_docs', 'total_terms'}
            if not all(key in data for key in required_keys):
                logging.error("Invalid index file format: missing required keys")
                return
                

            new_index = defaultdict(lambda: defaultdict(list))
            for term, doc_positions in data['index'].items():
                for doc_id, positions in doc_positions.items():
                    new_index[term][doc_id] = positions
                    
            new_documents = data['documents']
            new_total_docs = data['total_docs']
            new_total_terms = data['total_terms']
            new_symbol_counts = defaultdict(int, data.get('symbol_counts', {}))
            

            doc_count_diff = abs(new_total_docs - len(new_documents))
            if doc_count_diff > 5:
                logging.warning(f"Index integrity warning: total_docs ({new_total_docs}) != documents count ({len(new_documents)}), difference: {doc_count_diff}")

                new_total_docs = len(new_documents)
            elif doc_count_diff > 0:
                logging.info(f"Index integrity: minor difference detected, using actual document count")
                new_total_docs = len(new_documents)
                

            self.index = new_index
            self.documents = new_documents
            self.total_docs = new_total_docs
            self.total_terms = new_total_terms
            self.symbol_counts = new_symbol_counts
            

            try:
                texts = [doc['text'] for doc in self.documents.values()]
                if texts:
                    self.vectorizer.fit_transform(texts)
            except Exception as e:
                logging.error(f"Error preparing vectorizer: {e}")

            
            logging.info(f"Loaded index with {self.total_docs} documents and {self.total_terms} terms")
            

            unique_terms = len(self.index)
            logging.info(f"Index contains {unique_terms} unique terms")
            

            sample_terms = list(self.index.keys())[:5]
            logging.info(f"Sample terms in index: {sample_terms}")
            
        except Exception as e:
            logging.error(f"Error loading index: {e}")


    def _try_load_index_data(self, filename: str):
        
        try:

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            logging.info("Loaded index as standard pickle")
            return data
        except Exception as e1:
            logging.debug(f"Failed to load as standard pickle: {e1}")
            
            try:

                import lz4.frame
                with open(filename, 'rb') as f:
                    compressed_data = f.read()
                data = pickle.loads(lz4.frame.decompress(compressed_data))
                logging.info("Loaded index as LZ4 compressed pickle")
                return data
            except ImportError:
                logging.warning("LZ4 not available, skipping LZ4 decompression")
            except Exception as e2:
                logging.debug(f"Failed to load as LZ4 compressed: {e2}")
            
            try:

                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logging.info("Loaded index as JSON")
                return data
            except Exception as e3:
                logging.debug(f"Failed to load as JSON: {e3}")
            

            logging.error("All loading attempts failed. File may be corrupted.")
            return None

class FileHandler:
    @staticmethod
    def read_text_file(path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
        
        try:
            if os.path.getsize(path) > 10 * 1024 * 1024:
                chunks = []
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    while chunk := f.read(chunk_size):
                        chunks.append(chunk)
                return ''.join(chunks)
            else:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logging.error(f"Error reading text file {path}: {e}")
            return None

    @staticmethod
    def read_pdf_file(path: str, chunk_size: int = None) -> Optional[str]:
        
        try:
            text = []
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            logging.error(f"Error reading PDF file {path}: {e}")
            return None

    @staticmethod
    def read_csv_file(path: str, chunk_size: int = None) -> Optional[str]:
        
        try:

            encodings = ['utf-8', 'latin1', 'cp1252']
            for encoding in encodings:
                try:


                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        on_bad_lines='skip',
                        engine='python',
                        sep=None,
                        quoting=csv.QUOTE_MINIMAL
                    )
                    


                    return df.fillna('').to_string(index=False)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"Error reading CSV file {path} with encoding {encoding}: {e}")
                    continue
            

            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Error reading CSV file {path} with basic approach: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Error reading CSV file {path}: {e}")
            return None

class MemoryManager:
    
    def __init__(self):
        from .optimized_memory_manager import OptimizedMemoryManager
        self._optimized_manager = OptimizedMemoryManager()
    
    def get_detailed_memory_stats(self):
        return self._optimized_manager.get_detailed_stats()
    
    def set_memory_threshold(self, threshold):
        pass
    
    def set_emergency_threshold(self, threshold):
        pass
    
    def set_chunk_size(self, size):
        pass
    
    def emergency_clear(self):
        self._optimized_manager.emergency_clear()

class Indexer:
    def __init__(self, index_file: str = 'eiron_index.bin', hide_paths: bool = False, 
                 dynamic_output: bool = True, resource_limits: Optional[Dict] = None):
        self.index_file = index_file
        self.index = InvertedIndex()
        self.memory_manager = MemoryManager()
        self.file_handler = FileHandler()
        self.hide_paths = hide_paths
        self.dynamic_output = dynamic_output
        self.resource_limits = resource_limits or {}
        

        self.resource_manager = None
        if resource_limits:
            from .resource_manager import ResourceManager, ResourceLimits
            limits = ResourceLimits(**resource_limits)
            self.resource_manager = ResourceManager(limits)
        

        if os.path.exists(index_file):
            self.load(index_file)

    def index_directory(self, directory: str, extensions: List[str] = None) -> int:
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")


        directory = os.path.normpath(directory)
        logging.info(f"Indexing directory: {directory}")


        files_to_process = []
        for root, _, files in os.walk(directory):
            for file in files:
                if extensions is None or any(file.lower().endswith(ext.lower()) for ext in extensions):
                    files_to_process.append(os.path.join(root, file))

        logging.info(f"Found {len(files_to_process)} files to process")

        if not files_to_process:
            logging.warning(f"No files found in {directory} with specified extensions")
            return 0


        with ThreadPoolExecutor() as executor:
            futures = []
            for file_path in files_to_process:
                future = executor.submit(self._process_file, file_path)
                futures.append(future)


            processed_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                        if processed_count % 10 == 0:
                            self.save(self.index_file)
                except Exception as e:
                    logging.error(f"Error processing file: {e}")


        self.save(self.index_file)
        logging.info(f"Indexed {processed_count} of {len(files_to_process)} files")
        return processed_count

    def _process_file(self, file_path: str) -> bool:
        
        try:

            if self.resource_manager and not self.resource_manager.is_within_limits():
                logging.warning("Resource limits exceeded, skipping file processing")
                return False
            

            if not os.access(file_path, os.R_OK):
                logging.warning(f"File not readable: {file_path}")
                return False
                
            if os.path.getsize(file_path) == 0:
                logging.warning(f"Empty file: {file_path}")
                return False


            ext = os.path.splitext(file_path)[1].lower()
            

            text = None
            try:
                if ext == '.pdf':
                    text = self.file_handler.read_pdf_file(file_path)
                elif ext == '.csv':
                    text = self.file_handler.read_csv_file(file_path)
                else:
                    text = self.file_handler.read_text_file(file_path)
            except UnicodeDecodeError:
                logging.warning(f"Could not decode file: {file_path}")
                return False
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                return False

            if not text or not text.strip():
                logging.warning(f"No content extracted from file: {file_path}")
                return False


            text = clean_text(text)
            if not text:
                logging.warning(f"No valid text after cleaning: {file_path}")
                return False


            doc_id = self._get_document_id(file_path)
            

            self.index.add_document(doc_id, text, metadata={'original_path': file_path})
            logging.info(f"Successfully processed file: {file_path}")
            return True

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return False
    
    def _get_document_id(self, file_path: str) -> str:
        
        if self.hide_paths:

            filename = os.path.basename(file_path)

            import hashlib
            path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            return f"{filename}_{path_hash}"
        else:
            return file_path

    def save(self, filename: str):
        
        try:
            if not self.index.documents:
                logging.warning("No documents to save in index")
                return
                
            self.index.save(filename)
            logging.info(f"Saved index with {self.index.total_docs} documents")
        except Exception as e:
            logging.error(f"Error saving index: {e}")
            raise

    def load(self, filename: str):
        
        try:
            if not os.path.exists(filename):
                logging.warning(f"Index file not found: {filename}")
                return
                
            if os.path.getsize(filename) == 0:
                logging.warning(f"Index file is empty: {filename}")
                return
                
            self.index.load(filename)
            

            if self.hide_paths and self.index.documents:
                self._apply_hide_paths_to_existing_documents()
            
            if self.index.total_docs == 0:
                logging.warning("Loaded index contains no documents")
            else:
                logging.info(f"Loaded index with {self.index.total_docs} documents")
                
        except Exception as e:
            logging.error(f"Error loading index: {e}")


    def _apply_hide_paths_to_existing_documents(self):
        
        try:
            new_documents = {}
            new_index = defaultdict(lambda: defaultdict(list))
            
            for doc_id, doc_data in self.index.documents.items():

                if '_' in doc_id and len(doc_id.split('_')) == 2:

                    new_documents[doc_id] = doc_data
                    if doc_id in self.index.index:
                        new_index[doc_id] = self.index.index[doc_id]
                else:

                    new_doc_id = self._get_document_id(doc_id)
                    new_documents[new_doc_id] = doc_data
                    

                    if doc_id in self.index.index:
                        new_index[new_doc_id] = self.index.index[doc_id]
            

            self.index.documents = new_documents
            self.index.index = new_index
            self.index.total_docs = len(new_documents)
            
            logging.info(f"Applied hide_paths to {len(new_documents)} documents")
            
        except Exception as e:
            logging.error(f"Error applying hide_paths to existing documents: {e}")

    def get_indexed_documents(self) -> Generator[Tuple[str, str], None, None]:
        
        for doc_id, doc in self.index.documents.items():
            yield doc_id, doc['text']

    def search(self, query: str, use_semantic: bool = False, min_similarity: float = 0.1, 
               max_results: int = 10, dynamic_context: bool = True, 
               search_mode: str = 'hybrid') -> List[Tuple[str, float, Optional[str]]]:
        
        try:

            from .context_search import ContextSearchEngine
            
            search_engine = ContextSearchEngine(self)
            search_results = search_engine.search(query, max_results, search_mode)
            

            results = []
            for result in search_results:
                results.append((result.doc_id, result.relevance_score, result.context))
            
            return results
            
        except ImportError:

            logging.warning("Context search engine not available, using fallback search")
            results = self.index.search(query, use_semantic=use_semantic, min_similarity=min_similarity)
            results = results[:max_results]
            
            if dynamic_context and self.dynamic_output:
                results = self._apply_dynamic_context(results, query)
            
            return results
    
    def _apply_dynamic_context(self, results: List[Tuple[str, float, Optional[str]]], 
                              query: str) -> List[Tuple[str, float, Optional[str]]]:
        
        dynamic_results = []
        
        for doc_id, score, context in results:

            if not context:
                text = self.index.get_document_text(doc_id)
                if text:

                    context = self._find_optimal_context_position(text, query)
                else:
                    dynamic_results.append((doc_id, score, context))
                    continue
            
            if not context:
                dynamic_results.append((doc_id, score, context))
                continue
            


            base_length = 150
            max_length = 600
            dynamic_length = int(base_length + (score * (max_length - base_length)))
            

            importance_factor = self._calculate_content_importance(context, query)
            final_length = int(dynamic_length * importance_factor)
            

            final_length = max(80, min(final_length, 1000))
            

            if len(context) > final_length:
                truncated_context = self._truncate_at_optimal_boundary(context, final_length, query)
                dynamic_results.append((doc_id, score, truncated_context))
            else:
                dynamic_results.append((doc_id, score, context))
        
        return dynamic_results
    
    def _find_optimal_context_position(self, text: str, query: str) -> Optional[str]:
        
        if not text or not query:
            return None
        
        query_terms = query.lower().split()
        text_lower = text.lower()
        

        term_positions = {}
        for term in query_terms:
            positions = []
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            term_positions[term] = positions
        

        best_position = None
        best_score = 0
        
        for term, positions in term_positions.items():
            for pos in positions:

                nearby_terms = 0
                for other_term, other_positions in term_positions.items():
                    if other_term != term:
                        for other_pos in other_positions:
                            if abs(other_pos - pos) <= 100:
                                nearby_terms += 1
                

                position_score = 1.0 - (pos / len(text))
                total_score = nearby_terms * 0.3 + position_score * 0.7
                
                if total_score > best_score:
                    best_score = total_score
                    best_position = pos
        
        if best_position is not None:

            start = max(0, best_position - 100)
            end = min(len(text), best_position + 100)
            context = text[start:end]
            

            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            
            return context
        

        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                start = max(0, pos - 100)
                end = min(len(text), pos + 100)
                context = text[start:end]
                
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                
                return context
        
        return None
    
    def _truncate_at_optimal_boundary(self, text: str, max_length: int, query: str) -> str:
        
        if len(text) <= max_length:
            return text
        

        truncated = text[:max_length]
        

        for i in range(len(truncated) - 1, max(0, len(truncated) - 100), -1):
            if truncated[i] in '.!?':

                if self._is_good_break_point(truncated[:i+1], query):
                    return truncated[:i+1] + "..."
        

        for i in range(len(truncated) - 1, max(0, len(truncated) - 50), -1):
            if truncated[i] == '\n' and truncated[i-1] == '\n':
                if self._is_good_break_point(truncated[:i], query):
                    return truncated[:i] + "..."
        

        for i in range(len(truncated) - 1, max(0, len(truncated) - 30), -1):
            if truncated[i].isspace():
                if self._is_good_break_point(truncated[:i], query):
                    return truncated[:i] + "..."
        

        return truncated + "..."
    
    def _is_good_break_point(self, text: str, query: str) -> bool:
        
        if not text or not query:
            return True
        
        query_terms = query.lower().split()
        text_lower = text.lower()
        

        for term in query_terms:
            if term in text_lower:
                return True
        
        return False
    
    def _calculate_content_importance(self, text: str, query: str) -> float:
        
        if not text or not query:
            return 1.0
        

        text_lower = text.lower()
        query_lower = query.lower()
        

        query_terms = query_lower.split()
        term_matches = sum(1 for term in query_terms if term in text_lower)
        term_density = term_matches / len(query_terms) if query_terms else 0
        

        words = text.split()
        sentences = text.split('.')
        complexity_factor = min(2.0, len(words) / 50.0)
        

        proximity_factor = self._calculate_proximity_factor(text_lower, query_terms)
        

        importance = (term_density * 0.4 + 
                     complexity_factor * 0.3 + 
                     proximity_factor * 0.3)
        
        return max(0.5, min(2.0, importance))
    
    def _calculate_proximity_factor(self, text: str, query_terms: List[str]) -> float:
        
        if len(query_terms) < 2:
            return 1.0
        

        term_positions = {}
        for term in query_terms:
            positions = []
            start = 0
            while True:
                pos = text.find(term, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            term_positions[term] = positions
        

        total_distance = 0
        count = 0
        
        for i, term1 in enumerate(query_terms):
            for term2 in query_terms[i+1:]:
                for pos1 in term_positions.get(term1, []):
                    for pos2 in term_positions.get(term2, []):
                        distance = abs(pos2 - pos1)
                        total_distance += distance
                        count += 1
        
        if count == 0:
            return 1.0
        
        avg_distance = total_distance / count

        proximity_factor = max(0.5, min(2.0, 100.0 / (avg_distance + 10)))
        
        return proximity_factor
    
    def _truncate_at_sentence_boundary(self, text: str, max_length: int) -> str:
        
        if len(text) <= max_length:
            return text
        

        truncated = text[:max_length]
        

        for i in range(len(truncated) - 1, max(0, len(truncated) - 50), -1):
            if truncated[i] in '.!?':
                return truncated[:i+1] + "..."
        

        for i in range(len(truncated) - 1, max(0, len(truncated) - 30), -1):
            if truncated[i].isspace():
                return truncated[:i] + "..."
        

        return truncated + "..."

def list_files(directory: str, extensions: Optional[List[str]] = None) -> Generator[str, None, None]:
    
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                yield os.path.join(root, file)

def extract_documents(directory: str, extensions: Optional[List[str]] = None) -> Generator[tuple[str, str], None, None]:
    
    indexer = Indexer()
    for path, text in indexer.get_indexed_documents():
        yield path, text

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    for filepath, content in extract_documents(path):
        print(f"\n{filepath}:")
        print(textwrap.shorten(content, width=100))
