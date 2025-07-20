from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path
import json
import logging
from core.memory_manager import MemoryManager
from transformers import AutoTokenizer, AutoModel
import os

class SemanticSearchConfig:
    def __init__(self):
        self.batch_size = 32
        self.similarity_threshold = 0.7
        self.use_gpu = True
        self.device = self._get_device()
        self.model_name = "all-MiniLM-L6-v2"
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _get_device(self) -> str:
        
        try:
            if self.use_gpu:

                if torch.cuda.is_available():
                    return "cuda"
                

                if hasattr(torch, 'vulkan') and torch.vulkan.is_available():
                    return "vulkan"
                

                if hasattr(torch, 'mps') and torch.mps.is_available():
                    return "mps"
            

            return "cpu"
        except Exception as e:
            logging.warning(f"Error detecting device: {e}. Falling back to CPU.")
            return "cpu"

    def get_available_models(self) -> List[str]:
        
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]

    def set_model_parameters(self, selected_model: str, batch_size: int, 
                           similarity_threshold: float, use_gpu: bool):
        
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self.device = self._get_device()
        
        if selected_model != self.model_name:
            self.model_name = selected_model
            self._initialize_model()

    def _initialize_model(self):
        
        try:
            logging.info(f"Initializing model {self.model_name} on device: {self.device}")
            

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            

            self.model.eval()
            

            test_text = "Test encoding"
            self.model.encode(test_text)
            
            logging.info(f"Model initialized successfully on {self.device}")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")

            if self.device != "cpu":
                logging.info("Falling back to CPU")
                self.device = "cpu"
                self.use_gpu = False
                try:
                    self.model = SentenceTransformer(
                        self.model_name,
                        device="cpu"
                    )
                    self.model.eval()
                    logging.info("Model initialized successfully on CPU")
                except Exception as e:
                    logging.error(f"Failed to initialize model on CPU: {e}")
                    self.model = None

class EnhancedSemanticSearcher:
    def __init__(self, documents: List[Tuple[str, str]], config: Optional[SemanticSearchConfig] = None):
        self.documents = documents
        self.config = config or SemanticSearchConfig()
        if self.config.model is None:
            raise RuntimeError("Failed to initialize semantic search model")
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        
        try:
            if not self.documents:
                logging.warning("No documents provided for indexing")
                self.embeddings = torch.tensor([])
                return


            texts = [doc[1] for doc in self.documents]
            self.embeddings = self.config.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.config.device
            )
            logging.info(f"Initialized embeddings for {len(self.documents)} documents")
        except Exception as e:
            logging.error(f"Error initializing embeddings: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        
        try:
            if not query.strip():
                return []


            query_embedding = self.config.model.encode(
                query,
                convert_to_tensor=True,
                device=self.config.device
            )

            if len(self.embeddings) == 0:
                return []


            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.embeddings
            )


            top_results = []
            for idx in similarities.argsort(descending=True):
                score = similarities[idx].item()
                if score >= self.config.similarity_threshold:
                    top_results.append((self.documents[idx][0], score))
                    if len(top_results) >= top_k:
                        break

            return top_results
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return []

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        
        try:
            if not queries or len(self.embeddings) == 0:
                return [[] for _ in queries]


            valid_queries = [q for q in queries if q.strip()]
            if not valid_queries:
                return [[] for _ in queries]


            query_embeddings = self.config.model.encode(
                valid_queries,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.config.device
            )


            similarities = torch.nn.functional.cosine_similarity(
                query_embeddings.unsqueeze(1),
                self.embeddings.unsqueeze(0),
                dim=2
            )


            results = []
            for query_similarities in similarities:
                top_results = []
                for idx in query_similarities.argsort(descending=True):
                    score = query_similarities[idx].item()
                    if score >= self.config.similarity_threshold:
                        top_results.append((self.documents[idx][0], score))
                        if len(top_results) >= top_k:
                            break
                results.append(top_results)


            final_results = []
            query_idx = 0
            for q in queries:
                if q.strip():
                    final_results.append(results[query_idx])
                    query_idx += 1
                else:
                    final_results.append([])

            return final_results
        except Exception as e:
            logging.error(f"Error during batch search: {e}")
            return [[] for _ in queries]

    def get_available_models(self) -> List[str]:
        
        return list(self.config.models.keys())

    def set_model_parameters(self, **kwargs):
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.config.save_config()

    def clear_cache(self):
        
        self.models.clear()
        self.embeddings.clear()
        self.memory_manager.clear_cache()
        torch.cuda.empty_cache() if self.config.use_gpu else None 