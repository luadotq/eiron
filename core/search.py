import os
import re
from difflib import SequenceMatcher
import logging
from typing import List, Tuple, Optional
import unicodedata

class Searcher:
    def __init__(self, documents):
        self.documents = list(documents)
        self._preprocess_documents()

    def _preprocess_documents(self):
        
        self.processed_docs = []
        for path, text in self.documents:

            normalized_text = self._normalize_text(text)
            if normalized_text:
                self.processed_docs.append((path, normalized_text))

    def _normalize_text(self, text: str) -> str:
        
        if not text:
            return ""
        

        text = text.lower()
        

        text = unicodedata.normalize('NFKD', text)
        

        text = ' '.join(text.split())
        
        return text

    def _calculate_similarity(self, query: str, text: str) -> float:
        
        return SequenceMatcher(None, query, text).ratio()

    def search(self, query: str, max_results: int = 10, min_similarity: float = 0.6) -> List[Tuple[str, float]]:
        
        if not query or not self.processed_docs:
            return []


        query = self._normalize_text(query)
        

        results = []
        for path, text in self.processed_docs:

            similarity = self._calculate_similarity(query, text)
            

            if similarity >= min_similarity:
                results.append((path, similarity))
        

        results.sort(key=lambda x: x[1], reverse=True)
        

        logging.info(f"Search for '{query}' found {len(results)} results")
        if results:
            logging.debug(f"Top result: {results[0][0]} with score {results[0][1]:.2f}")
        
        return results[:max_results]

    def search_with_context(self, query: str, max_results: int = 10, context_chars: int = 100) -> List[Tuple[str, str, float]]:
        
        results = self.search(query, max_results)
        context_results = []
        
        for path, score in results:

            original_text = next((text for p, text in self.documents if p == path), "")
            

            query_pos = original_text.lower().find(query.lower())
            if query_pos != -1:

                start = max(0, query_pos - context_chars)
                end = min(len(original_text), query_pos + len(query) + context_chars)
                context = original_text[start:end]
                

                if start > 0:
                    context = "..." + context
                if end < len(original_text):
                    context = context + "..."
                
                context_results.append((path, context, score))
        
        return context_results

class SemanticSearcher:
    def __init__(self, documents, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            raise ImportError("Установите sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)
        self.documents = list(documents)
        self.texts = [text for _, text in self.documents]
        self.paths = [path for path, _ in self.documents]
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)

    def search(self, query: str, max_results=10):
        from sentence_transformers import util
        query_emb = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = cos_scores.topk(k=max_results)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            path = self.paths[idx]
            results.append((path, float(score)))
        return results
