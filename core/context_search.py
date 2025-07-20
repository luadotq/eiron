import re
import logging
import math
from typing import List, Tuple, Optional, Dict, Set, NamedTuple
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SearchResult(NamedTuple):
    
    doc_id: str
    relevance_score: float
    context: str
    query_terms_found: List[str]
    term_positions: Dict[str, List[int]]
    context_start: int
    context_end: int
    snippet_quality: float
    semantic_similarity: float

class ContextSearchEngine:
    
    
    def __init__(self, indexer, config: Dict = None):
        self.indexer = indexer
        self.config = config or self._get_default_config()
        self._setup_text_processing()
        
    def _get_default_config(self) -> Dict:
        
        return {
            'min_context_length': 50,
            'max_context_length': 800,
            'context_expansion': 150,
            'sentence_boundary_weight': 0.8,
            'paragraph_boundary_weight': 0.9,
            'query_term_density_threshold': 0.3,
            'semantic_similarity_threshold': 0.4,
            'proximity_bonus_range': 50,
            'phrase_match_bonus': 2.0,
            'exact_match_bonus': 1.5,
            'position_bonus_factor': 0.3,
            'use_semantic_search': True,
            'semantic_weight': 0.4,
            'keyword_weight': 0.6
        }
    
    def _setup_text_processing(self):
        

        self.stop_words = {

            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
            'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
            'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now', 'do', 'does', 'did', 'would', 'could', 'may', 'might',
            'must', 'shall', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            

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
    
    def search(self, query: str, max_results: int = 10, 
               search_mode: str = 'hybrid') -> List[SearchResult]:
        
        if not query.strip():
            return []
        

        query = query.strip()
        query_terms = self._extract_query_terms(query)
        
        if not query_terms:
            return []
        

        documents = list(self.indexer.get_indexed_documents())
        if not documents:
            return []
        

        if search_mode == 'keyword':
            results = self._keyword_search(documents, query_terms, max_results)
        elif search_mode == 'semantic':
            results = self._semantic_search(documents, query, max_results)
        elif search_mode == 'exact':
            results = self._exact_search(documents, query_terms, max_results)
        else:
            results = self._hybrid_search(documents, query, query_terms, max_results)
        

        results = self._post_process_results(results, query, query_terms)
        
        return results[:max_results]
    
    def _extract_query_terms(self, query: str) -> List[str]:
        
        if not query:
            return []
        

        query = query.lower()
        

        terms = re.findall(r'\b\w+\b', query)
        

        filtered_terms = []
        for term in terms:

            if term and len(term) > 2 and term not in self.stop_words and not term.isdigit():
                filtered_terms.append(term)
        
        return filtered_terms
    
    def _keyword_search(self, documents: List[Tuple[str, str]], 
                       query_terms: List[str], max_results: int) -> List[SearchResult]:
        
        results = []
        
        for doc_id, text in documents:
            if not text:
                continue
            

            term_positions = self._find_term_positions(text, query_terms)
            
            if not term_positions:
                continue
            

            relevance_score = self._calculate_keyword_relevance(text, query_terms, term_positions)
            
            if relevance_score > 0:

                context_info = self._extract_optimal_context(text, query_terms, term_positions)
                
                if context_info:
                    result = SearchResult(
                        doc_id=doc_id,
                        relevance_score=relevance_score,
                        context=context_info['context'],
                        query_terms_found=context_info['terms_found'],
                        term_positions=context_info['positions'],
                        context_start=context_info['start'],
                        context_end=context_info['end'],
                        snippet_quality=context_info['quality'],
                        semantic_similarity=0.0
                    )
                    results.append(result)
        

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _semantic_search(self, documents: List[Tuple[str, str]], 
                        query: str, max_results: int) -> List[SearchResult]:
        
        if not self.config['use_semantic_search']:
            return []
        
        try:

            texts = [text for _, text in documents if text]
            if not texts:
                return []
            
            vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                analyzer='word'
            )
            

            doc_vectors = vectorizer.fit_transform(texts)
            query_vector = vectorizer.transform([query])
            

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            

            results = []
            for i, (doc_id, text) in enumerate(documents):
                if similarities[i] > self.config['semantic_similarity_threshold']:

                    context_info = self._extract_semantic_context(text, query)
                    
                    if context_info:
                        result = SearchResult(
                            doc_id=doc_id,
                            relevance_score=similarities[i],
                            context=context_info['context'],
                            query_terms_found=context_info['terms_found'],
                            term_positions=context_info['positions'],
                            context_start=context_info['start'],
                            context_end=context_info['end'],
                            snippet_quality=context_info['quality'],
                            semantic_similarity=similarities[i]
                        )
                        results.append(result)
            
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results
            
        except Exception as e:
            logging.error(f"Error in semantic search: {e}")
            return []
    
    def _exact_search(self, documents: List[Tuple[str, str]], 
                     query_terms: List[str], max_results: int) -> List[SearchResult]:
        
        results = []
        
        for doc_id, text in documents:
            if not text:
                continue
            

            exact_matches = self._find_exact_matches(text, query_terms)
            
            if exact_matches:
                for match_info in exact_matches:
                    relevance_score = match_info['score']
                    context_info = self._extract_context_around_position(
                        text, match_info['position'], match_info['length']
                    )
                    
                    if context_info:
                        result = SearchResult(
                            doc_id=doc_id,
                            relevance_score=relevance_score,
                            context=context_info['context'],
                            query_terms_found=match_info['terms'],
                            term_positions=context_info['positions'],
                            context_start=context_info['start'],
                            context_end=context_info['end'],
                            snippet_quality=context_info['quality'],
                            semantic_similarity=0.0
                        )
                        results.append(result)
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _hybrid_search(self, documents: List[Tuple[str, str]], 
                      query: str, query_terms: List[str], max_results: int) -> List[SearchResult]:
        

        keyword_results = self._keyword_search(documents, query_terms, max_results * 2)
        

        semantic_results = self._semantic_search(documents, query, max_results * 2)
        

        combined_results = self._combine_search_results(keyword_results, semantic_results)
        
        return combined_results[:max_results]
    
    def _find_term_positions(self, text: str, query_terms: List[str]) -> Dict[str, List[int]]:
        
        positions = {}
        text_lower = text.lower()
        
        for term in query_terms:
            term_positions = []
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                term_positions.append(pos)
                start = pos + 1
            if term_positions:
                positions[term] = term_positions
        
        return positions
    
    def _calculate_keyword_relevance(self, text: str, query_terms: List[str], 
                                   term_positions: Dict[str, List[int]]) -> float:
        
        if not term_positions:
            return 0.0
        
        score = 0.0
        

        found_terms = len(term_positions)
        score += found_terms / len(query_terms)
        

        for term, positions in term_positions.items():
            score += len(positions) * 0.1
        

        proximity_bonus = self._calculate_proximity_bonus(term_positions)
        score += proximity_bonus
        

        position_bonus = self._calculate_position_bonus(term_positions, len(text))
        score += position_bonus
        

        phrase_bonus = self._calculate_phrase_bonus(text, query_terms)
        score += phrase_bonus
        
        return min(score, 1.0)
    
    def _calculate_proximity_bonus(self, term_positions: Dict[str, List[int]]) -> float:
        
        if len(term_positions) < 2:
            return 0.0
        
        bonus = 0.0
        positions = []
        

        for pos_list in term_positions.values():
            positions.extend(pos_list)
        
        positions.sort()
        

        if len(positions) > 1:
            distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_distance = sum(distances) / len(distances)
            

            if avg_distance <= self.config['proximity_bonus_range']:
                bonus = 0.3 * (1.0 - avg_distance / self.config['proximity_bonus_range'])
        
        return bonus
    
    def _calculate_position_bonus(self, term_positions: Dict[str, List[int]], text_length: int) -> float:
        
        if not term_positions:
            return 0.0
        

        all_positions = []
        for positions in term_positions.values():
            all_positions.extend(positions)
        
        if not all_positions:
            return 0.0
        
        earliest_position = min(all_positions)
        

        position_factor = 1.0 - (earliest_position / text_length)
        return position_factor * self.config['position_bonus_factor']
    
    def _calculate_phrase_bonus(self, text: str, query_terms: List[str]) -> float:
        
        if len(query_terms) < 2:
            return 0.0
        
        bonus = 0.0
        text_lower = text.lower()
        

        for i in range(len(query_terms) - 1):
            for j in range(i + 2, len(query_terms) + 1):
                phrase = ' '.join(query_terms[i:j])
                if phrase in text_lower:

                    phrase_length = j - i
                    bonus += self.config['phrase_match_bonus'] * phrase_length
        
        return min(bonus, 0.5)
    
    def _extract_optimal_context(self, text: str, query_terms: List[str], 
                               term_positions: Dict[str, List[int]]) -> Optional[Dict]:
        
        if not term_positions:
            return None
        

        best_position = self._find_best_context_position(term_positions)
        
        if best_position is None:
            return None
        

        context_info = self._extract_context_around_position(text, best_position)
        
        if not context_info:
            return None
        

        quality = self._calculate_snippet_quality(
            context_info['context'], query_terms, term_positions
        )
        
        context_info['quality'] = quality
        return context_info
    
    def _find_best_context_position(self, term_positions: Dict[str, List[int]]) -> Optional[int]:
        
        if not term_positions:
            return None
        

        all_positions = []
        for positions in term_positions.values():
            all_positions.extend(positions)
        
        if not all_positions:
            return None
        

        best_position = 0
        best_score = 0
        
        for pos in all_positions:
            nearby_terms = 0
            for other_pos in all_positions:
                if abs(other_pos - pos) <= self.config['context_expansion']:
                    nearby_terms += 1
            
            if nearby_terms > best_score:
                best_score = nearby_terms
                best_position = pos
        
        return best_position
    
    def _extract_context_around_position(self, text: str, position: int, 
                                       length: int = None) -> Optional[Dict]:
        
        if position is None or position < 0 or position >= len(text):
            return None
        
        if length is None:
            length = self.config['context_expansion']
        

        start = max(0, position - length)
        end = min(len(text), position + length)
        

        start = self._expand_to_sentence_boundary(text, start, direction='backward')
        end = self._expand_to_sentence_boundary(text, end, direction='forward')
        

        context = text[start:end].strip()
        
        if len(context) < self.config['min_context_length']:
            return None
        

        context_positions = {}
        context_lower = context.lower()
        for term in self._extract_query_terms(context):
            positions = []
            start_pos = 0
            while True:
                pos = context_lower.find(term, start_pos)
                if pos == -1:
                    break
                positions.append(pos)
                start_pos = pos + 1
            if positions:
                context_positions[term] = positions
        
        return {
            'context': context,
            'start': start,
            'end': end,
            'positions': context_positions,
            'terms_found': list(context_positions.keys()),
            'quality': 0.0
        }
    
    def _expand_to_sentence_boundary(self, text: str, position: int, direction: str) -> int:
        
        if direction == 'backward':

            for i in range(position, max(0, position - 200), -1):
                if text[i] in '.!?':
                    return min(position, i + 1)
            return max(0, position - 100)
        else:

            for i in range(position, min(len(text), position + 200)):
                if text[i] in '.!?':
                    return min(len(text), i + 1)
            return min(len(text), position + 100)
    
    def _calculate_snippet_quality(self, context: str, query_terms: List[str], 
                                 term_positions: Dict[str, List[int]]) -> float:
        
        if not context or not query_terms:
            return 0.0
        
        quality = 0.0
        

        found_terms = len([term for term in query_terms if term in context.lower()])
        term_density = found_terms / len(query_terms) if query_terms else 0
        quality += term_density * 0.4
        

        context_length = len(context)
        if self.config['min_context_length'] <= context_length <= self.config['max_context_length']:
            quality += 0.3
        elif context_length > self.config['max_context_length']:
            quality += 0.1
        

        if context.startswith('...') or context.endswith('...'):
            quality += 0.2
        

        context_lower = context.lower()
        for term in query_terms[:3]:
            if term in context_lower[:100]:
                quality += 0.1
                break
        
        return min(quality, 1.0)
    
    def _find_exact_matches(self, text: str, query_terms: List[str]) -> List[Dict]:
        
        matches = []
        text_lower = text.lower()
        

        for i in range(len(query_terms)):
            for j in range(i + 1, len(query_terms) + 1):
                phrase = ' '.join(query_terms[i:j])
                start = 0
                while True:
                    pos = text_lower.find(phrase, start)
                    if pos == -1:
                        break
                    
                    matches.append({
                        'position': pos,
                        'length': len(phrase),
                        'terms': query_terms[i:j],
                        'score': len(phrase) * self.config['exact_match_bonus']
                    })
                    start = pos + 1
        
        return matches
    
    def _extract_semantic_context(self, text: str, query: str) -> Optional[Dict]:
        

        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return None
        

        best_sentence = None
        best_score = 0
        
        for sentence in sentences:

            sentence_lower = sentence.lower()
            query_lower = query.lower()
            
            score = 0
            for word in query_lower.split():
                if word in sentence_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if not best_sentence:
            return None
        

        sentence_pos = text.find(best_sentence)
        if sentence_pos == -1:
            return None
        
        return self._extract_context_around_position(text, sentence_pos)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        
        if not text:
            return []
        

        sentences = re.split(r'[.!?]+', text)
        

        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _combine_search_results(self, keyword_results: List[SearchResult], 
                              semantic_results: List[SearchResult]) -> List[SearchResult]:
        

        combined = {}
        

        for result in keyword_results:
            combined[result.doc_id] = {
                'keyword_score': result.relevance_score,
                'semantic_score': 0.0,
                'result': result
            }
        

        for result in semantic_results:
            if result.doc_id in combined:
                combined[result.doc_id]['semantic_score'] = result.semantic_similarity
            else:
                combined[result.doc_id] = {
                    'keyword_score': 0.0,
                    'semantic_score': result.semantic_similarity,
                    'result': result
                }
        

        final_results = []
        for doc_id, scores in combined.items():
            combined_score = (
                scores['keyword_score'] * self.config['keyword_weight'] +
                scores['semantic_score'] * self.config['semantic_weight']
            )
            

            result = scores['result']
            updated_result = SearchResult(
                doc_id=result.doc_id,
                relevance_score=combined_score,
                context=result.context,
                query_terms_found=result.query_terms_found,
                term_positions=result.term_positions,
                context_start=result.context_start,
                context_end=result.context_end,
                snippet_quality=result.snippet_quality,
                semantic_similarity=result.semantic_similarity
            )
            final_results.append(updated_result)
        

        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return final_results
    
    def _post_process_results(self, results: List[SearchResult], 
                            query: str, query_terms: List[str]) -> List[SearchResult]:
        
        processed_results = []
        
        for result in results:

            improved_context = self._improve_context(
                result.context, query_terms, result.term_positions
            )
            

            updated_result = SearchResult(
                doc_id=result.doc_id,
                relevance_score=result.relevance_score,
                context=improved_context,
                query_terms_found=result.query_terms_found,
                term_positions=result.term_positions,
                context_start=result.context_start,
                context_end=result.context_end,
                snippet_quality=result.snippet_quality,
                semantic_similarity=result.semantic_similarity
            )
            processed_results.append(updated_result)
        
        return processed_results
    
    def _improve_context(self, context: str, query_terms: List[str], 
                        term_positions: Dict[str, List[int]]) -> str:
        
        if not context or not query_terms:
            return context
        


        

        context = re.sub(r'\s+', ' ', context).strip()
        
        return context 