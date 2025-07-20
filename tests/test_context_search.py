


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_context_search():
    
    print("=== Test context search engine ===\n")
    
    try:
        from core.indexer import Indexer
        from core.context_search import ContextSearchEngine
        

        indexer = Indexer()
        

        test_docs = {
            "doc1.txt": "Artificial Intelligence (AI) is a field of computer science that deals with creating systems capable of performing tasks that require human intelligence.",
            "doc2.txt": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
            "doc3.txt": "Deep learning uses neural networks to process large amounts of data and extract complex patterns."
        }
        

        print("Indexing test documents...")
        for doc_id, content in test_docs.items():
            indexer.index.add_document(doc_id, content)
        
        print(f"Indexed {len(test_docs)} documents\n")
        

        search_engine = ContextSearchEngine(indexer)
        

        test_queries = [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "Python programming",
            "deep learning"
        ]
        

        search_modes = ['keyword', 'semantic', 'hybrid', 'exact']
        
        for mode in search_modes:
            print(f"\nSearch mode: {mode.upper()}")
            print("=" * 60)
            
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                
                results = search_engine.search(query, max_results=3, search_mode=mode)
                
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"\nResult {i}: {result.doc_id}")
                        print(f"  Relevance: {result.relevance_score:.3f}")
                        print(f"  Found terms: {', '.join(result.query_terms_found)}")
                        print(f"  Context: {result.context[:150]}...")
                        print(f"  Snippet quality: {result.snippet_quality:.2f}")
                else:
                    print("Results not found")
                
                print("  " + "-" * 40)
        
    except Exception as e:
        print(f"Error in context search test: {e}")
        import traceback
        traceback.print_exc()

def test_advanced_features():
    
    print("\n=== Test advanced features ===\n")
    
    try:
        from core.indexer import Indexer
        from core.context_search import ContextSearchEngine
        

        indexer = Indexer()
        

        large_doc = """
        Introduction to Artificial Intelligence.
        
        Artificial Intelligence (AI) is an interdisciplinary field that combines computer science, mathematics, psychology, linguistics, and other disciplines. The main goal of AI is to create systems capable of performing tasks that traditionally require human intelligence.
        
        History of Artificial Intelligence Development.
        
        The history of AI began in the 1950s when Alan Turing proposed a test to determine a machine's ability to exhibit intelligent behavior. Since then, the field has gone through several periods of optimism and disappointment, known as "AI winters."
        
        Main Directions of Artificial Intelligence.
        
        Modern AI includes many directions: machine learning, natural language processing, computer vision, robotics, expert systems, and others. Each direction solves specific tasks and uses various approaches and algorithms.
        
        Machine Learning as a Key Technology.
        
        Machine learning is one of the most important areas of AI. It allows computers to automatically improve their performance based on experience gained from data. Machine learning algorithms can be divided into three main categories: supervised learning, unsupervised learning, and reinforcement learning.
        
        Deep Learning and Neural Networks.
        
        Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. These networks can automatically extract hierarchical representations from data, making them particularly effective for image recognition, natural language processing, and other complex tasks.
        
        Applications of Artificial Intelligence.
        
        AI finds applications in many areas: from medical diagnostics and financial analysis to autonomous vehicles and recommendation systems. AI technologies have already changed many aspects of our daily lives and continue to develop rapidly.
        
        Ethical Aspects and Future of AI.
        
        The development of AI raises important ethical questions related to privacy, security, transparency, and impact on the job market. Understanding and addressing these issues is critical for responsible development and deployment of AI technologies.
        """
        

        print("Indexing large document...")
        indexer.index.add_document("large_doc.txt", large_doc)
        print("Document indexed\n")
        

        search_engine = ContextSearchEngine(indexer)
        

        complex_queries = [
            "history of artificial intelligence development",
            "machine learning algorithms",
            "ethical aspects of AI",
            "neural networks deep learning",
            "artificial intelligence applications in medicine"
        ]
        
        print("Testing complex queries:")
        print("=" * 60)
        
        for query in complex_queries:
            print(f"\nQuery: '{query}'")
            

            results = search_engine.search(query, max_results=2, search_mode='hybrid')
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}: {result.doc_id}")
                    print(f"  Relevance: {result.relevance_score:.3f}")
                    print(f"  Found terms: {', '.join(result.query_terms_found)}")
                    print(f"  Context: {result.context}")
                    print(f"  Snippet quality: {result.snippet_quality:.2f}")
                    print(f"  Semantic similarity: {result.semantic_similarity:.3f}")
            else:
                print("Results not found")
            
            print("  " + "-" * 40)
        
    except Exception as e:
        print(f"Error in advanced features test: {e}")
        import traceback
        traceback.print_exc()

def main():
    
    print("\nContext Engine test")
    
    test_context_search()
    test_advanced_features()
    
    print("\nTests passed")

if __name__ == "__main__":
    main() 