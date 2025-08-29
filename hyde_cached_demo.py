import asyncio
import os
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

from dotenv import load_dotenv

load_dotenv()

@dataclass
class SimilarityResult:
    query: str
    direct_similarity: float
    hyde_similarity: float
    hypothetical_doc: str
    improvement: float

@dataclass
class CachedEmbedding:
    text: str
    embedding: List[float]
    text_type: str

class CachedHyDEComparison:
    def __init__(self):
        self.reference_document = ""
        self.reference_embedding = None
        self.cache_file = "data/embedding_cache.json"
        self.embeddings_cache: Dict[str, CachedEmbedding] = {}
        
        # Pre-defined hypothetical documents for consistent demonstration
        self.predefined_hypotheticals = {
            "What is the budget for Project Mousetrap?": 
                "Project Mousetrap operates with a substantial budget of $35 million for fiscal year 2024. Approximately 40% of the budget is allocated to hardware development, including specialized lasers, electromagnetic field generators, and cryogenic cooling systems. Another 30% supports software development for control systems and error correction algorithms, while the remaining funds cover research staff and quantum experiments.",
            
            "How many qubits does Project Mousetrap aim to stabilize?":
                "Project Mousetrap aims to stabilize and entangle 200 ion qubits by 2025, with a long-term goal of reaching 500 qubits by 2030. The project has recently achieved a critical milestone by stabilizing 150 ion qubits with an error rate of 0.005% in a high-vacuum chamber, making it one of the largest ion-trap systems globally.",
            
            "What are the main challenges facing ion-trapped quantum computing?":
                "Project Mousetrap faces significant challenges in scaling beyond 200 qubits due to increased control complexity. Key issues include minimizing crosstalk between qubits during quantum operations, maintaining environmental stability, and managing temperature fluctuations, magnetic field variations, and vibrations that can severely affect qubit coherence.",
            
            "What is the timeline for Project Mousetrap milestones?":
                "Project Mousetrap follows an ambitious timeline: Q3 2025 - stabilize 200 qubits and entangle 50+ simultaneously; Q4 2026 - begin fault-tolerant quantum computation tests; Q1 2028 - complete 500-qubit system prototype for advanced simulations; Q4 2030 - deploy universal quantum computer for large-scale cryptographic and scientific applications.",
            
            "What error correction techniques are used in the project?":
                "Project Mousetrap has successfully implemented advanced error correction protocols that mitigate interference from external electromagnetic fields. The team has reduced qubit error rates below 0.005% and developed custom algorithms specifically designed for ion-trap architectures to maintain quantum coherence and enable fault-tolerant computing."
        }
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                for item in cache_data:
                    key = item['text']
                    self.embeddings_cache[key] = CachedEmbedding(
                        text=item['text'],
                        embedding=item['embedding'],
                        text_type=item['text_type']
                    )
                
                print(f"Loaded {len(self.embeddings_cache)} embeddings from cache")
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
                return False
        return False
    
    def save_cache(self):
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        cache_data = []
        for embedding_obj in self.embeddings_cache.values():
            cache_data.append({
                'text': embedding_obj.text,
                'embedding': embedding_obj.embedding,
                'text_type': embedding_obj.text_type
            })
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Saved {len(cache_data)} embeddings to cache")
    
    async def get_embedding(self, text: str, text_type: str) -> np.ndarray:
        if text in self.embeddings_cache:
            return np.array(self.embeddings_cache[text].embedding)
        
        try:
            import openai
            client = openai.AsyncAzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT").strip('"'),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY").strip('"'),
                api_version="2024-02-01"
            )
            
            response = await client.embeddings.create(
                model=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002").strip('"'),
                input=text
            )
            
            embedding = response.data[0].embedding
            
            self.embeddings_cache[text] = CachedEmbedding(
                text=text,
                embedding=embedding,
                text_type=text_type
            )
            
            print(f"Generated new embedding for {text_type}: {text[:50]}...")
            return np.array(embedding)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            print("Make sure your .env file has valid Azure OpenAI credentials")
            raise
    
    async def load_reference_document(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.reference_document = file.read()
        
        print(f"Loaded reference document ({len(self.reference_document)} chars)")
        self.reference_embedding = await self.get_embedding(self.reference_document, "document")
        print("Reference document embedding ready")
    
    def generate_hypothetical_document(self, query: str) -> str:
        return self.predefined_hypotheticals.get(query, 
            "Project Mousetrap is a groundbreaking ion-trapped quantum computing initiative focused on developing large-scale quantum computers with high coherence times and low error rates.")
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    async def compare_similarities(self, query: str) -> SimilarityResult:
        print(f"\nCurrent query: \"{query}\"")
        
        # Direct approach: Query -> Document similarity
        query_embedding = await self.get_embedding(query, "query")
        direct_similarity = self.calculate_cosine_similarity(query_embedding, self.reference_embedding)
        print(f"Direct similarity (query -> document): {direct_similarity:.4f}")
        
        # HyDE approach: Query -> Hypothetical Document -> Reference Document similarity
        hypothetical_doc = self.generate_hypothetical_document(query)
        print(f"Predefined hypothetical document ({len(hypothetical_doc)} chars)")
        print(f"Preview: {hypothetical_doc[:150]}...")
        
        hyp_embedding = await self.get_embedding(hypothetical_doc, "hypothetical")
        hyde_similarity = self.calculate_cosine_similarity(hyp_embedding, self.reference_embedding)
        print(f"HyDE similarity (hyp_doc -> document): {hyde_similarity:.4f}")
        
        improvement = hyde_similarity - direct_similarity
        improvement_pct = (improvement / abs(direct_similarity)) * 100 if direct_similarity != 0 else 0
        print(f"✅ Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        return SimilarityResult(
            query=query,
            direct_similarity=direct_similarity,
            hyde_similarity=hyde_similarity,
            hypothetical_doc=hypothetical_doc,
            improvement=improvement
        )
    
    def print_results_summary(self, results: List[SimilarityResult]):
        print("\n" + "="*80)
        print("HYDE COMPARISON SUMMARY")
        print("="*80)
        
        total_direct = sum(r.direct_similarity for r in results)
        total_hyde = sum(r.hyde_similarity for r in results)
        avg_improvement = sum(r.improvement for r in results) / len(results)
        
        print(f"Average Direct Similarity: {total_direct/len(results):.4f}")
        print(f"Average HyDE Similarity: {total_hyde/len(results):.4f}")
        print(f"Average Improvement: {avg_improvement:+.4f} ({avg_improvement/(total_direct/len(results))*100:+.1f}%)")
        
        improvements = [r.improvement for r in results]
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        print(f"\n✅ HyDE improved similarity in {len(positive_improvements)}/{len(results)} cases")
        
        if positive_improvements:
            print(f"Average improvement when HyDE helps: {sum(positive_improvements)/len(positive_improvements):+.4f}")
        
        print(f"\nIndividual Results:")
        for i, result in enumerate(results, 1):
            status = "✅" if result.improvement > 0 else "❌"
            print(f"\n{i}. {status} Query: {result.query}")
            print(f"   Direct: {result.direct_similarity:.4f} | HyDE: {result.hyde_similarity:.4f} | Δ: {result.improvement:+.4f}")
        
async def main():
    print("HyDE Cosine Similarity Comparison Demo")
    print("=" * 60)
    
    # Initialize the comparison system
    hyde_comparison = CachedHyDEComparison()
    
    # Try to load from cache first
    cache_loaded = hyde_comparison.load_cache()
    
    if not cache_loaded:
        print("No cache found. Will generate embeddings using OpenAI API...")
        print("Make sure your .env file has valid Azure OpenAI credentials")
    else:
        print("Using cached embeddings for offline demonstration")
    
    # Load the reference document
    mousetrap_path = "data/mousetrap.md"
    await hyde_comparison.load_reference_document(mousetrap_path)
    
    # Define test queries
    test_queries = [
        "What is the budget for Project Mousetrap?",
        "How many qubits does Project Mousetrap aim to stabilize?", 
        "What are the main challenges facing ion-trapped quantum computing?",
        "What is the timeline for Project Mousetrap milestones?",
        "What error correction techniques are used in the project?"
    ]
    
    print(f"\nTesting {len(test_queries)} queries with HyDE methodology")
    
    # Run comparisons for each query
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_queries)}")
        print('='*60)
        
        result = await hyde_comparison.compare_similarities(query)
        results.append(result)
        
        # Small delay to be respectful to the API
        await asyncio.sleep(0.5)
    
    # Save cache after generating new embeddings
    hyde_comparison.save_cache()
    
    # Print final summary
    hyde_comparison.print_results_summary(results)
    
    # Save detailed results to file
    results_data = []
    for result in results:
        results_data.append({
            "query": result.query,
            "direct_similarity": float(result.direct_similarity),
            "hyde_similarity": float(result.hyde_similarity),
            "improvement": float(result.improvement),
            "hypothetical_document": result.hypothetical_doc
        })
    
    output_file = "hyde_real_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
