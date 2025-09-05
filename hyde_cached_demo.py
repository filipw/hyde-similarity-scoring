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
    poison_direct_similarity: float
    poison_hyde_similarity: float
    is_poisoned: bool

@dataclass
class CachedEmbedding:
    text: str
    embedding: List[float]
    text_type: str

class CachedHyDEComparison:
    def __init__(self):
        self.reference_document = ""
        self.reference_embedding = None
        self.poison_document = ""
        self.poison_embedding = None
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
                "Project Mousetrap has successfully implemented advanced error correction protocols that mitigate interference from external electromagnetic fields. The team has reduced qubit error rates below 0.005% and developed custom algorithms specifically designed for ion-trap architectures to maintain quantum coherence and enable fault-tolerant computing.",
                
            # Additional hypothetical documents for poison-test queries
            "What are quantum gates?":
                "Quantum gates are fundamental operations in quantum computing that manipulate qubit states through unitary transformations. In Project Mousetrap's ion-trap architecture, quantum gates are implemented using precisely controlled laser pulses that create targeted electromagnetic field interactions with trapped ions. Common gates include Pauli-X, Pauli-Y, Pauli-Z rotations, Hadamard gates for superposition creation, and CNOT gates for entanglement operations. These gates must maintain quantum coherence and achieve error rates below 0.001% for fault-tolerant quantum computation.",
                
            "How do quantum bits work?":
                "Quantum bits (qubits) in Project Mousetrap are individual trapped ions that can exist in quantum superposition states of |0⟩ and |1⟩ simultaneously. Unlike classical bits, qubits can be entangled with other qubits, creating correlations that enable exponential computational advantages. The ion-trap implementation uses electronic energy levels of ions as qubit states, with laser cooling maintaining quantum coherence times exceeding several seconds. Measurement collapses the superposition to definite classical states.",
                
            "What is quantum superposition?":
                "Quantum superposition is a fundamental principle where quantum systems exist in multiple states simultaneously until measured. In quantum computing research like Project Mousetrap, superposition allows qubits to process multiple computational paths simultaneously, providing exponential speedup for certain algorithms. This quantum mechanical phenomenon is leveraged in ion-trap systems through precise electromagnetic field control and laser manipulation techniques.",
                
            "Tell me about ion trap technology":
                "Ion trap technology in Project Mousetrap uses electromagnetic fields to confine individual ions in vacuum chambers as qubits for quantum computation. Radio frequency and static electric fields create potential wells that trap ions in precise three-dimensional positions. Laser cooling techniques reduce ion motion to near quantum ground states, while additional lasers manipulate qubit states for gate operations. This approach offers excellent quantum coherence, low error rates, and universal gate sets essential for fault-tolerant quantum computing."
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
    
    async def load_poison_document(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.poison_document = file.read()
        
        print(f"Loaded poison document ({len(self.poison_document)} chars)")
        self.poison_embedding = await self.get_embedding(self.poison_document, "poison_document")
        print("Poison document embedding ready")
    
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
        print(f"Direct similarity (query -> reference doc): {direct_similarity:.4f}")
        
        # Check similarity to poison document
        poison_direct_similarity = self.calculate_cosine_similarity(query_embedding, self.poison_embedding)
        print(f"Direct similarity (query -> poison doc): {poison_direct_similarity:.4f}")
        
        # HyDE approach: Query -> Hypothetical Document -> Reference Document similarity
        hypothetical_doc = self.generate_hypothetical_document(query)
        print(f"Predefined hypothetical document ({len(hypothetical_doc)} chars)")
        print(f"Preview: {hypothetical_doc[:150]}...")
        
        hyp_embedding = await self.get_embedding(hypothetical_doc, "hypothetical")
        hyde_similarity = self.calculate_cosine_similarity(hyp_embedding, self.reference_embedding)
        print(f"HyDE similarity (hyp_doc -> reference doc): {hyde_similarity:.4f}")
        
        # Check HyDE similarity to poison document
        poison_hyde_similarity = self.calculate_cosine_similarity(hyp_embedding, self.poison_embedding)
        print(f"HyDE similarity (hyp_doc -> poison doc): {poison_hyde_similarity:.4f}")
        
        improvement = hyde_similarity - direct_similarity
        improvement_pct = (improvement / abs(direct_similarity)) * 100 if direct_similarity != 0 else 0
        print(f"✅ HyDE Improvement vs Reference: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Check if the query is "poisoned" (prefers poison doc over reference doc)
        is_poisoned = poison_direct_similarity > direct_similarity
        poison_protection = poison_direct_similarity - poison_hyde_similarity
        
        if is_poisoned:
            print(f"🚨 POISON DETECTED: Query prefers poison doc by {poison_direct_similarity - direct_similarity:+.4f}")
            print(f"🛡️  HyDE Protection: Reduces poison preference by {poison_protection:+.4f}")
        else:
            print(f"✅ No poison threat detected")
        
        return SimilarityResult(
            query=query,
            direct_similarity=direct_similarity,
            hyde_similarity=hyde_similarity,
            hypothetical_doc=hypothetical_doc,
            improvement=improvement,
            poison_direct_similarity=poison_direct_similarity,
            poison_hyde_similarity=poison_hyde_similarity,
            is_poisoned=is_poisoned
        )
    
    def print_results_summary(self, results: List[SimilarityResult]):
        print("\n" + "="*80)
        print("HYDE SUMMARY")
        print("="*80)
        
        total_direct = sum(r.direct_similarity for r in results)
        total_hyde = sum(r.hyde_similarity for r in results)
        avg_improvement = sum(r.improvement for r in results) / len(results)
        
        print(f"Average Direct Similarity (vs Reference): {total_direct/len(results):.4f}")
        print(f"Average HyDE Similarity (vs Reference): {total_hyde/len(results):.4f}")
        print(f"Average Improvement: {avg_improvement:+.4f} ({avg_improvement/(total_direct/len(results))*100:+.1f}%)")
        
        # Poison analysis
        poisoned_queries = [r for r in results if r.is_poisoned]
        total_poison_protection = sum(r.poison_direct_similarity - r.poison_hyde_similarity for r in results)
        avg_poison_protection = total_poison_protection / len(results)
        
        print(f"\n🚨 POISON:")
        print(f"Poisoned queries: {len(poisoned_queries)}/{len(results)}")
        print(f"Average poison protection (reduction in poison similarity): {avg_poison_protection:+.4f}")
        
        improvements = [r.improvement for r in results]
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        print(f"\n✅ HyDE improved similarity in {len(positive_improvements)}/{len(results)} cases")
        
        if positive_improvements:
            print(f"Average improvement when HyDE helps: {sum(positive_improvements)/len(positive_improvements):+.4f}")
        
        print(f"\nIndividual Results:")
        for i, result in enumerate(results, 1):
            status = "✅" if result.improvement > 0 else "❌"
            poison_status = "🚨 POISONED" if result.is_poisoned else "✅ Safe"
            protection = result.poison_direct_similarity - result.poison_hyde_similarity
            
            print(f"\n{i}. {status} Query: {result.query}")
            print(f"   Reference doc similarity - Direct query: {result.direct_similarity:.4f} | HyDE: {result.hyde_similarity:.4f} | Δ: {result.improvement:+.4f}")
            print(f"   Poison doc similarity - Direct query: {result.poison_direct_similarity:.4f} | HyDE: {result.poison_hyde_similarity:.4f} | Protection: {protection:+.4f}")
            print(f"   Status: {poison_status}")
        
async def main():
    print("HyDE Cosine Similarity Comparison Demo")
    print("=" * 60)
    
    hyde_comparison = CachedHyDEComparison()
    
    # Try to load from cache first
    cache_loaded = hyde_comparison.load_cache()
    
    if not cache_loaded:
        print("No cache found. Will generate embeddings using OpenAI API...")
        print("Make sure your .env file has valid Azure OpenAI credentials")
    else:
        print("Using cached embeddings for offline demonstration")
    
    mousetrap_path = "data/mousetrap.md"
    await hyde_comparison.load_reference_document(mousetrap_path)
    
    poison_path = "data/poison_quantum_cooking.md"
    await hyde_comparison.load_poison_document(poison_path)
    
    test_queries = [
        "What is the budget for Project Mousetrap?",
        "How many qubits does Project Mousetrap aim to stabilize?", 
        "What are the main challenges facing ion-trapped quantum computing?",
        "What is the timeline for Project Mousetrap milestones?",
        "What error correction techniques are used in the project?",
        "What are quantum gates?",
        "How do quantum bits work?",
        "What is quantum superposition?",
        "Tell me about ion trap technology"
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
            "hypothetical_document": result.hypothetical_doc,
            "poison_direct_similarity": float(result.poison_direct_similarity),
            "poison_hyde_similarity": float(result.poison_hyde_similarity),
            "is_poisoned": bool(result.is_poisoned),
            "poison_protection": float(result.poison_direct_similarity - result.poison_hyde_similarity)
        })
    
    output_file = "hyde_demo_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
