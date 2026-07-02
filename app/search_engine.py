import os
import pickle
import numpy as np
import google.generativeai as genai

class HybridSearchEngine:
    def __init__(self, data_dir: str = "data_store"):
        self.data_dir = data_dir
        self.manifest_path = os.path.join(data_dir, "catalog_manifest.pkl")
        self.bm25_path = os.path.join(data_dir, "bm25_index.pkl")
        self.embeddings_path = os.path.join(data_dir, "vector_embeddings.npy")
        
        # Initialize Google SDK natively using your active free token string
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            
        self.load_indices()

    def load_indices(self):
        """Loads compiled sparse and dense matrices into cache memory."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Missing data store artifacts. Run init_db.py first.")
            
        print("💾 Loading compiled RAG index matrices into server cache...")
        with open(self.manifest_path, "rb") as f:
            self.catalog = pickle.load(f)
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        self.embeddings = np.load(self.embeddings_path)
        print(f"✅ Search memory loaded with {len(self.catalog)} candidate assessments.")

    def _get_vector_ranks(self, query_text: str) -> list:
        """Retrieves dense vector score sequence ranking using Cosine Similarity."""
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=query_text,
                task_type="retrieval_query"
            )
            query_vector = np.array(response['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Vector engine query failed: {e}. Falling back to default baseline matrix.")
            return list(range(len(self.catalog)))

        # Vectorized calculation running completely locally on your CPU via NumPy
        matrix_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm > 0 and len(matrix_norms) > 0:
            similarities = np.dot(self.embeddings, query_vector) / (matrix_norms * query_norm)
        else:
            similarities = np.zeros(len(self.catalog))
            
        return np.argsort(similarities)[::-1].tolist()

    def _get_lexical_ranks(self, query_text: str) -> list:
        """Retrieves sparse keyword token relevance ranking via the BM25 index matrix."""
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return np.argsort(scores)[::-1].tolist()

    def get_hybrid_shortlist(self, query_text: str, top_k: int = 10, target_level: str = None, test_type_filter: str = None) -> list:
        """
        Fuses sparse and dense search results using Reciprocal Rank Fusion (RRF),
        then applies metadata filters to output highly relevant matches.
        """
        dense_ranks = self._get_vector_ranks(query_text)
        lexical_ranks = self._get_lexical_ranks(query_text)
        
        # Standardize hyperparameter constant scale
        k_constant = 60
        rrf_scores = {}
        
        # Map out dense scores
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k_constant + rank)
            
        # Add sparse layout rankings
        for rank, doc_idx in enumerate(lexical_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k_constant + rank)
            
        # Compile items sorted deterministically by highest rank convergence
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_shortlist = []
        for doc_idx, score in sorted_indices:
            item = self.catalog[doc_idx]
            
            # --- Metadata Filtering Layer ---
            # Strict assignment guardrail: Filters matching candidate roles mid-conversation
            if target_level:
                job_levels_clean = [lvl.lower() for lvl in item.get("job_levels", [])]
                if target_level.lower() not in job_levels_clean:
                    continue
                    
            if test_type_filter:
                # 'P' maps to Personality & Behavior; 'K' maps to Ability/Skills/Knowledge
                keys_combined = " ".join(item.get("keys", [])).lower()
                if test_type_filter.upper() == "P" and "personality" not in keys_combined:
                    continue
                if test_type_filter.upper() == "K" and "ability" not in keys_combined and "aptitude" not in keys_combined:
                    continue

            # Map directly to your expected response schema fields
            final_shortlist.append({
                "name": item.get("name"),
                "url": item.get("link"),
                "test_type": "P" if "personality" in " ".join(item.get("keys", [])).lower() else "K"
            })
            
            if len(final_shortlist) == top_k:
                break
                
        return final_shortlist