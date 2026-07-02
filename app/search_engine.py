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
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            
        self.load_indices()

    def load_indices(self):
        """Loads compiled sparse and dense matrices into cache memory."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Missing data store artifacts. Run init_db.py first.")
            
        with open(self.manifest_path, "rb") as f:
            self.catalog = pickle.load(f)
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        self.embeddings = np.load(self.embeddings_path)
        print(f"✅ Search memory loaded with {len(self.catalog)} candidate assessments.")

    def _get_vector_scores(self, query_text: str) -> list:
        """Retrieves dense vector similarity scores using Cosine Similarity."""
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=query_text,
                task_type="retrieval_query"
            )
            query_vector = np.array(response['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Vector engine query failed: {e}.")
            return []

        matrix_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm > 0 and len(matrix_norms) > 0:
            similarities = np.dot(self.embeddings, query_vector) / (matrix_norms * query_norm)
        else:
            similarities = np.zeros(len(self.catalog))
            
        return list(enumerate(similarities.tolist()))

    def _get_lexical_scores(self, query_text: str) -> list:
        """Retrieves sparse keyword token relevance scores via the BM25 index matrix."""
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return list(enumerate(scores.tolist()))

    def get_hybrid_shortlist(self, query_text: str, top_k: int = 10, target_level: str = None, test_type_filter: str = None) -> list:
        """
        Fuses sparse and dense search results using Reciprocal Rank Fusion (RRF)
        with tie-breaking protection, then applies uniform metadata filters.
        """
        dense_scores = self._get_vector_scores(query_text)
        lexical_scores = self._get_lexical_scores(query_text)
        
        if not dense_scores or not lexical_scores:
            return []

        # 🎯 RRF Tie-Breaking Protection: Only fuse the top 50 dense hits 
        # and lexical matches with a score strictly greater than 0.0
        sorted_dense = sorted(dense_scores, key=lambda x: x[1], reverse=True)[:50]
        sorted_lexical = sorted([x for x in lexical_scores if x[1] > 0], key=lambda x: x[1], reverse=True)
        
        k_constant = 60
        rrf_scores = {}
        
        for rank, (doc_idx, score) in enumerate(sorted_dense):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k_constant + rank)
            
        for rank, (doc_idx, score) in enumerate(sorted_lexical):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k_constant + rank)
            
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_shortlist = []
        for doc_idx, rrf_score in sorted_indices:
            item = self.catalog[doc_idx]
            
            # 1. Seniority Level Metadata Filter
            if target_level:
                job_levels_clean = [lvl.lower() for lvl in item.get("job_levels", [])]
                if target_level.lower() not in job_levels_clean:
                    continue
            
            # 2. Unified Test Type Normalization
            keys_combined = " ".join(item.get("keys", [])).lower() + " " + item.get("description", "").lower()
            is_personality = "personality" in keys_combined or "behavioral" in keys_combined or "opq" in keys_combined
            
            if test_type_filter:
                if test_type_filter.upper() == "P" and not is_personality:
                    continue
                if test_type_filter.upper() == "K" and is_personality:
                    continue

            # Every URL returned comes strictly from your scraped catalog [cite: 64]
            final_shortlist.append({
                "name": item.get("name"),
                "url": item.get("link"),
                "test_type": "P" if is_personality else "K"
            })
            
            if len(final_shortlist) == top_k:
                break
                
        return final_shortlist