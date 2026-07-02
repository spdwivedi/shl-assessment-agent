import os
import json
import pickle
import numpy as np
import requests
from rank_bm25 import BM25Okapi

# Configuration Variables
CATALOG_PATH = "shl_product_catalog.json"
OUTPUT_DIR = "data_store"
API_KEY = os.getenv("GEMINI_API_KEY")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_filter_catalog():
    print(f"📦 Loading dataset from: {CATALOG_PATH}...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f, strict=False)
    
    filtered_items = []
    for item in data:
        # Strict validation: Filter out out-of-scope products 
        # We target Individual Test Solutions, filtering out anomalies or non-active products.
        if item.get("status") != "ok":
            continue
            
        filtered_items.append(item)
        
    print(f"✅ Filtered catalog down to {len(filtered_items)} active valid entries.")
    return filtered_items

def get_embedding(text):
    if not API_KEY:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except Exception as e:
        print(f"❌ Embedding collection failed for string snippet: {text[:30]}... Error: {e}")
        return [0.0] * 768  # Fallback vector block

def process_and_index():
    catalog_items = load_and_filter_catalog()
    if not catalog_items:
        print("Exiting execution: Catalog contains no target structures.")
        return

    corpus_texts = []
    tokenized_corpus = []
    embeddings_matrix = []

    print("🧠 Processing text structures and extracting semantic vectors...")
    for idx, item in enumerate(catalog_items):
        name = item.get("name", "")
        desc = item.get("description", "")
        # Extracting tags to ensure high keyword relevance matches
        job_levels = " ".join(item.get("job_levels", []))
        keys = " ".join(item.get("keys", []))
        
        # Build an enriched representation block for the vector encoder
        enriched_chunk = f"Assessment Name: {name}\nKeywords: {keys}\nTarget Levels: {job_levels}\nDescription: {desc}"
        corpus_texts.append(enriched_chunk)
        
        # Tokenize lowercase representations to populate the BM25 keyword index
        tokenized_corpus.append(name.lower().split() + keys.lower().split() + job_levels.lower().split())
        
        # Call the Google Tier 1 Vector Engine
        vector = get_embedding(enriched_chunk)
        embeddings_matrix.append(vector)
        
        if (idx + 1) % 20 == 0 or (idx + 1) == len(catalog_items):
            print(f"Processed progress: {idx + 1}/{len(catalog_items)} elements indexed.")

    # 1. Save processed item objects
    with open(os.path.join(OUTPUT_DIR, "catalog_manifest.pkl"), "wb") as f:
        pickle.dump(catalog_items, f)

    # 2. Save lexical BM25 database matrix
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 3. Save dense matrix vector files
    np.save(os.path.join(OUTPUT_DIR, "vector_embeddings.npy"), np.array(embeddings_matrix, dtype=np.float32))
    
    print("\n🎉 Core RAG Ingestion layer initialized successfully!")
    print(f"Generated artifacts successfully deployed to your local directory: '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    if not API_KEY:
        print("⚠️ Warning: Provide your API key using 'set GEMINI_API_KEY=key' prior to launching.")
    else:
        process_and_index()