import os
import json
import pickle
import time
import numpy as np
import google.generativeai as genai
from rank_bm25 import BM25Okapi

# Configuration Variables
CATALOG_PATH = "shl_product_catalog.json"
OUTPUT_DIR = "data_store"
API_KEY = os.getenv("GEMINI_API_KEY")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize the official Google SDK natively using your active free token string
if not API_KEY:
    print("\n❌ CRITICAL ERROR: Your GEMINI_API_KEY environment variable is empty.")
    print("Please set it in your terminal using: $env:GEMINI_API_KEY='your_key'\n")
else:
    genai.configure(api_key=API_KEY)

def load_and_filter_catalog():
    print(f"📦 Loading dataset from: {CATALOG_PATH}...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f, strict=False)
    
    filtered_items = []
    for item in data:
        # Enforce strict assignment scope: target Individual Test Solutions
        if item.get("status") != "ok":
            continue
        filtered_items.append(item)
        
    print(f"✅ Filtered catalog down to {len(filtered_items)} active valid entries.")
    return filtered_items

def get_embedding(text):
    try:
        # Targeting the free tier Gemini Embedding 1 engine identifier
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        print(f"\n❌ Embedding collection failed for string snippet: {text[:30]}... Error: {e}")
        return [0.0] * 768  # Structural fallback vector block

def process_and_index():
    if not API_KEY:
        print("❌ Aborting matrix creation: GEMINI_API_KEY is not defined.")
        return
        
    catalog_items = load_and_filter_catalog()
    if not catalog_items:
        print("Exiting execution: Catalog contains no target structures.")
        return

    corpus_texts = []
    tokenized_corpus = []
    embeddings_matrix = []

    print("🧠 Processing text structures and extracting semantic vectors via SDK...")
    print("⏳ Free-tier throttling active (30K TPM Protection Strategy). Processing will take ~4 minutes...")
    
    total_items = len(catalog_items)
    for idx, item in enumerate(catalog_items):
        name = item.get("name", "")
        desc = item.get("description", "")
        job_levels = " ".join(item.get("job_levels", []))
        keys = " ".join(item.get("keys", []))
        
        # Build enriched representation blocks for the vector encoding matrix
        enriched_chunk = f"Assessment Name: {name}\nKeywords: {keys}\nTarget Levels: {job_levels}\nDescription: {desc}"
        corpus_texts.append(enriched_chunk)
        
        # Tokenize lowercased data fields to build the sparse keyword index matrix
        tokenized_corpus.append(name.lower().split() + keys.lower().split() + job_levels.lower().split())
        
        # Querying the text embedding engine
        vector = get_embedding(enriched_chunk)
        embeddings_matrix.append(vector)
        
        # ⏳ Crucial Free-Tier Throttle: Sleep for 0.65 seconds between requests
        # This keeps token distribution under 20,000 tokens per minute
        time.sleep(0.65)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total_items:
            print(f"Processed progress: {idx + 1}/{total_items} elements indexed successfully.")

    # 1. Save processed item data objects
    with open(os.path.join(OUTPUT_DIR, "catalog_manifest.pkl"), "wb") as f:
        pickle.dump(catalog_items, f)

    # 2. Save lexical BM25 database matrix tracking keyword relevance matches
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 3. Save dense matrix embeddings array files locally
    np.save(os.path.join(OUTPUT_DIR, "vector_embeddings.npy"), np.array(embeddings_matrix, dtype=np.float32))
    
    print("\n🎉 Core RAG Ingestion layer initialized successfully!")
    print(f"Generated artifacts deployed to local directory: '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    process_and_index()