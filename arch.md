D:\Project_Files\SHL\Code\
│
├── .clinerules                 <-- Instructions that govern Cline's automation
├── CHANGELOG_AI.md             <-- The automated file modifications log
├── requirements.txt            <-- Project python dependencies
├── shl_product_catalog.json    <-- Copy your catalog file here
│
├── app/                        <-- Main FastAPI application package
│   ├── __init__.py
│   ├── main.py                 <-- API endpoints (/health and /chat)
│   ├── schemas.py              <-- Strict Pydantic v2 data verification models
│   ├── core_agent.py           <-- Multi-turn routing logic and guardrails
│   └── search_engine.py        <-- Lexical (BM25) + Dense Hybrid Retrieval
│
├── data_store/                 <-- Embedded database indices (Generated later)
│   ├── bm25_index.pkl
│   └── vector_embeddings.npy
│
└── scripts/                    <-- Automation, database seeding, and hooks
    ├── init_db.py              <-- Catalog chunking and embedding generator
    └── git_commit_agent.py     <-- Parallel background git hook processor