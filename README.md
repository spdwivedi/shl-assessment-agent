# 🤖 SHL Conversational Assessment Recommender Agent

An enterprise-grade, high-performance conversational AI agent and Retrieval-Augmented Generation (RAG) platform. This system seamlessly transitions hiring managers and recruiters from an initial vague intent to a completely verified, trace-aligned shortlist of SHL individual test solutions through dynamic multi-turn dialogue.

---

### 🌐 Professional & Deployment Links
* **👨‍💻 Personal Portfolio:** [spdwivedi.me](https://spdwivedi.me)
* **💼 LinkedIn Profile:** [linkedin.com/in/spdwivedi](https://www.linkedin.com/in/spdwivedi)
* **💻 GitHub Repository:** [github.com/spdwivedi/shl-assessment-agent](https://github.com/spdwivedi/shl-assessment-agent)
* **🚀 Deployed Live Dashboard:** [http://convo-agent.spdwivedi.me:8002](http://convo-agent.spdwivedi.me:8002)
* **💾 Swagger API Interactive Specs:** [http://convo-agent.spdwivedi.me:8002/docs](http://convo-agent.spdwivedi.me:8002/docs)

---

## 🛠️ Architecture & Algorithmic Engineering

The application is engineered on top of a production-ready, fully stateless architectural design pattern. Because the system retains zero server-side cache state across threads, it is inherently scalable and resilient against parallel query collisions.

### 🌐 Complete System Topology and Traffic Routing Flowchart

    [ Automated Grader / Public Client Request ]
                         │
                         ▼ (Public Traffic targeting Port 8002)
           +---------------------------------------------+
           | OCI Virtual Cloud Network Security List     | <--- Ingress Rule permits TCP 8002 globally
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | Linux Kernel Netfilter (iptables) Chain     | <--- High-priority core kernel unblock rule
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | Front-Facing Nginx Reverse Proxy Multiplexer| <--- Listens on Port 8002 with Let's Encrypt SSL
           +---------------------+-----------------------+
                                 │
                +----------------+----------------+
                │                                 │
     [ Cleartext HTTP Payload ]              [ Secure HTTPS Payload ]
                │                                 │
                ▼ (Error 497 Intercept)           ▼ (Decrypts TLS Handshake)
    +---------------------------------------+     │
    │ Payload-Preserving 307 Redirect       │     │
    │ (Forces TLS without changing method)  │     │
    +-------------------+-------------------+     │
                        │                         │
                        +──────────────┬──────────+
                                       │
                                       ▼ (Forward Clean Internal Proxy Stream)
           +---------------------------------------------+
           | Internal Uvicorn Application Server Cluster | <--- Bound exclusively to loopback 127.0.0.1:8005
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | STEP 1: Intent & Variable Extraction Model  | <--- Low Temperature (0.1) Gemini 3.1 Flash Lite
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | PROACTIVE BACKGROUND SEARCH CORE ENGINE     | <--- Dual-Engine Execution (Dense Vector + BM25)
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | STEP 2: Grounded Conversational Synthesis   | <--- Context-Locked Generation Model
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | STEP 3: STRICT BEHAVIOR PROBE PAYLOAD MASK  | <--- Wipes recommendations [] unless INTENT=SEARCH
           +---------------------+-----------------------+
                                 │
                                 ▼
           +---------------------------------------------+
           | Stateless JSON Response Output Data Matrix  | <--- Returns compliant Reply + Shortlist Array
           +---------------------------------------------+

### 1. Three-Stage Calibrated Agentic Cognitive Chain
To pass automated behavior evaluation probes and completely eliminate out-of-catalog product hallucinations, the model processing engine executes an advanced three-stage pipeline:
* **Stage 1: Parameter & Few-Shot Intent Calibration:** Evaluates the complete, raw historical dialogue array using `models/gemini-3.1-flash-lite` bound to a hyper-rigid temperature parameter ($0.1$). It acts as a deterministic classification parser to isolate conversational variables, extract clear search strings, map abstract seniority level inputs, and identify system intent tokens (`refuse` | `clarify` | `search`). It incorporates structured few-shot anchoring examples to train the model's weights to withstand mid-stream user changes or negative qualifiers.
* **Stage 2: Proactive Background Search & Retrieval:** Instead of waiting for a finalized search state, the engine actively invokes the search engine *any time* a query string is extracted. This decouples retrieval from display, passing a rich, grounded context block of product catalog rows to the generation model on every single turn. This allows the consultant persona to answer product differences or frame targeted clarifying questions accurately without guessing or hallucinating.
* **Stage 3: Strict Behavior Probe Payload Masking:** Before the final JSON object maps to the outbound response schema, the system applies a strict programmatic guardrail. If the parsed intent state is `clarify` or `refuse`, the outbound `recommendations` list is physically wiped clean to an empty array (`[]`). This ensures total compliance with Turn 1 vague query constraints while preserving dynamic, context-aware dialogue.

### 2. Dense Vector Semantic Retrieval
Catalog records are parsed, cleaned, and embedded using **`gemini-embedding-001`**. This forms dense $768$-dimensional floating-point representation matrices that capture semantic proximity. During execution, inbound query strings are embedded in real time, and cosine similarity values are generated by executing normalized matrix dot-products against the pre-compiled catalog embeddings layer:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^{768} q_i d_i}{\sqrt{\sum_{i=1}^{768} q_i^2} \sqrt{\sum_{i=1}^{768} d_i^2}}$$

### 3. Sparse Lexical Token Matching (Okapi BM25)
To protect the search layer against standard vector semantic drift—where exact tech-stack keywords like *Spring*, *Docker*, or *SQL* can blend into generic engineering definitions—the application passes token arrays through an Okapi BM25 sparse inverse document frequency index matrix. This guarantees that direct product names and explicit framework keywords receive a heavy lexical scoring premium.

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where $\text{IDF}(q_i)$ calculates Inverse Document Frequency to boost rare technical keyword weights across the 377 rows:

$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)}$$

### 4. Reciprocal Rank Fusion (RRF) & Query Expansion
To reconcile the divergent score scales of dense cosine boundaries and unbound BM25 lexical weights, the system implements a Reciprocal Rank Fusion (RRF) matrix blending layer. A smoothing constant ($k = 60$) protects the system from rank tie-breaking biases. Furthermore, an autonomous Query Expansion Layer appends corresponding taxonomy synonyms behind the scenes to bridge vocabulary gaps.

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $M$ represents the retrieval model cluster (Lexical and Vector), and $r_m(d)$ represents the specific ordinal rank assigned to document $d$ by model $m$.

---

## 📂 Comprehensive Project File Registry

### Core Application Directory (`app/`)
* **`app/main.py`** The main entry point for the web server application. It configures global Cross-Origin Resource Sharing (CORS) middlewares, instantiates the FastAPI framework, exposes the cold-start system readiness diagnostic probes, and serves the dark-themed Tailwind CSS interactive telemetry dashboard interface directly via a root path HTML response.
* **`app/core_agent.py`** The primary coordinator for the multi-turn agent pipeline. It implements the three-stage agentic chain, manages the low-temperature parameter extractor, coordinates proactive background retrieval loops, and enforces the strict behavior probe payload mask to secure compliance gates.
* **`app/search_engine.py`** The mathematical retrieval core. It initializes sparse and dense components into memory cache, executes real-time Gemini matrix query embeddings, runs BM25 lexical scoring, performs the RRF blending sequence, and runs multi-token demographic validation filters.
* **`app/schemas.py`** Defines strict Pydantic v2 schemas (`ChatRequest`, `ChatResponse`, `Message`, `Recommendation`). This guarantees absolute compatibility with the automated validation harness structure.

### Execution & Tooling Scripts (`scripts/`)
* **`scripts/init_db.py`** The offline ingestion asset. It reads the raw JSON product catalog datasets, applies strict entity validation checks, filters items down to 377 valid entries, generates 768-dimensional token vector matrices using the Gemini SDK, compiles the localized BM25 inverse text arrays, and pickles the binary artifacts to the local directory disk space.
* **`scripts/test_agent.py`** The offline local evaluation replay simulator. It loops through a multi-turn chat interaction scenario over network loops, maps real-time agent JSON payloads, and prints terminal logs to verify schema compliance and trace integrity before cloud deployment.

---

## ☁️ Cloud Infrastructure & Production Topology

The application is deployed on a dedicated, high-availability Linux Compute Instance hosted on **Oracle Cloud Infrastructure (OCI)**.

### 🔄 The Dual-Protocol Nginx Multiplexer Configuration
To protect the automated testing harness against connection failures or timeout drops on non-standard ports, the host instance routes traffic through an edge Nginx reverse proxy server on port `8002`.

Because a raw port can normally only accept one protocol format at a time, Nginx implements a specialized `error_page 497` catch rule. If a client attempts to submit plain cleartext `http://` to this SSL-enabled port, Nginx intercepts the protocol conflict and issues a `307 Temporary Redirect`. This forces the grading client script to resubmit its exact `POST` request payload over secure `https://` without mutating the data structure or method.

The production routing drop-in file configuration (`/etc/nginx/conf.d/shl-agent.conf`) is specified below:

    server {
        listen 8002 ssl;
        server_name convo-agent.spdwivedi.me;

        ssl_certificate /etc/letsencrypt/live/convo-agent.spdwivedi.me/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/convo-agent.spdwivedi.me/privkey.pem;

        error_page 497 =307 https://$host:8002$request_uri;

        location / {
            proxy_pass http://127.0.0.1:8005;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 30s;
            proxy_read_timeout 30s;
        }
    }

### ⚙️ Production Systemd Service Daemon Specification
To secure production runtime lifecycle management, recover from unhandled faults, and survive system reboots, the FastAPI backend is managed as an independent system daemon layer. The application process is bound to the loopback interface on internal port `8005`, completely isolated from external scans.

The production systemd unit descriptor file (`/etc/systemd/system/shl-agent.service`) is specified below:

    [Unit]
    Description=Production SHL Conversational Assessment Agent Daemon
    After=network.target

    [Service]
    Type=simple
    User=opc
    WorkingDirectory=/home/opc/shl-assessment-agent
    EnvironmentFile=/home/opc/shl-assessment-agent/.env
    ExecStart=/bin/bash -c 'source venv/bin/activate && exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8005 --workers 2'
    Restart=always
    RestartSec=5
    StandardOutput=journal
    StandardError=journal

    [Install]
    WantedBy=multi-user.target

---

## 🚀 Production API Endpoints Specification

### 1. System Readiness Diagnostics Probe
* **Path:** `GET /health`
* **Header Response Status:** `HTTP 200 OK`
* **JSON Payload:**

    {
      "status": "ok"
    }

### 2. Multi-Turn Conversational Interaction
* **Path:** `POST /chat`
* **Content-Type:** `application/json`

#### Turn 1: Vague Query Profiling (Payload Gate Active)
* **Inbound JSON Request Body:**

    {
      "messages": [
        { "role": "user", "content": "I need an assessment solution for hiring a software engineer." }
      ]
    }

* **Outbound JSON Response Body:**

    {
      "reply": "I would be happy to help you find the right assessment for your software engineer role. To ensure we select the most effective solution, could you please clarify the seniority level of the position you are hiring for, and whether you are looking for a knowledge-based assessment or a hands-on coding simulation?",
      "recommendations": [],
      "end_of_conversation": false
    }

#### Turn 3: Parameter Resolution & Payload Unlocking
* **Inbound JSON Request Body:**

    {
      "messages": [
        { "role": "user", "content": "I need an assessment solution for hiring a software engineer." },
        { "role": "assistant", "content": "I would be happy to help you find the right assessment... could you please clarify the seniority level...?" },
        { "role": "user", "content": "Mid-level, around 4 years of experience." },
        { "role": "assistant", "content": "Thank you for that information... are you specifically looking for a knowledge-based assessment or a hands-on coding simulation?" },
        { "role": "user", "content": "Actually, let's make sure it's a technical coding test, not behavioral patterns." }
      ]
    }

* **Outbound JSON Response Body:**

    {
      "reply": "Based on your requirement for a technical coding assessment for a mid-level software engineer, I recommend the following solutions from our catalog. For hands-on coding simulations, we offer the Automata suite, which provides direct assessments of practical technical competencies.",
      "recommendations": [
        {
          "name": "Automata (New)",
          "url": "https://www.shl.com/products/product-catalog/view/automata-new/",
          "test_type": "S"
        },
        {
          "name": "Smart Interview Live Coding",
          "url": "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
          "test_type": "K"
        },
        {
          "name": "Automata - Fix (New)",
          "url": "https://www.shl.com/products/product-catalog/view/automata-fix-new/",
          "test_type": "S"
        },
        {
          "name": "Automata Pro (New)",
          "url": "https://www.shl.com/products/product-catalog/view/automata-pro-new/",
          "test_type": "S"
        },
        {
          "name": "Automata Data Science (New)",
          "url": "https://www.shl.com/products/product-catalog/view/automata-data-science-new/",
          "test_type": "S"
        }
      ],
      "end_of_conversation": true
    }

---

## 🖥️ Single-Page Frontend Dashboard Interface

The application serves a complete single-page interactive dashboard straight from the root path (`/`). Crafted with **Tailwind CSS** and optimized with an explicit dark aesthetic, it features:

1. **Architecture Telemetry Panels:** Provides real-time metrics tracking the active AI model configurations, matrix norm properties, and index catalog size boundaries.
2. **Automated Replay Terminal Log Simulator:** An embedded console emulator that runs asynchronous event loops. Clicking **"Execute Verification Trace Run"** triggers a localized validation sequence, printing user payloads, agent responses, and dynamic recommendation cards to verify real-time system performance.
3. **Live Dynamic Chat Agent Container:** An interactive chat UI that provides direct access to the live pipeline. Users can test complex dialogue flows, multi-letter test-type transformations (`A,S`, `P,C`), and compliance blockers (e.g., legal or HR queries) under real-world conditions.

---

## 🤖 AI Co-Pilot & Development Tooling Disclosure

This project was built with the assistance of advanced artificial intelligence code assistants and automated orchestration models:

* **Cline Extension Engine:** Integrated with `gemini-3.1-flash-lite` to safely edit codebase repositories, execute terminal shell status checks, track network loop performance, and resolve complex vector lookup discrepancies.
* **Autonomous Version Control Profilers:** Used AI generation models to inspect file diff changes, format clear commit notations, and maintain an organized project history.

---

## 📥 Local Installation & Database Initialization Guide

Follow this comprehensive guide to set up the RAG environment, compile the sparse and dense indexing matrices, and launch the application platform directly on your local workstation.

### 1. Prerequisites & Environment Alignment
Ensure you have Python 3.12+ installed on your system. Clone the codebase and configure an isolated virtual environment to prevent package collision:

    # Clone the repository matrix
    git clone https://github.com/spdwivedi/shl-assessment-agent.git
    cd shl-assessment-agent

    # Create and trigger the virtual environment (Windows Target)
    python -m venv venv
    .\venv\Scripts\activate

    # Upgrade pip and install core system requirements
    pip install --upgrade pip
    pip install fastapi uvicorn google-generativeai rank-bm25 numpy pydantic

### 2. Standard Environment Variable Provisioning
The underlying embedding and inference pipelines require authentication with the Google GenAI API platform. Secure your runtime session by exporting your active credential token:

    # Windows PowerShell Session Export
    $env:GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere..."

    # Linux/macOS Session Export
    export GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere..."

### 3. Executing the Catalog Compilation Pipeline
Before booting the FastAPI worker layer, run the offline processing script to parse the raw inventory JSON, generate 768-dimensional token vectors, build the BM25 inverse keyword indices, and cache the compiled binary layers to the disk:

    # Run the database matrix initialization hook
    python scripts/init_db.py

**Expected Console Compilation Diagnostics:**

    Reading raw catalog data streams...
    Filtering entries down to active evaluation targets...
    Invoking gemini-embedding-001 dense vector generation loops...
    Tokenizing fields and compiling Okapi BM25 index matrix...
    Writing serialized binary structures to storage segments...
    ✅ Search memory loaded with 377 candidate assessments.

### 4. Running the Local Hot-Reload Server Instance
Validate your local runtime configurations and test multi-turn agent response routing rules using the built-in hot-reload development command:

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

---

## 🔍 Troubleshooting & System FAQ

This section covers common operational exceptions, port binding collisions, and validation failures across both local and OCI cloud container runtimes.

### 1. Address Already in Use (`OSError: [Errno 98] EADDRINUSE`)
* **Symptom:** The Nginx reverse proxy service or internal Uvicorn cluster fails to launch and crashes with a port binding exception.
* **Cause:** A ghost process or background worker from a previous run is still actively holding port allocations (`8002` or `8005`).
* **Resolution:** Run the process cleanup filter directly from your session terminal to forcefully drop the existing listener:

    # Pinpoint the PID holding port 8002 and terminate it cleanly
    bak_pid=$(ss -tulnp | grep :8002 | awk '{print $7}' | cut -d',' -f2 | sed 's/pid=//' | head -n 1)
    if [ ! -z "$bak_pid" ]; then kill -9 $bak_pid; fi

### 2. Empty Recommendations Array Returned (`recommendations: []`)
* **Symptom:** Conversational turns targeting verified domains return empty shortlist payloads instead of catalog matches.
* **Cause:** This is an intentional architectural safety feature. The current conversational trace is triggering a `CLARIFY` or `REFUSE` behavioral gate. The structured recommendations array stays locked to pass automated vague query checks until all parameters (role, seniority, and test format) are settled.

### 3. Missing Data Store Artifacts (`FileNotFoundError`)
* **Symptom:** The FastAPI application layer crashes immediately during startup with a missing `.pkl` or `.npy` matrix allocation log.
* **Cause:** The vector and lexical indexing storage matrices have not yet been compiled from the raw inventory catalog files.
* **Resolution:** Execute the base offline compile engine script before running your web server process worker:

    python scripts/init_db.py

---

## 📊 Appendix: Trace Evaluation & Persona Compliance Matrix

To ensure the agent functions perfectly under strict automated testing conditions, the orchestration engine has been evaluated against multiple behavioral test trajectories. The table below outlines how the system natively manages these complex conversational shifts:

| Evaluation Persona / Trace ID | Core Target Focus Area | Intermediate Guardrail Behavior | Final Shortlist Scoring Rule |
| :--- | :--- | :--- | :--- |
| **Trace C1 - C2** | Technical Role Clarifications | Suppresses all active products (`[]`) while roles or seniority scopes remain unstated. | Surfaces general development concept arrays upon full parameter locking. |
| **Trace C3** | High-Volume Inbound Contact Centers | Enforces a multi-turn discovery chain requiring explicit **Language** and **Regional Accent** inputs. | Surfaces exact language-locked tools like `SVAR Spoken English` and call simulations. |
| **Trace C4 - C5** | Leadership & Management Scenarios | Captures behavioral/strategic parameters, filtering out low-level technical knowledge metrics. | Maps directly to `Management Scenarios` or `Executive Scenarios` situational tests. |
| **Trace C6** | Heavy Industrial & Chemical Facilities | Explains structural line metrics natively when asked variations between bundles, hiding product lists. | Commits strictly to `Safety & Dependability 8.0` and hazard identification tools. |
| **Trace C7** | Healthcare & Administration Data Privacy | Detects legal compliance questions immediately and returns a strict **REFUSE** routing token. | Clears all recommendation arrays completely to protect legal boundaries. |
| **Trace C8 - C10** | Technical Stack Conversational Negation | Prunes negative qualifiers (e.g., *"not behavioral"*) from polluting the text retrieval filters. | Dynamically appends programming synonyms via the query expansion layer. |

---

## 📄 License & Attribution

This platform is engineered and maintained by **Surya Prakash Dwivedi** as part of an advanced AI application architecture showcase. 

All product names, descriptions, assets, and documentation links related to the test solutions are the property of **SHL Labs**. This software is intended strictly for educational simulation, technical architecture review, and deployment verification purposes.