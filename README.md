# 🏥 MediAssist — Healthcare AI Assistant

> **RAG-powered healthcare AI built for MediCare Health System**
> Llama 3 (local, GPU) · ChromaDB · FastAPI · sentence-transformers · Docker

---

## 🎥 Demo Video
[Watch Demo](https://drive.google.com/file/d/1FOZv3SLKTTni33hibIOyvGMfVhTQxa8j/view?usp=drive_link) 

---

## What This Is

MediAssist is a production-grade prototype AI assistant that answers patient questions grounded in healthcare policy documents. It uses a full Retrieval-Augmented Generation (RAG) pipeline with a locally-running LLM (Llama 3 via Ollama), an agentic router, and a clean chat UI — running entirely on-premise with zero external AI API calls. No patient data ever leaves the machine.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Agent Router (agent.py)            │
│                                                 │
│  ┌─────────────┐  ┌───────────┐  ┌──────────┐  │
│  │  Emergency  │  │Appointment│  │Knowledge │  │
│  │  Handler    │  │ Tool      │  │  (RAG)   │  │
│  └──────┬──────┘  └─────┬─────┘  └────┬─────┘  │
└─────────┼───────────────┼─────────────┼────────┘
          │               │             │
          ▼               ▼             ▼
    Immediate       Mock Slot      ChromaDB
    Safety          Checker        Retrieval
    Response        Tool           (Top-K chunks)
                                       │
                                       ▼
                               Ollama (Llama 3)
                               Local GPU Inference
                               RTX 3050 · ~3-6s
                                       │
                                       ▼
                               Grounded Answer
                               + Source Citations
                               + Confidence Score
```

**Document Ingestion Flow:**
```
data/*.txt → Chunking (600 chars) → Embeddings (all-MiniLM-L6-v2) → ChromaDB
```

---

## Tech Stack

| Component       | Tool                                   | Reason                                          |
|-----------------|----------------------------------------|-------------------------------------------------|
| LLM             | Ollama · Llama 3 8B (local GPU)        | On-premise, no PHI leakage, bonus points        |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 | Free, fast, no API key, 384-dim vectors         |
| Vector Database | ChromaDB (persistent)                  | Simple, file-based, no extra service needed     |
| Backend         | FastAPI + Uvicorn                      | Async, auto-docs, production-ready              |
| Agent           | Custom router logic                    | Lightweight, full control, no overhead          |
| Frontend        | Vanilla HTML/CSS/JS                    | Zero dependencies, instant load, clean UI       |
| Containers      | Docker + docker-compose                | Reproducible, demo-ready, production-grade      |

---

## Project Structure

```
healthcare-ai-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, all endpoints
│   ├── rag.py           # Document ingestion, chunking, ChromaDB retrieval
│   ├── embeddings.py    # sentence-transformers wrapper
│   ├── llm.py           # Ollama integration + prompt engineering
│   ├── agent.py         # Query router + appointment mock tool
│   └── config.py        # All configuration (env-driven)
├── data/
│   ├── telehealth_policy.txt
│   ├── medication_refill_policy.txt
│   ├── patient_discharge_instructions.txt
│   ├── hipaa_privacy_guidelines.txt
│   ├── insurance_eligibility_faq.txt
│   └── appointment_scheduling_policy.txt
├── static/
│   └── index.html       # Chat UI frontend
├── vector_store/        # ChromaDB persisted data (auto-created)
├── tests/
│   ├── conftest.py
│   └── test_app.py
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup & Run

### ✅ Option 1: Docker Compose (Recommended)

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Ollama](https://ollama.com) installed locally with Llama 3 pulled

**Step 1: Pull Llama 3**
```bash
ollama pull llama3
```

**Step 2: Clone and configure**
```bash
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant
cp .env.example .env
```

**Step 3: Start with Docker Compose**
```bash
docker-compose up --build
```

**Step 4: Ingest documents**
```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{}"
```

**Step 5: Open the UI**
```
http://localhost:8000
```

---

### Option 2: Local (without Docker)

**Prerequisites:** Python 3.11+, Ollama installed

```bash
# Clone
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant

# Install
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Configure
cp .env.example .env

# Start Ollama (separate terminal)
ollama serve

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Ingest
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{}"
```

Open: `http://localhost:8000`

---

## Environment Configuration (.env)

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB
CHROMA_PERSIST_DIR=./vector_store
CHROMA_COLLECTION=healthcare_docs

# Ingestion
DATA_DIR=./data
CHUNK_SIZE=600
CHUNK_OVERLAP=100

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.70

# LLM Generation
NUM_PREDICT=250

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## API Reference

### `POST /ingest`
Ingest all documents from `/data` into ChromaDB.
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" -d "{}"
```
```json
{
  "status": "success",
  "ingested_files": ["telehealth_policy.txt", "..."],
  "total_chunks": 74,
  "message": "Successfully ingested 6 documents into 74 chunks."
}
```

### `POST /ask`
Ask a healthcare question.
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"Can a patient request a medication refill through telehealth?\"}"
```
```json
{
  "question": "Can a patient request a medication refill through telehealth?",
  "answer": "Yes, you can refill your non-controlled substance medication during a scheduled telehealth consultation. Controlled substances including Schedule II medications cannot be refilled via telehealth.",
  "sources": [
    {
      "document": "telehealth_policy.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits...",
      "relevance_distance": 0.18
    }
  ],
  "confidence": "high",
  "query_type": "knowledge",
  "tool_used": "rag_pipeline",
  "response_time_ms": 4558
}
```

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "vector_store": { "total_chunks": 74, "status": "ready" },
  "ollama": { "ollama_running": true, "model_available": true, "configured_model": "llama3" },
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "version": "1.0.0"
}
```

---

## Sample Q&A

| Question | Answer Type | Response |
|----------|-------------|----------|
| Can I refill medication via telehealth? | ✅ RAG | Yes, non-controlled substances only |
| What documents for cashless hospitalization? | ✅ RAG | Lists Aadhaar, insurance card, TPA card... |
| Book cardiology appointment Monday | ✅ Appointment Tool | Shows available slots |
| Heart specialist slots tomorrow | ✅ Appointment Tool | Cardiology not available Tuesday, shows alternatives |
| I have chest pain and can't breathe | 🚨 Emergency | Call 112 immediately |
| What antibiotic for red rash? | ❌ Safe Refusal | Not in documents, contact doctor |
| What's the hospital Wi-Fi password? | ❌ Safe Refusal | Not in documents |

---

## Prompt Engineering Strategy

System prompt in `app/llm.py` is intentionally concise for faster inference:

```
You are MediAssist, a helpful healthcare assistant for MediCare Health System.
- Answer ONLY from provided context. No outside knowledge.
- Only say "I could not find this" if context has NO relevant info.
- Never diagnose or prescribe medication.
- For emergencies, direct to 112 or ER.
- Keep answers under 4 sentences.
```

Key decisions:
- `temperature: 0.15` — near-deterministic, factual answers
- `num_ctx: 1024` — sufficient for 3 chunks + question
- `num_predict: 250` — complete answers without runaway generation

---

## Agentic Workflow

Three-way router in `app/agent.py`:

```
Query → classify_query() → [EMERGENCY | APPOINTMENT | KNOWLEDGE]
                               ↓              ↓              ↓
                        emergency_      check_available_   rag_
                        handler()        slots()          pipeline
```

Priority order: **Emergency > Appointment > Knowledge**

`check_available_slots(department, day)` supports natural language:
- "heart specialist" → cardiology
- "tomorrow" → tuesday
- "child doctor" → pediatrics

---

## Dataset

All 6 documents are 100% synthetic — no real PHI.

| File | Content |
|------|---------|
| `telehealth_policy.txt` | Eligibility, scheduling, medication refills via virtual visits |
| `medication_refill_policy.txt` | Refill channels, controlled substance rules, chronic disease management |
| `patient_discharge_instructions.txt` | Post-hospital care, wound care, emergency signs |
| `hipaa_privacy_guidelines.txt` | Patient rights, PHI definition, breach notification |
| `insurance_eligibility_faq.txt` | Accepted plans, cashless process, PM-JAY, claim filing |
| `appointment_scheduling_policy.txt` | Booking channels, department schedules, cancellation policy |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Security & PHI Compliance

- No real PHI anywhere in this system
- LLM runs fully local — zero data sent to external APIs
- System prompt prevents medical diagnoses
- Production additions would include: JWT auth, rate limiting, audit logging, PHI detection on ingest, HTTPS/TLS

---

## Limitations & Future Improvements

| Limitation | Production Fix |
|------------|---------------|
| No authentication | JWT / OAuth2 |
| Mock appointment tool | Integrate real HMS/EHR |
| No query caching | Redis semantic cache |
| English only | Multilingual embeddings |
| No PHI detection on ingest | Microsoft Presidio scanner |
| Single-node ChromaDB | Weaviate / Pinecone for scale |

---

## Author

Built for Mindbowser AI Engineer Hackathon.