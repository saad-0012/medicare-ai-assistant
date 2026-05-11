# 🏥 MediAssist — Healthcare AI Assistant

> **RAG-powered healthcare AI built for MediCare Health System**
> Built with Ollama (Llama 3) · ChromaDB · FastAPI · sentence-transformers

---

## What This Is

MediAssist is a production-grade prototype AI assistant that answers patient questions grounded in healthcare policy documents. It uses a full Retrieval-Augmented Generation (RAG) pipeline with a locally-running LLM, an agentic router, and a clean chat UI — all running entirely on-premise with no external AI API calls.

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
                               Local LLM Generation
                                       │
                                       ▼
                               Grounded Answer
                               + Source Citations
                               + Confidence Score
```

**Document Ingestion Flow:**
```
data/*.txt  →  Text Chunking  →  Embeddings (all-MiniLM-L6-v2)  →  ChromaDB
```

---

## Tech Stack

| Component         | Tool / Library                          | Reason                                    |
|-------------------|-----------------------------------------|-------------------------------------------|
| LLM               | Ollama · Llama 3 (local)                | On-premise, bonus points, no API cost     |
| Embeddings        | sentence-transformers/all-MiniLM-L6-v2  | Free, fast, no API key, 384-dim vectors   |
| Vector Database   | ChromaDB (persistent)                   | Simple, file-based, no extra service      |
| Backend           | FastAPI + Uvicorn                       | Async, auto-docs, production-ready        |
| Agent Framework   | Custom router logic                     | Lightweight, full control, no overhead    |
| Frontend          | Vanilla HTML/CSS/JS                     | Zero deps, instant load, clean UI         |
| Containerization  | Docker + docker-compose                 | Full bonus, reproducible, demo-ready      |

---

## Project Structure

```
healthcare-ai-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, all endpoints
│   ├── rag.py           # Document ingestion, chunking, ChromaDB retrieval
│   ├── embeddings.py    # sentence-transformers wrapper
│   ├── llm.py           # Ollama integration + system prompt
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
│   └── test_app.py      # Unit + integration tests
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup & Run

### Option 1: Local (Recommended for Demo)

**Prerequisites:**
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

**Step 1: Clone and install**
```bash
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2: Start Ollama and pull Llama 3**
```bash
ollama serve                   # In a separate terminal (if not already running)
ollama pull llama3
```

**Step 3: Configure environment**
```bash
cp .env.example .env
# .env is pre-configured for local use — no changes needed
```

**Step 4: Start the server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 5: Ingest documents**
```bash
curl -X POST http://localhost:8000/ingest
```

**Step 6: Open the UI**
```
http://localhost:8000
```

---

### Option 2: Docker Compose (Full Bonus)

```bash
docker-compose up --build
```

This starts **Ollama + MediAssist API** together. Ollama will pull Llama 3 automatically on first start (~4GB, takes a few minutes).

Then ingest documents:
```bash
curl -X POST http://localhost:8000/ingest
```

Open: `http://localhost:8000`

---

## API Reference

### `POST /ingest`

Ingest documents from the `/data` folder into ChromaDB.

```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{}'
```

**Response:**
```json
{
  "status": "success",
  "ingested_files": [
    "telehealth_policy.txt",
    "medication_refill_policy.txt",
    "patient_discharge_instructions.txt",
    "hipaa_privacy_guidelines.txt",
    "insurance_eligibility_faq.txt",
    "appointment_scheduling_policy.txt"
  ],
  "total_chunks": 87,
  "message": "Successfully ingested 6 documents into 87 chunks."
}
```

---

### `POST /ask`

Ask a healthcare question.

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Can a patient request a medication refill through telehealth?"}'
```

**Response:**
```json
{
  "question": "Can a patient request a medication refill through telehealth?",
  "answer": "Yes, patients can request medication refills through telehealth if the medication is already prescribed by a MediCare physician and does not require an in-person evaluation. Controlled substances including Schedule II medications cannot be refilled via telehealth and require an in-person consultation.",
  "sources": [
    {
      "document": "telehealth_policy.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits under the following conditions: The medication is already prescribed by a MediCare physician...",
      "relevance_distance": 0.1823
    }
  ],
  "confidence": "high",
  "query_type": "knowledge",
  "tool_used": "rag_pipeline",
  "route_reason": "General knowledge question — routed to RAG pipeline.",
  "response_time_ms": 1842
}
```

---

### `GET /health`

Check system status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store": {
    "total_chunks": 87,
    "collection": "healthcare_docs",
    "status": "ready"
  },
  "ollama": {
    "ollama_running": true,
    "configured_model": "llama3",
    "model_available": true,
    "available_models": ["llama3:latest"]
  },
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "version": "1.0.0"
}
```

---

## Sample Questions & Answers

| Question | Type | Routed To |
|----------|------|-----------|
| "Can I refill my medication via telehealth?" | Knowledge | RAG pipeline |
| "Book a cardiology appointment for Monday" | Appointment | `check_available_slots` tool |
| "I have chest pain and can't breathe" | Emergency | Emergency handler |
| "What are my rights under HIPAA?" | Knowledge | RAG pipeline |
| "What documents do I need for cashless hospitalization?" | Knowledge | RAG pipeline |
| "What can't be covered by insurance?" | Knowledge | RAG pipeline |
| "What should I do after being discharged?" | Knowledge | RAG pipeline |

**Unknown information example:**
```json
{
  "question": "What is the hospital's Wi-Fi password?",
  "answer": "I could not find this information in the provided documents. Please contact MediCare Health System directly at 1800-MED-CARE for assistance.",
  "confidence": "none"
}
```

---

## Prompt Engineering Strategy

The system prompt in `app/llm.py` enforces:

```
You are MediAssist, a professional AI healthcare assistant for MediCare Health System.

STRICT RULES:
1. Answer ONLY using information explicitly present in the provided context.
2. If information is not in context, respond: "I could not find this information..."
3. Do NOT guess, infer, or fabricate information.
4. Do NOT provide personal medical diagnoses or drug prescriptions.
5. For emergencies, always direct to 112 or nearest Emergency Room.
6. Maintain professional, empathetic, clear tone.
```

**LLM settings:** `temperature=0.1` for maximum factual consistency.

---

## Agentic Workflow

The agent in `app/agent.py` implements a **three-way router**:

```
Query → classify_query() → [EMERGENCY | APPOINTMENT | KNOWLEDGE]
                               ↓              ↓              ↓
                       emergency_       check_available_  rag_
                       handler()        slots()          pipeline
```

**Priority order:** Emergency > Appointment > Knowledge

The `check_available_slots(department, day)` tool supports:
- Natural language department aliases: "heart" → cardiology, "child" → pediatrics
- Day normalization: "today" → monday, "tomorrow" → tuesday, etc.
- Graceful fallback if department/day not found

---

## Dataset

All documents are **100% synthetic** — no real patient data or PHI.

| File | Content |
|------|---------|
| `telehealth_policy.txt` | Telehealth eligibility, scheduling, medication refills via virtual visits |
| `medication_refill_policy.txt` | Refill channels, controlled substance rules, chronic disease management |
| `patient_discharge_instructions.txt` | Post-hospitalization care, wound care, activity restrictions, emergency signs |
| `hipaa_privacy_guidelines.txt` | Patient rights, PHI definition, security measures, breach notification |
| `insurance_eligibility_faq.txt` | Accepted plans, cashless hospitalization, claim process, PM-JAY |
| `appointment_scheduling_policy.txt` | Booking channels, department schedules, rescheduling/cancellation policy |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: query classification, appointment extraction, text chunking, API endpoint validation, routing logic.

---

## Security & PHI Compliance Notes

- **No real PHI** is used anywhere in this system.
- All documents are synthetic.
- The system prompt explicitly prevents the LLM from giving medical diagnoses.
- In production, the following would be added:
  - JWT authentication on API endpoints
  - Rate limiting per user/IP
  - Audit logging for all queries
  - PHI detection layer before document ingestion
  - HTTPS/TLS termination at reverse proxy
  - Encryption at rest for ChromaDB storage

---

## Limitations & Future Improvements

| Limitation | Production Fix |
|------------|---------------|
| No authentication | Add JWT / OAuth2 |
| Single-node ChromaDB | Migrate to Weaviate or Pinecone for scale |
| Mock appointment tool | Integrate with real HMS/EHR system |
| No query caching | Add Redis semantic cache |
| English-only | Add multilingual embeddings |
| No feedback loop | Add thumbs up/down + fine-tuning pipeline |
| Document re-ingestion clears all data | Add incremental ingestion with document hashing |
| No PHI detection | Add Presidio PII/PHI scanner on ingest |

---

## Author

Built for Mindbowser AI Engineer Hackathon Assignment.
