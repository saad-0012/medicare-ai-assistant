import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Configuration ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── Embedding Configuration ────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Vector Store Configuration ─────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vector_store")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "healthcare_docs")

# ── Document Ingestion Settings ────────────────────────────────────────────────
DATA_DIR: str = os.getenv("DATA_DIR", "./data")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

# ── RAG Retrieval Settings ─────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# ── API Settings ───────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Appointment Mock Data ──────────────────────────────────────────────────────
MOCK_SLOTS: dict = {
    "cardiology": {
        "monday": ["10:00 AM", "11:00 AM", "12:00 PM"],
        "wednesday": ["10:30 AM", "11:30 AM"],
        "friday": ["10:00 AM", "12:00 PM"],
    },
    "general medicine": {
        "monday": ["9:00 AM", "10:00 AM", "2:00 PM", "4:00 PM"],
        "tuesday": ["9:30 AM", "11:00 AM", "3:00 PM"],
        "wednesday": ["10:00 AM", "1:00 PM", "5:00 PM"],
        "thursday": ["9:00 AM", "2:30 PM", "4:00 PM"],
        "friday": ["11:00 AM", "2:00 PM", "3:30 PM"],
        "saturday": ["9:00 AM", "10:30 AM"],
    },
    "orthopedics": {
        "tuesday": ["9:00 AM", "10:00 AM", "11:00 AM"],
        "thursday": ["9:30 AM", "10:30 AM", "12:00 PM"],
    },
    "dermatology": {
        "tuesday": ["10:00 AM", "11:30 AM", "2:00 PM"],
        "thursday": ["10:30 AM", "12:00 PM", "3:00 PM"],
        "saturday": ["10:00 AM", "11:00 AM", "1:00 PM"],
    },
    "pediatrics": {
        "monday": ["9:00 AM", "10:00 AM", "11:00 AM", "3:00 PM"],
        "tuesday": ["9:30 AM", "10:30 AM", "2:00 PM"],
        "wednesday": ["9:00 AM", "11:00 AM", "4:00 PM"],
        "thursday": ["9:30 AM", "10:00 AM", "3:30 PM"],
        "friday": ["10:00 AM", "11:30 AM", "2:30 PM"],
    },
    "endocrinology": {
        "monday": ["11:00 AM", "12:00 PM", "2:00 PM"],
        "wednesday": ["11:30 AM", "1:00 PM"],
        "friday": ["11:00 AM", "2:00 PM"],
    },
    "mental health": {
        "monday": ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"],
        "tuesday": ["10:00 AM", "1:00 PM", "3:00 PM"],
        "wednesday": ["9:30 AM", "11:30 AM", "2:30 PM"],
        "thursday": ["10:00 AM", "12:00 PM", "4:00 PM"],
        "friday": ["9:00 AM", "11:00 AM", "3:30 PM"],
    },
    "neurology": {
        "monday": ["10:00 AM", "11:00 AM", "12:00 PM"],
        "thursday": ["10:30 AM", "11:30 AM"],
    },
    "gynecology": {
        "monday": ["9:00 AM", "10:30 AM", "2:00 PM"],
        "tuesday": ["9:30 AM", "11:00 AM", "3:00 PM"],
        "wednesday": ["10:00 AM", "12:00 PM", "4:00 PM"],
        "thursday": ["9:00 AM", "11:30 AM"],
        "friday": ["10:30 AM", "2:30 PM"],
        "saturday": ["9:00 AM", "11:00 AM"],
    },
}
