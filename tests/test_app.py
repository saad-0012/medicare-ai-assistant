"""
tests/test_app.py
Unit and integration tests for MediAssist.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.agent import QueryType, classify_query, extract_appointment_details
from app.rag import split_into_chunks


# ── Test Client ────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ── Agent Tests ────────────────────────────────────────────────────────────────

class TestQueryClassification:
    def test_emergency_detection_chest_pain(self):
        assert classify_query("I have severe chest pain") == QueryType.EMERGENCY

    def test_emergency_detection_unconscious(self):
        assert classify_query("The patient is unconscious") == QueryType.EMERGENCY

    def test_appointment_detection_book(self):
        assert classify_query("I want to book a cardiology appointment") == QueryType.APPOINTMENT

    def test_appointment_detection_slots(self):
        assert classify_query("What slots are available on Monday?") == QueryType.APPOINTMENT

    def test_appointment_detection_schedule(self):
        assert classify_query("Schedule me for orthopedics on Thursday") == QueryType.APPOINTMENT

    def test_knowledge_detection_medication(self):
        assert classify_query("Can I refill my medication via telehealth?") == QueryType.KNOWLEDGE

    def test_knowledge_detection_hipaa(self):
        assert classify_query("What are my rights under HIPAA?") == QueryType.KNOWLEDGE

    def test_knowledge_detection_discharge(self):
        assert classify_query("What should I do after discharge?") == QueryType.KNOWLEDGE


class TestAppointmentExtraction:
    def test_extract_cardiology_monday(self):
        dept, day = extract_appointment_details("Book a cardiology appointment for Monday")
        assert dept == "cardiology"
        assert day == "monday"

    def test_extract_orthopedics_thursday(self):
        dept, day = extract_appointment_details("I need an orthopedics appointment on Thursday")
        assert dept == "orthopedics"
        assert day == "thursday"

    def test_extract_alias_heart(self):
        dept, day = extract_appointment_details("I need to see a heart specialist on Friday")
        assert dept == "cardiology"

    def test_extract_alias_child(self):
        dept, day = extract_appointment_details("Book appointment for my child on Wednesday")
        assert dept == "pediatrics"

    def test_default_fallback(self):
        dept, day = extract_appointment_details("I want to see a doctor")
        assert dept == "general medicine"
        assert day == "monday"


# ── Chunking Tests ─────────────────────────────────────────────────────────────

class TestChunking:
    def test_basic_chunking(self):
        text = "This is a test sentence. " * 100
        chunks = split_into_chunks(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_no_empty_chunks(self):
        text = "Hello world. " * 50
        chunks = split_into_chunks(text, chunk_size=100, overlap=20)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_overlap(self):
        text = "A" * 1000
        chunks = split_into_chunks(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        text = "This is a short document."
        chunks = split_into_chunks(text, chunk_size=500, overlap=80)
        assert len(chunks) == 1
        assert chunks[0] == text


# ── API Endpoint Tests ─────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "vector_store" in data
        assert "ollama" in data
        assert "embedding_model" in data


class TestAskEndpointValidation:
    def test_ask_empty_question_rejected(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_too_short_rejected(self, client):
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 422

    def test_ask_valid_format(self, client):
        # This may fail if Ollama is not running, but response format should be correct
        response = client.post("/ask", json={"question": "What is the telehealth policy?"})
        # 200 if Ollama running, 503 if not — both are valid test cases
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "confidence" in data
            assert "query_type" in data
            assert "tool_used" in data

    def test_emergency_routing(self, client):
        response = client.post("/ask", json={"question": "I have severe chest pain and cannot breathe"})
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "emergency"
        assert "112" in data["answer"]

    def test_appointment_routing(self, client):
        response = client.post("/ask", json={"question": "Book a cardiology appointment for Monday"})
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "appointment"
        assert data["tool_used"] == "check_available_slots"
