"""
llm.py
"""

import logging
from typing import List, Optional

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MediAssist, a professional AI healthcare assistant.
Answer the patient's question using ONLY the provided CONTEXT.
If the context does not contain the answer, respond exactly with: "I could not find this information in the provided documents. Please contact MediCare Health System directly at 1800-MED-CARE for assistance."
Do not guess or fabricate information.
NEVER start your answer with phrases like "According to the context", "Based on the documents", or "The provided text states". Answer directly."""


def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    context_block = "\n\n---\n\n".join(
        [f"[Document Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    return f"""CONTEXT FROM KNOWLEDGE BASE:
{context_block}

PATIENT QUESTION:
{question}

Direct Answer (without mentioning the context):"""


async def generate_answer(
    question: str,
    context_chunks: List[str],
    model: Optional[str] = None,
) -> str:
    if not context_chunks:
        return (
            "I could not find this information in the provided documents. "
            "Please contact MediCare Health System directly at 1800-MED-CARE for assistance."
        )

    selected_model = model or OLLAMA_MODEL
    prompt = build_rag_prompt(question, context_chunks)

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 256,
        },
    }

    logger.info(f"Ollama [{selected_model}] | chunks={len(context_chunks)}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Run: ollama serve")
        except httpx.ReadTimeout:
            raise RuntimeError(f"Ollama timed out. Check if {selected_model} is running properly.")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama error: {e.response.status_code}")
        except KeyError:
            raise RuntimeError("Unexpected response format from Ollama.")


async def check_ollama_health() -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models_data = resp.json()
            available_models = [m["name"] for m in models_data.get("models", [])]
            model_available = any(OLLAMA_MODEL in m for m in available_models)
            return {
                "ollama_running":   True,
                "configured_model": OLLAMA_MODEL,
                "model_available":  model_available,
                "available_models": available_models,
            }
        except Exception as e:
            return {
                "ollama_running":   False,
                "configured_model": OLLAMA_MODEL,
                "model_available":  False,
                "error": str(e),
            }