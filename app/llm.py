"""
llm.py
LLM integration via Ollama (local) with structured prompt engineering.
"""

import logging
from typing import List, Optional

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────────
# This prompt enforces grounded, safe, and professional responses.
SYSTEM_PROMPT = """You are MediAssist, a professional AI healthcare assistant for MediCare Health System.

Your role is to answer patient questions clearly and accurately using ONLY the provided context documents.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using information explicitly present in the provided context.
2. If the context does not contain enough information to answer the question, respond with exactly:
   "I could not find this information in the provided documents. Please contact MediCare Health System directly at 1800-MED-CARE for assistance."
3. Do NOT guess, infer, or fabricate information not present in the context.
4. Do NOT provide personal medical diagnoses, drug prescriptions, or medical advice beyond what is stated in the documents.
5. Always maintain a professional, empathetic, and clear tone.
6. If a question involves a medical emergency, always advise the patient to call 112 or visit the nearest emergency room immediately.
7. Be concise but complete. Structure your answer clearly.
8. Do not repeat the question back or mention "based on the context provided" — just answer directly.

You serve patients who may be anxious or unfamiliar with medical processes. Be warm, clear, and helpful."""


def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Build the RAG prompt by combining retrieved context with the user question.

    Args:
        question: The user's question.
        context_chunks: List of retrieved document chunks.

    Returns:
        Formatted prompt string.
    """
    context_block = "\n\n---\n\n".join(
        [f"[Document Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    return f"""CONTEXT FROM KNOWLEDGE BASE:
{context_block}

PATIENT QUESTION:
{question}

ANSWER:"""


async def generate_answer(
    question: str,
    context_chunks: List[str],
    model: Optional[str] = None,
) -> str:
    """
    Generate a grounded answer using Ollama.

    Args:
        question: The user's question.
        context_chunks: Retrieved document chunks.
        model: Override model name (defaults to config).

    Returns:
        Generated answer string.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    selected_model = model or OLLAMA_MODEL

    if not context_chunks:
        logger.info("No context chunks available — returning not-found response.")
        return (
            "I could not find this information in the provided documents. "
            "Please contact MediCare Health System directly at 1800-MED-CARE for assistance."
        )

    prompt = build_rag_prompt(question, context_chunks)
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,      # Low temperature for factual, consistent answers
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    logger.info(f"Sending request to Ollama [{selected_model}] with {len(context_chunks)} context chunks.")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["message"]["content"].strip()
            logger.info(f"LLM response received ({len(answer)} chars).")
            return answer
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Is Ollama running?")
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Please start Ollama with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")
            raise RuntimeError(f"Ollama returned an error: {e.response.status_code}")
        except KeyError as e:
            logger.error(f"Unexpected Ollama response format: {data}")
            raise RuntimeError(f"Unexpected response from Ollama: missing key {e}")


async def check_ollama_health() -> dict:
    """Check if Ollama is running and the configured model is available."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Check if Ollama is running
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models_data = resp.json()
            available_models = [m["name"] for m in models_data.get("models", [])]
            model_available = any(OLLAMA_MODEL in m for m in available_models)
            return {
                "ollama_running": True,
                "configured_model": OLLAMA_MODEL,
                "model_available": model_available,
                "available_models": available_models,
            }
        except Exception as e:
            return {
                "ollama_running": False,
                "configured_model": OLLAMA_MODEL,
                "model_available": False,
                "error": str(e),
            }
