"""
embeddings.py
Handles sentence-transformer embedding generation.
"""

import logging
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model (loaded once at startup)."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully.")
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Generate an embedding for a single query string.

    Args:
        query: The query string.

    Returns:
        Embedding vector as list of floats.
    """
    model = get_embedding_model()
    embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    return embedding[0].tolist()
