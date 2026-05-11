"""
rag.py
Core RAG pipeline: document ingestion, chunking, vector storage, and retrieval.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)
from app.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)

# ── ChromaDB Client ────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Text Chunking ──────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by word boundary.

    Args:
        text: Full document text.
        chunk_size: Target number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunk strings.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to end at a sentence boundary
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ── Document Ingestion ─────────────────────────────────────────────────────────

def ingest_documents(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """
    Read all documents from data_dir, chunk them, embed them, and store in ChromaDB.

    Args:
        data_dir: Directory containing source documents.

    Returns:
        Summary dict with ingested file count and total chunks.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    supported_extensions = {".txt", ".md", ".pdf"}
    files = [f for f in data_path.iterdir() if f.suffix.lower() in supported_extensions]

    if not files:
        raise ValueError(f"No supported documents found in {data_dir}")

    client = get_chroma_client()
    collection = get_collection(client)

    # Clear existing documents for clean re-ingestion
    existing = collection.count()
    if existing > 0:
        logger.info(f"Clearing {existing} existing chunks before re-ingestion.")
        client.delete_collection(CHROMA_COLLECTION)
        collection = get_collection(client)

    total_chunks = 0
    ingested_files = []

    for file_path in files:
        try:
            logger.info(f"Ingesting: {file_path.name}")
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = split_into_chunks(text)

            if not chunks:
                logger.warning(f"No chunks generated for {file_path.name}, skipping.")
                continue

            embeddings = embed_texts(chunks)
            ids = [f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": file_path.name,
                    "chunk_index": i,
                    "file_stem": file_path.stem,
                }
                for i in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            ingested_files.append(file_path.name)
            logger.info(f"  → {len(chunks)} chunks stored for {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {e}", exc_info=True)

    logger.info(f"Ingestion complete. Files: {len(ingested_files)}, Chunks: {total_chunks}")
    return {
        "ingested_files": ingested_files,
        "total_chunks": total_chunks,
        "status": "success",
    }


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    top_k: int = TOP_K_RESULTS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query: The user's question.
        top_k: Number of top chunks to retrieve.
        threshold: Maximum cosine distance allowed (lower = more similar).

    Returns:
        Tuple of (list of chunk texts, list of metadata dicts).
    """
    client = get_chroma_client()
    collection = get_collection(client)

    if collection.count() == 0:
        logger.warning("Vector store is empty. Please run ingestion first.")
        return [], []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Filter by similarity threshold (cosine distance: 0=identical, 2=opposite)
    filtered_chunks = []
    filtered_meta = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        if dist <= threshold:
            filtered_chunks.append(chunk)
            filtered_meta.append({**meta, "distance": round(dist, 4)})
        else:
            logger.debug(f"Chunk filtered out (distance {dist:.4f} > threshold {threshold}): {chunk[:60]}...")

    logger.info(f"Retrieved {len(filtered_chunks)}/{len(chunks)} chunks within threshold for query.")
    return filtered_chunks, filtered_meta


def get_store_stats() -> Dict[str, Any]:
    """Return stats about the current vector store."""
    try:
        client = get_chroma_client()
        collection = get_collection(client)
        count = collection.count()
        return {"total_chunks": count, "collection": CHROMA_COLLECTION, "status": "ready" if count > 0 else "empty"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
