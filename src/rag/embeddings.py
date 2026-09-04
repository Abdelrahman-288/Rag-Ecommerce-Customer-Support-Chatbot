"""Embedding generation for the RAG knowledge base.

Uses sentence-transformers/all-MiniLM-L6-v2 to convert each knowledge-base
instruction into a dense vector for semantic similarity search. This is a
genuine deep-learning workload (a transformer encoder), so it uses
get_device() — unlike the TF-IDF components elsewhere in this project.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from configs.config import EMBEDDING_MODEL_NAME
from src.utils.device import get_device

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = get_device()
        logger.info("Loading embedding model %s on %s", EMBEDDING_MODEL_NAME, device)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts, returning a (n_texts, embedding_dim) float32 array.

    Embeddings are L2-normalized so that FAISS inner-product search is
    equivalent to cosine similarity — this is the standard, simplest way to
    do cosine similarity search in FAISS without a separate normalization
    step at query time.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_texts = [
        "where is my order",
        "i want a refund",
        "how do i reset my password",
    ]
    embs = embed_texts(sample_texts, show_progress=False)
    print("Embedding shape:", embs.shape)
    print("Sample embedding norm (should be ~1.0 due to normalization):", np.linalg.norm(embs[0]))