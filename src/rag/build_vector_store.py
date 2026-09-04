"""Builds and persists the FAISS vector index from the knowledge base."""

import logging

import faiss
import numpy as np

from configs.config import VECTOR_STORE_DIR
from src.rag.embeddings import embed_texts
from src.rag.knowledge_base import load_knowledge_base

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
DOC_IDS_PATH = VECTOR_STORE_DIR / "doc_ids.npy"


def build_vector_store():
    kb_df = load_knowledge_base()
    logger.info("Embedding %d knowledge-base instructions...", len(kb_df))

    embeddings = embed_texts(kb_df["instruction"].tolist())
    dim = embeddings.shape[1]

    # Inner product on L2-normalized vectors = cosine similarity search.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # doc_ids[i] gives the knowledge_base.csv doc_id for FAISS row i -- since
    # we build the index in the same row order as kb_df, this is just the
    # doc_id column, saved separately so retriever.py doesn't need to trust
    # row-order assumptions implicitly.
    doc_ids = kb_df["doc_id"].to_numpy()
    np.save(DOC_IDS_PATH, doc_ids)

    logger.info(
        "Saved FAISS index (%d vectors, dim=%d) and doc_ids to %s",
        index.ntotal, dim, VECTOR_STORE_DIR,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vector_store()