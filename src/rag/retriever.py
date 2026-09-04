"""Semantic retrieval over the FAISS vector store.

Given a query, embeds it with the same model used to build the index,
searches FAISS for the top_k nearest knowledge-base entries by cosine
similarity, and returns each result's instruction, response, similarity
score, and metadata.
"""

import logging

import faiss
import numpy as np

from configs.config import VECTOR_STORE_DIR
from src.rag.build_vector_store import DOC_IDS_PATH, FAISS_INDEX_PATH
from src.rag.embeddings import embed_texts
from src.rag.knowledge_base import load_knowledge_base

logger = logging.getLogger(__name__)

_index = None
_doc_ids = None
_kb_df = None


def _load_retrieval_assets():
    global _index, _doc_ids, _kb_df
    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run `python -m src.rag.build_vector_store` first."
            )
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        _doc_ids = np.load(DOC_IDS_PATH)
        _kb_df = load_knowledge_base().set_index("doc_id")
        logger.info("Loaded FAISS index (%d vectors) and knowledge base", _index.ntotal)
    return _index, _doc_ids, _kb_df


def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top_k most semantically similar knowledge-base entries.

    Returns a list of dicts, each with:
        doc_id, instruction, response, intent, category, similarity_score
    Ordered by descending similarity (best match first).
    """
    index, doc_ids, kb_df = _load_retrieval_assets()

    query_embedding = embed_texts([query], show_progress=False)
    scores, positions = index.search(query_embedding, top_k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue  # FAISS pads with -1 if fewer than top_k results exist
        doc_id = int(doc_ids[pos])
        row = kb_df.loc[doc_id]
        results.append({
            "doc_id": doc_id,
            "instruction": row["instruction"],
            "response": row["response"],
            "intent": row["intent"],
            "category": row["category"],
            "similarity_score": float(score),
        })
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Evaluation queries suggested in the course PDF (Stage 19).
    eval_queries = [
        "Where is my package?",
        "How can I get a refund?",
        "I want to change my order.",
        "I cannot access my account.",
    ]

    for query in eval_queries:
        separator = "=" * 70
        print("\n" + separator)
        print("QUERY:", query)
        print(separator)
        results = retrieve_documents(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r['similarity_score']:.4f}  intent={r['intent']}  category={r['category']}")
            print("    Q:", r["instruction"])
            answer = r["response"]
            preview = answer[:120] + "..." if len(answer) > 120 else answer
            print("    A:", preview)