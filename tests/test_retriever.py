"""Tests for the RAG retriever: real FAISS index, real embeddings.

No mocking needed here -- retrieval is local (no external API call), and
running it is fast enough (index is already built and loaded from disk).
"""

from src.rag.retriever import retrieve_documents


def test_retrieve_documents_returns_results():
    results = retrieve_documents("Where is my package?", top_k=3)
    assert len(results) == 3


def test_retrieve_documents_result_shape():
    results = retrieve_documents("How can I get a refund?", top_k=1)
    doc = results[0]
    assert "doc_id" in doc
    assert "instruction" in doc
    assert "response" in doc
    assert "intent" in doc
    assert "category" in doc
    assert "similarity_score" in doc


def test_retrieve_documents_relevant_to_refund_query():
    results = retrieve_documents("How can I get a refund?", top_k=3)
    # Every top-3 result for this near-exact-phrase query should be
    # refund-related -- a real regression check on retrieval quality.
    assert all(r["category"] == "REFUND" for r in results)


def test_retrieve_documents_respects_top_k():
    results = retrieve_documents("I cannot access my account.", top_k=5)
    assert len(results) == 5


def test_retrieve_documents_scores_are_descending():
    results = retrieve_documents("I want to change my order.", top_k=5)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)