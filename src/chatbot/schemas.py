"""Typed structures for chatbot pipeline input/output.

Shared by both Streamlit and FastAPI (Stage 24/25) so neither duplicates
the NLP/RAG logic -- they just call src.chatbot.pipeline and render this
structure.
"""

from dataclasses import dataclass, field


@dataclass
class RetrievedDoc:
    doc_id: int
    instruction: str
    response: str
    intent: str
    category: str
    similarity_score: float


@dataclass
class ChatbotResponse:
    language: str
    language_confidence: float
    sentiment: str
    sentiment_confidence: float
    intent: str
    intent_confidence: float
    intent_source: str  # "rule" or "model"
    route: str  # "direct", "rag", "rag_escalate", "refusal"
    retrieved_documents: list[RetrievedDoc] = field(default_factory=list)
    response: str = ""
    escalate: bool = False

    def to_dict(self) -> dict:
        """Serialize for Streamlit/FastAPI consumption."""
        return {
            "language": self.language,
            "language_confidence": self.language_confidence,
            "sentiment": self.sentiment,
            "sentiment_confidence": self.sentiment_confidence,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "intent_source": self.intent_source,
            "route": self.route,
            "retrieved_documents": [vars(d) for d in self.retrieved_documents],
            "response": self.response,
            "escalate": self.escalate,
        }