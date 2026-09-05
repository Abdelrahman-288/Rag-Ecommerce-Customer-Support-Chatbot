"""FastAPI interface for the RAG-based e-commerce support chatbot.

Exposes POST /chat, reusing the exact same pipeline as the Streamlit app
(src/chatbot/pipeline.py) -- no NLP/RAG logic is duplicated here.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chatbot.pipeline import process_message
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at server startup rather than on the first
    real request, avoiding a slow first response.
    """
    logger.info("Warming up pipeline (loading models)...")
    process_message("hello")
    logger.info("Pipeline ready.")
    yield


app = FastAPI(
    title="E-commerce Support Chatbot API",
    description=(
        "RAG-based customer support chatbot with language detection, "
        "sentiment analysis, intent classification, and grounded retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=0, description="The customer's message")


class ChatResponse(BaseModel):
    language: str
    language_confidence: float
    sentiment: str
    sentiment_confidence: float
    intent: str
    intent_confidence: float
    intent_source: str
    route: str
    retrieved_documents: list[dict]
    response: str
    escalate: bool


@app.get("/")
def health_check() -> dict:
    """Basic health check -- confirms the API is up and reachable."""
    return {"status": "ok", "service": "ecommerce-support-chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    """Process a customer message through the full chatbot pipeline.

    Empty messages are handled gracefully by the pipeline itself (returns
    a clarifying response rather than an error), so no special-casing is
    needed here beyond catching genuinely unexpected failures.
    """
    try:
        result = process_message(request.message)
        return result.to_dict()
    except Exception as e:
        logger.error("Pipeline error while processing request: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your message.",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)