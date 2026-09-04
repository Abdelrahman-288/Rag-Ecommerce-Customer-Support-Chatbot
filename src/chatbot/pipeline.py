"""End-to-end chatbot pipeline: language -> sentiment -> intent -> router
-> RAG/direct/escalation -> LLM -> structured response.

Shared by Streamlit (Stage 8) and FastAPI (Stage 10) so neither duplicates
this logic -- both call process_message() and render/serialize the result.
"""

import logging

from src.chatbot.router import DIRECT_RESPONSES, REFUSAL_RESPONSE, determine_route
from src.chatbot.schemas import ChatbotResponse, RetrievedDoc
from src.intent.predictor import predict_intent
from src.language_detection.predictor import predict_language
from src.rag.generator import generate_response
from src.rag.retriever import retrieve_documents
from src.sentiment.predictor import predict_sentiment

logger = logging.getLogger(__name__)

RAG_TOP_K = 3


def process_message(user_message: str) -> ChatbotResponse:
    """Run a customer message through the full chatbot pipeline.

    Empty/whitespace input is handled explicitly (Stage 32 error handling)
    rather than being passed through to the NLP models.
    """
    if not user_message or not user_message.strip():
        return ChatbotResponse(
            language="unknown",
            language_confidence=0.0,
            sentiment="Neutral",
            sentiment_confidence=0.0,
            intent="out_of_scope",
            intent_confidence=0.0,
            intent_source="rule",
            route="refusal",
            response="It looks like your message was empty — could you tell me what you need help with?",
        )

    # --- Language detection ---
    lang_result = predict_language(user_message)
    logger.info("Language: %s (%.2f)", lang_result["language"], lang_result["confidence"])

    # --- Sentiment classification ---
    sentiment_result = predict_sentiment(user_message)
    logger.info("Sentiment: %s (%.2f)", sentiment_result["sentiment"], sentiment_result["confidence"])

    # --- Intent classification (rule-based small talk, else ML) ---
    intent_result = predict_intent(user_message)
    logger.info(
        "Intent: %s (%.2f, source=%s)",
        intent_result["intent"],
        intent_result["confidence"],
        intent_result["source"],
    )

    # --- Routing decision ---
    route = determine_route(
        intent_result["intent"],
        intent_result["confidence"],
        intent_result["source"],
        sentiment=sentiment_result["sentiment"],
    )
    logger.info("Route: %s", route)

    retrieved_docs: list[RetrievedDoc] = []
    escalate = False
    response_text = ""

    if route == "direct":
        response_text = DIRECT_RESPONSES.get(
            intent_result["intent"], "How can I help you today?"
        )

    elif route == "refusal":
        response_text = REFUSAL_RESPONSE

    elif route in ("rag", "rag_escalate"):
        raw_docs = retrieve_documents(user_message, top_k=RAG_TOP_K)
        retrieved_docs = [RetrievedDoc(**d) for d in raw_docs]
        response_text = generate_response(
            user_message,
            raw_docs,
            sentiment=sentiment_result["sentiment"],
        )
        if route == "rag_escalate":
            escalate = True
            response_text += (
                "\n\nI've also flagged this conversation for review by a human agent "
                "given the nature of your concern."
            )

    return ChatbotResponse(
        language=lang_result["language"],
        language_confidence=lang_result["confidence"],
        sentiment=sentiment_result["sentiment"],
        sentiment_confidence=sentiment_result["confidence"],
        intent=intent_result["intent"],
        intent_confidence=intent_result["confidence"],
        intent_source=intent_result["source"],
        route=route,
        retrieved_documents=retrieved_docs,
        response=response_text,
        escalate=escalate,
    )


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    test_messages = [
        "Hi there!",
        "Where is my order?",
        "This is the worst service ever, I want a refund now!",
        "Thanks for your help",
        "asdkjaslkdj random gibberish text",
    ]

    for msg in test_messages:
        print("\n" + "=" * 70)
        print("USER:", msg)
        print("=" * 70)
        result = process_message(msg)
        print(f"Language: {result.language} ({result.language_confidence:.2f})")
        print(f"Sentiment: {result.sentiment} ({result.sentiment_confidence:.2f})")
        print(f"Intent: {result.intent} ({result.intent_confidence:.2f}, {result.intent_source})")
        print(f"Route: {result.route} | Escalate: {result.escalate}")
        print(f"Response: {result.response}")