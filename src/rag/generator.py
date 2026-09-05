"""LLM response generation via the Groq API, grounded in retrieved context."""

import logging

from groq import Groq

from configs.config import GROQ_API_KEY, GROQ_MODEL_NAME
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_response(
    user_message: str,
    retrieved_docs: list[dict],
    sentiment: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Generate a grounded response using retrieved context.

    Low temperature (0.3) by design -- this is a support-answer generation
    task where faithfulness to retrieved context matters more than creative
    variation.
    """
    client = _get_client()
    user_prompt = build_user_prompt(user_message, retrieved_docs, sentiment)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return (
            "I'm having trouble generating a response right now. "
            "Please try again shortly, or contact human support for immediate help."
        )


if __name__ == "__main__":
    import sys
    from src.rag.retriever import retrieve_documents
    from src.utils.logging_config import setup_logging

    setup_logging()
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_query = "How can I get a refund?"
    docs = retrieve_documents(test_query, top_k=3)
    response = generate_response(test_query, docs, sentiment="Neutral")

    print(f"\nQuery: {test_query}")
    print(f"\nGenerated response:\n{response}")