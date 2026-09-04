"""Prompt construction for grounded RAG generation."""

SYSTEM_PROMPT = """You are a helpful, professional customer support assistant for an online retailer.

Answer the customer's question using ONLY the information in the retrieved support responses below. Follow these rules strictly:

1. Use only the retrieved context to answer -- do not invent policies, facts, order details, or refund decisions.
2. Some retrieved responses contain placeholders like {{Order Number}} or {{Person Name}}. These are generic template markers, NOT real values. Never repeat a placeholder verbatim to the customer and never invent a value for it. Instead, phrase your answer generically (e.g. "you can find this in your order confirmation email" rather than repeating "{{Order Number}}"), or ask the customer to provide the specific detail if it's genuinely needed.
3. If the customer sounds frustrated, acknowledge that briefly and empathetically before answering.
4. If the retrieved context does not cover the question, say so honestly and recommend escalating to a human agent -- do not guess.
5. Keep responses concise and professional.
"""


def build_user_prompt(user_message: str, retrieved_docs: list[dict], sentiment: str | None = None) -> str:
    """Build the user-turn prompt: retrieved context + the customer's question.

    retrieved_docs: list of dicts from retriever.retrieve_documents(), each
    with at least a 'response' key (used as grounding context).
    sentiment: optional detected sentiment label, injected so the LLM can
    calibrate tone (e.g. more apologetic for a frustrated customer).
    """
    if not retrieved_docs:
        context_block = "(No relevant support information was found in the knowledge base.)"
    else:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Support response {i}]\n{doc['response']}")
        context_block = "\n\n".join(context_parts)

    sentiment_line = f"\nDetected customer sentiment: {sentiment}" if sentiment else ""

    return f"""Context (retrieved past support responses):
{context_block}
{sentiment_line}
Customer question: "{user_message}"
"""