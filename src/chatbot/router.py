"""Routing logic: decides how to handle a message based on detected intent
and confidence.

Routing table (per project spec, Stage 22):
    greeting / goodbye / gratitude  -> direct response (no retrieval needed)
    order_status / order_management /
    billing_and_refunds / account_management -> RAG
    complaint                        -> RAG + escalation flag
    out_of_scope                     -> polite refusal
    low-confidence ML prediction     -> treated as out_of_scope (safe fallback)
"""

import logging

from configs.config import INTENT_CONFIDENCE_THRESHOLD, INTENT_NOISE_FLOOR

logger = logging.getLogger(__name__)

DIRECT_INTENTS = {"greeting", "goodbye", "gratitude"}
RAG_INTENTS = {"order_status", "order_management", "billing_and_refunds", "account_management"}
COMPLAINT_INTENT = "complaint"
OUT_OF_SCOPE_INTENT = "out_of_scope"

# Canned responses for small-talk intents -- no LLM call needed, keeps
# these interactions instant and free of API cost/latency.
DIRECT_RESPONSES = {
    "greeting": "Hello! How can I help you with your order or account today?",
    "goodbye": "Thanks for reaching out — have a great day!",
    "gratitude": "You're very welcome! Let me know if there's anything else I can help with.",
}

REFUSAL_RESPONSE = (
    "I'm sorry, I'm not able to help with that particular request. "
    "Could you rephrase your question, or would you like me to connect you with a human agent?"
)


def determine_route(intent: str, confidence: float, source: str, sentiment: str | None = None) -> str:
    """Decide the routing path for a given intent prediction.

    Rule-based small-talk predictions (source='rule') are always trusted at
    face value (deterministic pattern match, confidence fixed at 1.0).
    ML-based predictions below the confidence threshold are downgraded to
    out_of_scope handling as a safe fallback -- per Stage 15 of the spec,
    an uncertain prediction should not be blindly trusted.

    Two exceptions to that downgrade:
    1. 'complaint' is never downgraded -- per the project spec, complaints
       must be flagged for priority handling regardless of classifier
       confidence.
    2. A low-but-not-negligible-confidence prediction (between
       INTENT_NOISE_FLOOR and INTENT_CONFIDENCE_THRESHOLD) paired with
       strongly Negative/Frustrated sentiment is treated as an unclear
       complaint rather than out_of_scope -- sentiment is a second,
       independent signal that the message is a real, upset customer even
       when the specific intent is ambiguous.

    Below INTENT_NOISE_FLOOR, the prediction is too unreliable to trust for
    any purpose, including escalation -- e.g. gibberish input can trigger
    a spuriously confident sentiment label alongside a near-random intent
    guess, and escalating that to a human agent queue would be a false
    positive we want to avoid.
    """
    low_confidence = source == "model" and confidence < INTENT_CONFIDENCE_THRESHOLD
    is_noise = source == "model" and confidence < INTENT_NOISE_FLOOR

    if low_confidence and intent != COMPLAINT_INTENT:
        if not is_noise and sentiment == "Negative/Frustrated":
            logger.info(
                "Low-confidence intent '%s' (%.2f) but sentiment is Negative/Frustrated "
                "— routing as complaint-style escalation instead of refusal",
                intent,
                confidence,
            )
            intent = COMPLAINT_INTENT
        else:
            logger.info(
                "Low-confidence intent '%s' (%.2f) below threshold %.2f — routing as out_of_scope",
                intent,
                confidence,
                INTENT_CONFIDENCE_THRESHOLD,
            )
            intent = OUT_OF_SCOPE_INTENT

    if intent in DIRECT_INTENTS:
        return "direct"
    if intent == COMPLAINT_INTENT:
        return "rag_escalate"
    if intent in RAG_INTENTS:
        return "rag"
    return "refusal"