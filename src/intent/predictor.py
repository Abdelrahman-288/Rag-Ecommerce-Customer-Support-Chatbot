"""Inference-time intent prediction.

Architecture: a lightweight rule-based layer handles greeting/goodbye/
gratitude (closed, low-ambiguity vocabulary; Bitext has zero training
examples for these), and everything else falls through to the trained
TF-IDF + Logistic Regression classifier. See README for the rationale.
"""

import logging
import re

import joblib

from src.intent.preprocessing import clean_text
from src.intent.train import INTENT_MODEL_DIR

logger = logging.getLogger(__name__)

# Small, deliberately conservative keyword/regex sets. Precision matters more
# than recall here — a missed greeting just falls through to the ML model
# and likely lands in out_of_scope, which is a safe failure mode.
GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|greetings)\b", re.IGNORECASE
)
GOODBYE_PATTERNS = re.compile(
    r"\b(bye|goodbye|good bye|see you|take care|farewell)\b", re.IGNORECASE
)
GRATITUDE_PATTERNS = re.compile(
    r"\b(thanks|thank you|thx|appreciate it|much appreciated)\b", re.IGNORECASE
)

_model = None
_vectorizer = None


def _load_model():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(INTENT_MODEL_DIR / "intent_model.joblib")
        _vectorizer = joblib.load(INTENT_MODEL_DIR / "intent_vectorizer.joblib")
        logger.info("Loaded intent model and vectorizer")
    return _model, _vectorizer


def detect_small_talk(text: str) -> str | None:
    """Return 'greeting' / 'goodbye' / 'gratitude' if text matches a
    closed-vocabulary small-talk pattern, else None.

    Checked in this order because a message could contain both a greeting
    and thanks ("hi, thanks for the help") — greeting is checked first as
    it's most often the true intent of the message opener.
    """
    if GREETING_PATTERNS.search(text):
        return "greeting"
    if GOODBYE_PATTERNS.search(text):
        return "goodbye"
    if GRATITUDE_PATTERNS.search(text):
        return "gratitude"
    return None


def predict_intent(text: str) -> dict:
    """Predict the routing-level intent for a single customer message.

    Returns a dict: {"intent": str, "confidence": float, "source": str}
    source is "rule" for small-talk matches (confidence fixed at 1.0 — these
    are deterministic pattern matches, not probabilistic), or "model" for
    ML-classified intents (confidence = predicted class probability).
    """
    cleaned = clean_text(text)

    small_talk_intent = detect_small_talk(cleaned)
    if small_talk_intent is not None:
        return {"intent": small_talk_intent, "confidence": 1.0, "source": "rule"}

    model, vectorizer = _load_model()
    X = vectorizer.transform([cleaned])
    probs = model.predict_proba(X)[0]
    pred_idx = probs.argmax()
    pred_label = model.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    return {"intent": pred_label, "confidence": confidence, "source": "model"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_messages = [
        "Hi there!",
        "Thanks so much for your help",
        "Bye, talk later",
        "Where is my order?",
        "I want a refund for my broken item",
        "How do I reset my password",
    ]
    for msg in test_messages:
        result = predict_intent(msg)
        print(f"{msg!r:55} -> {result}")