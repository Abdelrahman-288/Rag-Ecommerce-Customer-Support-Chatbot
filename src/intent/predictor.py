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
    r"^\s*(hi(\s+there)?|hello(\s+there)?|hey(\s+there)?|good\s+(morning|afternoon|evening)|greetings)\b",
    re.IGNORECASE,
)
GOODBYE_PATTERNS = re.compile(
    r"\b(bye(\s+bye)?|goodbye|good\s+bye|see\s+you(\s+later)?|take\s+care|farewell)\b",
    re.IGNORECASE,
)
GRATITUDE_PATTERNS = re.compile(
    r"\b(thanks(\s+(a\s+lot|so\s+much|very\s+much))?|thank\s+you(\s+(so\s+much|very\s+much))?|thx|appreciate\s+it|much\s+appreciated)\b",
    re.IGNORECASE,
)

# Common conversational filler words allowed in pure small talk
SMALL_TALK_FILLERS = {
    "there", "all", "everyone", "bot", "assistant", "team", "support",
    "friend", "you", "so", "much", "very", "a", "lot", "for", "your",
    "the", "help", "talk", "later", "again", "soon", "now", "have",
    "good", "great", "day", "one", "guys", "folks", "to", "and",
}

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
    """Return 'greeting' / 'goodbye' / 'gratitude' if text is standalone
    small talk, or None if it contains substantive customer support queries.

    Checked in priority order. If small talk tokens match, remaining words
    are checked to ensure the user isn't asking a real question (e.g.
    'Hello, where is my order?' falls through to the ML model).
    """
    cleaned = text.lower().strip()

    m = GREETING_PATTERNS.search(cleaned)
    if m:
        remaining = GREETING_PATTERNS.sub("", cleaned).strip()
        remaining_words = [w for w in re.findall(r"\w+", remaining) if w not in SMALL_TALK_FILLERS]
        if not remaining_words:
            return "greeting"

    m = GOODBYE_PATTERNS.search(cleaned)
    if m:
        remaining = GOODBYE_PATTERNS.sub("", cleaned).strip()
        remaining_words = [w for w in re.findall(r"\w+", remaining) if w not in SMALL_TALK_FILLERS]
        if not remaining_words:
            return "goodbye"

    m = GRATITUDE_PATTERNS.search(cleaned)
    if m:
        remaining = GRATITUDE_PATTERNS.sub("", cleaned).strip()
        remaining_words = [w for w in re.findall(r"\w+", remaining) if w not in SMALL_TALK_FILLERS]
        if not remaining_words:
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