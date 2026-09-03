"""Inference-only predictor for the language detection model.

Loads the saved model and vectorizer once and reuses them — does not
retrain. Used by the chatbot pipeline (Stage 7) to detect a customer
message's language at runtime.
"""

import logging

import joblib

from configs.config import MODELS_DIR
from src.language_detection.preprocessing import clean_text

logger = logging.getLogger(__name__)

LANGUAGE_MODEL_DIR = MODELS_DIR / "language"

_model = None
_vectorizer = None


def _load_artifacts():
    """Load model and vectorizer once, caching them at module level."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        try:
            _model = joblib.load(LANGUAGE_MODEL_DIR / "model.joblib")
            _vectorizer = joblib.load(LANGUAGE_MODEL_DIR / "vectorizer.joblib")
            logger.info("Loaded language detection model and vectorizer")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Language detection model artifacts not found. "
                "Run 'python -m src.language_detection.train' first."
            ) from e
    return _model, _vectorizer


def predict_language(text: str) -> dict:
    """Predict the language of a single piece of text.

    Returns
    -------
    dict with keys:
        'language': predicted language code (e.g. 'en', 'fr')
        'confidence': probability of the predicted class
    """
    if not text or not text.strip():
        return {"language": "unknown", "confidence": 0.0}

    model, vectorizer = _load_artifacts()

    cleaned = clean_text(text)
    if not cleaned:
        return {"language": "unknown", "confidence": 0.0}

    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)

    return {"language": prediction, "confidence": float(confidence)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        "Where is my order?",
        "¿Dónde está mi pedido?",
        "Où est ma commande?",
        "أين طلبي؟",
    ]
    for text in samples:
        result = predict_language(text)
        print(f"{text!r} -> {result}")