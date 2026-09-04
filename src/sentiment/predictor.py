"""Inference-time sentiment prediction using the fine-tuned DistilBERT model."""

import logging

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.sentiment.preprocessing import SENTIMENT_LABELS
from src.sentiment.evaluate import SENTIMENT_MODEL_DIR
from src.utils.device import get_device

logger = logging.getLogger(__name__)

MAX_LENGTH = 64

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        device = get_device()
        _tokenizer = DistilBertTokenizerFast.from_pretrained(SENTIMENT_MODEL_DIR)
        _model = DistilBertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
        _model.to(device)
        _model.eval()
        logger.info("Loaded sentiment model onto %s", device)
    return _model, _tokenizer


def predict_sentiment(text: str) -> dict:
    """Predict sentiment for a single customer message.

    Returns {"sentiment": str, "confidence": float}.
    """
    model, tokenizer = _load_model()
    device = get_device()

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {"sentiment": SENTIMENT_LABELS[pred_idx], "confidence": confidence}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_messages = [
        "I'm so happy with my order, thank you!",
        "Where is my package?",
        "This is absolutely unacceptable, I've been waiting for weeks!",
    ]
    for msg in test_messages:
        result = predict_sentiment(msg)
        print(f"{msg!r:60} -> {result}")