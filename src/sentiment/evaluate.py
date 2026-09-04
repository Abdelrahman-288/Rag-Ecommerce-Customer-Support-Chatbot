"""Evaluate the fine-tuned sentiment model on the held-out test set."""

import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from configs.config import MODELS_DIR, REPORTS_DIR
from src.sentiment.preprocessing import SENTIMENT_LABELS, load_emotion_dataset
from src.utils.device import get_device

logger = logging.getLogger(__name__)

SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment" / "final"
SENTIMENT_REPORTS_DIR = REPORTS_DIR / "evaluation"
MAX_LENGTH = 64


def evaluate_sentiment_classifier():
    device = get_device()
    logger.info("Evaluating on device: %s", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained(SENTIMENT_MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
    model.to(device)
    model.eval()

    _, _, test_df = load_emotion_dataset()

    all_preds = []
    batch_size = 32
    texts = test_df["text"].tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**encodings).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    y_true = test_df["sentiment_id"].values
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Test macro precision: %.4f", precision)
    logger.info("Test macro recall: %.4f", recall)
    logger.info("Test macro F1: %.4f", f1)

    report = classification_report(y_true, y_pred, target_names=SENTIMENT_LABELS, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    import pandas as pd

    cm_df = pd.DataFrame(cm, index=SENTIMENT_LABELS, columns=SENTIMENT_LABELS)
    print("\nConfusion Matrix:\n")
    print(cm_df)

    SENTIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(SENTIMENT_REPORTS_DIR / "sentiment_confusion_matrix.csv")
    with open(SENTIMENT_REPORTS_DIR / "sentiment_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved confusion matrix and classification report to %s", SENTIMENT_REPORTS_DIR)

    errors = test_df.copy()
    errors["predicted_id"] = y_pred
    errors["predicted"] = errors["predicted_id"].map(lambda i: SENTIMENT_LABELS[i])
    errors = errors[errors["sentiment"] != errors["predicted"]]

    if len(errors) == 0:
        logger.info("No misclassifications on the test set.")
    else:
        logger.info("Total misclassified: %d / %d", len(errors), len(test_df))
        confusion_pairs = errors.groupby(["sentiment", "predicted"]).size().sort_values(ascending=False)
        print("\nTop confused sentiment pairs (true -> predicted):\n")
        print(confusion_pairs.head(10))
        print("\nSample misclassified examples:\n")
        print(errors[["text", "emotion", "sentiment", "predicted"]].head(10).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_sentiment_classifier()