"""Evaluate the trained intent classifier on the held-out test set."""

import logging

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from configs.config import REPORTS_DIR
from src.intent.preprocessing import ROUTING_INTENTS
from src.intent.train import INTENT_MODEL_DIR

logger = logging.getLogger(__name__)

INTENT_REPORTS_DIR = REPORTS_DIR / "evaluation"


def evaluate_intent_classifier():
    clf = joblib.load(INTENT_MODEL_DIR / "intent_model.joblib")
    vectorizer = joblib.load(INTENT_MODEL_DIR / "intent_vectorizer.joblib")
    test_df = pd.read_csv(INTENT_MODEL_DIR / "test_split.csv")

    X_test = vectorizer.transform(test_df["text"])
    y_true = test_df["routing_intent"]
    y_pred = clf.predict(X_test)

    labels_present = sorted(y_true.unique())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Test macro precision: %.4f", precision)
    logger.info("Test macro recall: %.4f", recall)
    logger.info("Test macro F1: %.4f", f1)

    report = classification_report(y_true, y_pred, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    cm_df = pd.DataFrame(cm, index=labels_present, columns=labels_present)
    print("\nConfusion Matrix:\n")
    print(cm_df)

    INTENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(INTENT_REPORTS_DIR / "intent_confusion_matrix.csv")
    with open(INTENT_REPORTS_DIR / "intent_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved confusion matrix and classification report to %s", INTENT_REPORTS_DIR)

    errors = test_df.copy()
    errors["predicted"] = y_pred
    errors = errors[errors["routing_intent"] != errors["predicted"]]

    if len(errors) == 0:
        logger.info("No misclassifications on the test set.")
    else:
        logger.info("Total misclassified: %d / %d", len(errors), len(test_df))
        confusion_pairs = (
            errors.groupby(["routing_intent", "predicted"]).size().sort_values(ascending=False)
        )
        print("\nConfused intent pairs (true -> predicted):\n")
        print(confusion_pairs.head(15))
        print("\nAll misclassified examples:\n")
        print(errors[["instruction", "intent", "routing_intent", "predicted"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_intent_classifier()