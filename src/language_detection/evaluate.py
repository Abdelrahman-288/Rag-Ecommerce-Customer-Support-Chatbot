"""Evaluate the trained language detection model on the held-out test set.

Uses the test split (never seen during training or hyperparameter choices)
to get an unbiased estimate of real-world performance, plus error analysis
to understand what the model confuses.
"""

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

from configs.config import MODELS_DIR, REPORTS_DIR
from src.language_detection.preprocessing import (
    load_language_id_dataset,
    preprocess_dataframe,
)

logger = logging.getLogger(__name__)

LANGUAGE_MODEL_DIR = MODELS_DIR / "language"
LANGUAGE_REPORTS_DIR = REPORTS_DIR / "evaluation"


def evaluate_language_detector():
    clf = joblib.load(LANGUAGE_MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(LANGUAGE_MODEL_DIR / "vectorizer.joblib")

    _, _, test_df = load_language_id_dataset()
    test_df = preprocess_dataframe(test_df)

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["labels"]
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Test macro precision: %.4f", precision)
    logger.info("Test macro recall: %.4f", recall)
    logger.info("Test macro F1: %.4f", f1)

    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

    LANGUAGE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(LANGUAGE_REPORTS_DIR / "language_confusion_matrix.csv")

    with open(LANGUAGE_REPORTS_DIR / "language_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Saved confusion matrix and classification report to %s", LANGUAGE_REPORTS_DIR)

    errors = test_df.copy()
    errors["predicted"] = y_pred
    errors = errors[errors["labels"] != errors["predicted"]]

    if len(errors) == 0:
        logger.info("No misclassifications on the test set.")
    else:
        logger.info("Total misclassified: %d / %d", len(errors), len(test_df))
        confusion_pairs = errors.groupby(["labels", "predicted"]).size().sort_values(ascending=False)
        print("\nTop confused language pairs (true -> predicted):\n")
        print(confusion_pairs.head(10))
        print("\nSample misclassified examples:\n")
        print(errors[["text", "labels", "predicted"]].head(10).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_language_detector()