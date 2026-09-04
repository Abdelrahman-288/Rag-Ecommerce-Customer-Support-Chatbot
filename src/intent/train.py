"""Train TF-IDF + Logistic Regression intent classifier."""

import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from configs.config import MODELS_DIR
from src.intent.preprocessing import ROUTING_INTENTS, load_intent_dataset

logger = logging.getLogger(__name__)

INTENT_MODEL_DIR = MODELS_DIR / "intent"


def train_intent_classifier():
    train_df, val_df, test_df = load_intent_dataset()

    # Fit TF-IDF ONLY on training data — no leakage.
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])

    y_train = train_df["routing_intent"]
    y_val = val_df["routing_intent"]

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # billing_and_refunds is ~3.5x complaint's size
        random_state=42,
    )
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    logger.info("Validation macro F1: %.4f", val_f1)
    print("\nValidation Classification Report:\n")
    print(classification_report(y_val, val_preds, zero_division=0))

    INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, INTENT_MODEL_DIR / "intent_model.joblib")
    joblib.dump(vectorizer, INTENT_MODEL_DIR / "intent_vectorizer.joblib")
    # Persist the test split so evaluate.py uses the exact same held-out rows.
    test_df.to_csv(INTENT_MODEL_DIR / "test_split.csv", index=False)
    logger.info("Saved model, vectorizer, and test split to %s", INTENT_MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_intent_classifier()