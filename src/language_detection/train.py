"""Train a TF-IDF + Logistic Regression language detection classifier.

Fits TfidfVectorizer strictly on the training split only, to avoid any
leakage from validation/test data into the vocabulary or IDF weights.
Training data is augmented with short-text snippets sampled from the
original sentences, since real customer messages are short while the
source dataset consists of full sentences — without this, the model
underperforms on short queries at inference time.
"""

import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from configs.config import MODELS_DIR
from src.language_detection.preprocessing import (
    augment_with_short_snippets,
    load_language_id_dataset,
    preprocess_dataframe,
)

logger = logging.getLogger(__name__)

LANGUAGE_MODEL_DIR = MODELS_DIR / "language"


def train_language_detector() -> None:
    """Load data, preprocess, augment, fit TF-IDF + Logistic Regression, save artifacts."""
    train_df, val_df, _ = load_language_id_dataset()

    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)

    logger.info("Generating short-text augmented training samples...")
    short_snippets = augment_with_short_snippets(
        train_df, snippets_per_row=1, min_words=2, max_words=6
    )
    short_snippets = preprocess_dataframe(short_snippets)
    logger.info("Added %d short-text augmented samples", len(short_snippets))

    train_df = pd.concat([train_df, short_snippets], ignore_index=True)
    logger.info("Total training samples after augmentation: %d", len(train_df))

    logger.info("Fitting TF-IDF vectorizer on training data only...")
    # char n-grams work better than word n-grams for language ID: they
    # capture language-specific letter patterns (e.g. accented characters,
    # digraphs) even for short texts or unseen words.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        max_features=50_000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])

    y_train = train_df["labels"]
    y_val = val_df["labels"]

    logger.info("Training Logistic Regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        C=10.0,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    logger.info("Train accuracy: %.4f", train_acc)
    logger.info("Validation accuracy: %.4f", val_acc)

    LANGUAGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, LANGUAGE_MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, LANGUAGE_MODEL_DIR / "vectorizer.joblib")
    logger.info("Saved model and vectorizer to %s", LANGUAGE_MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_language_detector()