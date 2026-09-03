"""Data loading and text preprocessing for language detection.

The papluca/language-identification dataset ships with pre-split
train/validation/test sets, so we load them directly rather than
splitting ourselves — this avoids any risk of leakage between splits.
"""

import logging
import random
import re

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_language_id_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the papluca/language-identification dataset as pandas DataFrames.

    Returns
    -------
    (train_df, val_df, test_df) : each with columns ['text', 'labels']
    """
    logger.info("Loading papluca/language-identification from Hugging Face...")
    dataset = load_dataset("papluca/language-identification")

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    logger.info(
        "Loaded dataset — train: %d, val: %d, test: %d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def clean_text(text: str) -> str:
    """Light text cleaning for language detection.

    Deliberately minimal: no stemming/lemmatization/stopword removal,
    since those are language-specific operations and would distort the
    very signal (language-specific character/word patterns) the model
    relies on. We only strip things that add noise without carrying
    language information — URLs, extra whitespace — and lowercase for
    consistency.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply clean_text to a DataFrame's text column, returning a copy."""
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    # Drop any rows that became empty after cleaning
    before = len(df)
    df = df[df[text_col].str.len() > 0]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with empty text after cleaning", dropped)
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column — for inspection/logging."""
    return df.isnull().sum()


def check_class_distribution(df: pd.DataFrame, label_col: str = "labels") -> pd.Series:
    """Return value counts of the label column — for inspection/logging."""
    return df[label_col].value_counts()


def augment_with_short_snippets(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "labels",
    snippets_per_row: int = 1,
    min_words: int = 2,
    max_words: int = 6,
    seed: int = 42,
    prefer_start_prob: float = 0.6,
) -> pd.DataFrame:
    """Generate short-text training examples from longer sentences.

    The source dataset consists of full sentences/paragraphs, but real
    customer support messages are short (3-10 words). Character n-gram
    models need enough characters to build reliable signal, so a model
    trained only on long sentences underperforms on short queries at
    inference time. This samples short contiguous word spans from
    existing training sentences to teach the model what each language
    looks like in short-message form, correcting the train/deployment
    distribution mismatch.

    Sentence-initial spans are sampled more often than fully random
    spans (controlled by prefer_start_prob), since sentence-initial
    punctuation and capitalization (e.g. Spanish '¿') are strong,
    language-specific cues that a fully random mid-sentence crop would
    usually discard.
    """
    rng = random.Random(seed)
    augmented_rows = []

    for _, row in df.iterrows():
        words = row[text_col].split()
        if len(words) <= min_words:
            continue
        for _ in range(snippets_per_row):
            span_len = rng.randint(min_words, min(max_words, len(words)))
            if rng.random() < prefer_start_prob:
                start = 0
            else:
                start = rng.randint(0, len(words) - span_len)
            snippet = " ".join(words[start : start + span_len])
            augmented_rows.append({text_col: snippet, label_col: row[label_col]})

    return pd.DataFrame(augmented_rows)