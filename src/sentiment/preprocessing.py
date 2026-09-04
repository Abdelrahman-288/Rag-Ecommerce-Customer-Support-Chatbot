"""Data loading and preprocessing for sentiment classification.

Maps the dair-ai/emotion dataset's 6 fine-grained emotions down to 3
routing-relevant buckets: Negative/Frustrated, Neutral, Positive/Satisfied.

Note: this dataset is English-language Twitter text, not customer-support
text — a real domain shift from deployment-time messages. This is a known,
documented limitation (see project README's Limitations section) rather
than something fixable at the preprocessing stage.
"""

import logging

import pandas as pd
from datasets import load_dataset

from src.sentiment.augmentation_data import SENTIMENT_AUGMENTED_EXAMPLES

logger = logging.getLogger(__name__)

# dair-ai/emotion integer label -> emotion name (per dataset card)
EMOTION_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# Emotion -> 3-bucket sentiment mapping used for routing.
# Documented limitation: 'surprise' is mapped to Neutral as an approximation
# — surprise can be positive or negative depending on context, and the
# dataset does not disambiguate this.
EMOTION_TO_SENTIMENT = {
    "sadness": "Negative/Frustrated",
    "anger": "Negative/Frustrated",
    "fear": "Negative/Frustrated",
    "joy": "Positive/Satisfied",
    "love": "Positive/Satisfied",
    "surprise": "Neutral",
}

SENTIMENT_LABELS = ["Negative/Frustrated", "Neutral", "Positive/Satisfied"]
SENTIMENT_TO_ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {i: label for label, i in SENTIMENT_TO_ID.items()}


def load_emotion_dataset(augment: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dair-ai/emotion as pandas DataFrames with mapped sentiment labels.

    If augment=True (default), a small set of hand-written, naturally-phrased
    customer-support examples (see augmentation_data.py) is appended to the
    TRAINING split only -- never to val/test. This corrects the model's
    tendency to misread plain support questions as Negative/Frustrated,
    without touching the metrics we report (see Stage 7 finding).

    Returns
    -------
    (train_df, val_df, test_df) : each with columns
        ['text', 'emotion', 'sentiment', 'sentiment_id']
    """
    logger.info("Loading dair-ai/emotion from Hugging Face...")
    dataset = load_dataset("dair-ai/emotion")

    def to_mapped_df(split) -> pd.DataFrame:
        df = split.to_pandas()
        df["emotion"] = df["label"].map(EMOTION_NAMES)
        df["sentiment"] = df["emotion"].map(EMOTION_TO_SENTIMENT)
        df["sentiment_id"] = df["sentiment"].map(SENTIMENT_TO_ID)
        return df[["text", "emotion", "sentiment", "sentiment_id"]]

    train_df = to_mapped_df(dataset["train"])
    val_df = to_mapped_df(dataset["validation"])
    test_df = to_mapped_df(dataset["test"])

    if augment:
        aug_df = pd.DataFrame(SENTIMENT_AUGMENTED_EXAMPLES, columns=["text", "sentiment"])
        aug_df["emotion"] = "synthetic"  # not a real dair-ai emotion; flags these rows if inspected
        aug_df["sentiment_id"] = aug_df["sentiment"].map(SENTIMENT_TO_ID)
        aug_df = aug_df[["text", "emotion", "sentiment", "sentiment_id"]]
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        logger.info(
            "Added %d hand-written naturally-phrased examples to training split only",
            len(aug_df),
        )

    logger.info(
        "Loaded dataset — train: %d, val: %d, test: %d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def check_sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    return df["sentiment"].value_counts()