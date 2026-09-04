"""Data loading, intent mapping, and preprocessing for intent classification.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
Columns of interest: 'instruction' (customer message), 'intent' (gold label,
27 fine-grained classes), 'category' (10 coarse classes, not used directly).

We collapse the 27 fine-grained intents into 9 routing-level classes. Note:
greeting / goodbye / gratitude are kept as SEPARATE classes at the model
layer (deliberate choice — see README) even though the router later sends
all three down the same "direct response" path. Collapsing at the model
layer would destroy information the classifier can learn for free.

IMPORTANT: this dataset's 27 raw intents do NOT include greet/bye/thank —
small talk is not represented in this dataset at all. See README /
Stage 5 notes for how greeting/goodbye/gratitude are handled instead
(rule-based detection, not the trained classifier).

IMPORTANT: Bitext contains ~1.5k exact-duplicate instruction texts (a known
property of its templated generation process). Duplicates are dropped
BEFORE splitting in load_intent_dataset() to prevent train/test leakage —
see check_duplicates.py for the measured impact.
"""

import logging
import re

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.intent.augmentation_data import AUGMENTED_EXAMPLES

logger = logging.getLogger(__name__)

# --- Fine-grained (dataset) intent -> routing-level intent ---
# Left side must exactly match the dataset's raw intent strings.
# Verified against actual dataset output on 2026-09 (26,872 rows, 27 raw intents).
INTENT_MAPPING = {
    # order_status
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",

    # order_management
    "cancel_order": "order_management",
    "change_order": "order_management",
    "place_order": "order_management",
    "check_cancellation_fee": "order_management",

    # billing_and_refunds
    "check_invoice": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "check_refund_policy": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    "check_payment_methods": "billing_and_refunds",

    # account_management
    "create_account": "account_management",
    "edit_account": "account_management",
    "delete_account": "account_management",
    "switch_account": "account_management",
    "recover_password": "account_management",
    "registration_problems": "account_management",

    # complaint (priority-flagged, still uses RAG)
    "complaint": "complaint",
    "review": "complaint",

    # out_of_scope
    "contact_customer_service": "out_of_scope",
    "contact_human_agent": "out_of_scope",
    "change_shipping_address": "out_of_scope",
    "set_up_shipping_address": "out_of_scope",
    "newsletter_subscription": "out_of_scope",
}

ROUTING_INTENTS = [
    "greeting",
    "goodbye",
    "gratitude",
    "order_status",
    "order_management",
    "billing_and_refunds",
    "account_management",
    "complaint",
    "out_of_scope",
]


def load_raw_dataset() -> pd.DataFrame:
    """Load the Bitext dataset from Hugging Face and return as a DataFrame."""
    logger.info("Loading bitext/Bitext-customer-support-llm-chatbot-training-dataset...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    df = ds.to_pandas()
    logger.info("Loaded %d rows", len(df))
    return df


def inspect_raw_dataset(df: pd.DataFrame) -> None:
    """Print structure, missing values, and class distribution — inspect only."""
    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Missing values ===")
    print(df.isnull().sum())

    print("\n=== Raw intent value counts (27 fine-grained classes) ===")
    print(df["intent"].value_counts())

    print("\n=== Category value counts (10 coarse classes) ===")
    if "category" in df.columns:
        print(df["category"].value_counts())

    print("\n=== Sample rows ===")
    print(df[["instruction", "intent"]].sample(5, random_state=42).to_string(index=False))


def apply_intent_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw 27-class intents to the 9-class routing scheme.

    Raises a ValueError listing any raw intent values not covered by
    INTENT_MAPPING, rather than silently dropping or mislabeling rows.
    Note: greeting/goodbye/gratitude are NOT produced by this function,
    since the source dataset has no corresponding raw intents — those
    three routing classes are handled separately (see README).
    """
    unmapped = set(df["intent"].unique()) - set(INTENT_MAPPING.keys())
    if unmapped:
        raise ValueError(
            f"INTENT_MAPPING is missing raw intent values: {sorted(unmapped)}. "
            "Update INTENT_MAPPING before proceeding."
        )

    df = df.copy()
    df["routing_intent"] = df["intent"].map(INTENT_MAPPING)
    return df


def inspect_mapped_dataset(df: pd.DataFrame) -> None:
    """Print the resulting class distribution after mapping — sanity check."""
    print("\n=== Routing-level intent distribution (from dataset; excludes greeting/goodbye/gratitude) ===")
    counts = df["routing_intent"].value_counts()
    print(counts)
    print("\n=== As percentages ===")
    print((counts / len(df) * 100).round(2))


def clean_text(text: str) -> str:
    """Lowercase and normalize whitespace. Placeholders like {{Order Number}}
    are intentionally preserved — they're informative tokens, not noise.
    """
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def load_intent_dataset(test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42, augment: bool = True):
    """Load, map, clean, and split the Bitext dataset into train/val/test.

    Stratified on routing_intent to preserve class proportions across splits.
    Exact-duplicate instruction texts are dropped BEFORE splitting — Bitext
    contains ~1.5k literally-repeated instructions, and letting duplicates
    land on both sides of a split silently leaks test rows into training.

    If augment=True (default), a small set of hand-written, naturally-phrased
    examples (see augmentation_data.py) is appended to the TRAINING split
    only, after splitting — never to val/test. This recalibrates the
    classifier's confidence on organic user phrasing without touching the
    metrics we report (see Stage 5 confidence-gap finding).

    Returns (train_df, val_df, test_df), each with 'text' and 'routing_intent' columns.
    """
    df = load_raw_dataset()
    df = apply_intent_mapping(df)
    df["text"] = df["instruction"].apply(clean_text)

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    logger.info("Dropped %d exact-duplicate instruction texts (%d -> %d rows)",
                before - len(df), before, len(df))

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["routing_intent"],
        random_state=random_state,
    )
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["routing_intent"],
        random_state=random_state,
    )

    if augment:
        aug_df = pd.DataFrame(AUGMENTED_EXAMPLES, columns=["text", "routing_intent"])
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        logger.info(
            "Added %d hand-written naturally-phrased examples to training split only", len(aug_df)
        )

    logger.info(
        "Split — train: %d, val: %d, test: %d rows", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw_df = load_raw_dataset()
    inspect_raw_dataset(raw_df)
    mapped_df = apply_intent_mapping(raw_df)
    inspect_mapped_dataset(mapped_df)