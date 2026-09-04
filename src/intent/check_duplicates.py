"""Sanity check: how templated/duplicated is the Bitext instruction text?

If near-identical instructions appear across train/val/test splits, our
'held-out' evaluation is inflated by template leakage rather than genuine
generalization. This script quantifies that risk.
"""

import logging

from src.intent.preprocessing import load_intent_dataset

logger = logging.getLogger(__name__)


def check_duplicates():
    train_df, val_df, test_df = load_intent_dataset()

    print(f"Train rows: {len(train_df)}, unique texts: {train_df['text'].nunique()}")
    print(f"Val rows: {len(val_df)}, unique texts: {val_df['text'].nunique()}")
    print(f"Test rows: {len(test_df)}, unique texts: {test_df['text'].nunique()}")

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    print(f"\nExact-duplicate texts shared between train and test: {len(train_texts & test_texts)}")
    print(f"Exact-duplicate texts shared between train and val: {len(train_texts & val_texts)}")

    # Placeholder-stripped version: replace {{...}} to see near-duplicate templates
    import re

    def strip_placeholders(text):
        return re.sub(r"\{\{.*?\}\}", "{{PLACEHOLDER}}", text)

    train_templates = set(train_df["text"].apply(strip_placeholders))
    test_templates = set(test_df["text"].apply(strip_placeholders))
    shared_templates = train_templates & test_templates

    print(f"\nUnique templates (placeholders normalized) — train: {len(train_templates)}, test: {len(test_templates)}")
    print(f"Templates shared between train and test (after normalizing placeholders): {len(shared_templates)}")
    print(f"As % of test templates: {len(shared_templates) / len(test_templates) * 100:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_duplicates()