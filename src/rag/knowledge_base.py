"""Builds the RAG knowledge base from the Bitext customer-support dataset.

Each knowledge-base entry pairs:
- 'instruction' — the customer question (this is what gets embedded/searched)
- 'response' — the agent's answer (this is what gets injected into the LLM
  prompt as grounding context, NOT the instruction)
- metadata: intent, category (useful for debugging/UI display)

Reuses the same load + dedup logic as the intent classifier (Stage 5) so the
knowledge base and intent training data stay consistent — no duplicate
instructions bloating the vector store with redundant retrievable chunks.
"""

import logging

import pandas as pd

from configs.config import PROCESSED_DATA_DIR
from src.intent.preprocessing import clean_text, load_raw_dataset

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = PROCESSED_DATA_DIR / "knowledge_base.csv"


def build_knowledge_base() -> pd.DataFrame:
    """Load Bitext, dedupe on instruction text, and produce the KB dataframe.

    Returns a DataFrame with columns: doc_id, instruction, response, intent,
    category. doc_id is a stable integer index used later to map FAISS
    vector positions back to their source document.
    """
    df = load_raw_dataset()

    df["instruction_clean"] = df["instruction"].apply(clean_text)
    before = len(df)
    df = df.drop_duplicates(subset="instruction_clean", keep="first").reset_index(drop=True)
    logger.info(
        "Dropped %d duplicate instructions for KB (%d -> %d rows)",
        before - len(df), before, len(df),
    )

    kb_df = df[["instruction", "response", "intent", "category"]].copy()
    kb_df.insert(0, "doc_id", range(len(kb_df)))

    return kb_df


def save_knowledge_base(kb_df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb_df.to_csv(KNOWLEDGE_BASE_PATH, index=False)
    logger.info("Saved knowledge base (%d docs) to %s", len(kb_df), KNOWLEDGE_BASE_PATH)


def load_knowledge_base() -> pd.DataFrame:
    """Load the persisted knowledge base CSV (built once, reused everywhere)."""
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}. "
            "Run `python -m src.rag.knowledge_base` to build it first."
        )
    return pd.read_csv(KNOWLEDGE_BASE_PATH)


def inspect_knowledge_base(kb_df: pd.DataFrame) -> None:
    print("\n=== Knowledge base shape ===")
    print(kb_df.shape)

    print("\n=== Columns ===")
    print(kb_df.columns.tolist())

    print("\n=== Intent distribution (top 10) ===")
    print(kb_df["intent"].value_counts().head(10))

    print("\n=== Sample entries ===")
    for _, row in kb_df.sample(3, random_state=42).iterrows():
        print(f"\n[doc_id={row['doc_id']}] intent={row['intent']} category={row['category']}")
        print(f"  Q: {row['instruction']}")
        print(f"  A: {row['response'][:150]}{'...' if len(row['response']) > 150 else ''}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb = build_knowledge_base()
    inspect_knowledge_base(kb)
    save_knowledge_base(kb)