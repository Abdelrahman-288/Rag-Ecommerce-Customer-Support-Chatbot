# Data

This folder holds raw and processed datasets used by the project. Contents are
git-ignored (see root `.gitignore`) — datasets are downloaded/generated via
`scripts/download_data.py`, not committed to the repository.

## Datasets used
- `papluca/language-identification` — language detection
- `dair-ai/emotion` — sentiment/emotion classification
- `bitext/Bitext-customer-support-llm-chatbot-training-dataset` — intent classification + RAG knowledge base

## Structure
- `raw/` — untouched, as-downloaded data
- `processed/` — cleaned/split data produced by preprocessing scripts