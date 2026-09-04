"""Project-wide configuration, loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- API keys / secrets ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model settings (filled in as later stages are implemented) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"  # verified against Groq's model list, Sept 2026

# --- Routing ---
INTENT_CONFIDENCE_THRESHOLD = 0.55
INTENT_NOISE_FLOOR = 0.3  # below this, treat as noise regardless of sentiment