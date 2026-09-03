"""Centralized device-selection utility.

Ensures deep-learning components (sentiment model, Transformers, Sentence
Transformer embeddings) use CUDA when available, while traditional CPU
algorithms (TF-IDF, Logistic Regression, scikit-learn preprocessing) are left
untouched — those should never call this function.
"""

import logging

logger = logging.getLogger(__name__)

_DEVICE = None


def get_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'.

    The result is computed once and cached, and the selected device is
    logged so it's always clear at runtime whether GPU acceleration is
    active.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    try:
        import torch

        if torch.cuda.is_available():
            _DEVICE = "cuda"
            logger.info(
                "CUDA available — using GPU: %s", torch.cuda.get_device_name(0)
            )
        else:
            _DEVICE = "cpu"
            logger.info("CUDA not available — using CPU")
    except ImportError:
        _DEVICE = "cpu"
        logger.warning("PyTorch not installed — defaulting to CPU")

    return _DEVICE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Selected device:", get_device())