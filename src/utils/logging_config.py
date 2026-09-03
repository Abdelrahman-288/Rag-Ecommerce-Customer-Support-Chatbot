"""Centralized logging configuration for the project."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, consistently, across scripts/app/api.

    Call this at the entrypoint of any script, the Streamlit app, or the
    FastAPI app — not inside individual modules — to avoid duplicate
    handlers.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured (e.g. Streamlit re-running the script) — skip.
        return

    root_logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)