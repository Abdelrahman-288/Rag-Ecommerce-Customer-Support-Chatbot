"""Tests for language detection predictor.

Uses the actual trained model (fast: TF-IDF + LogisticRegression loads
and predicts in milliseconds, no need to mock).
"""

import pytest

from src.language_detection.predictor import predict_language


def test_predict_language_returns_expected_keys():
    result = predict_language("Hello, how are you?")
    assert "language" in result
    assert "confidence" in result


def test_predict_language_english():
    result = predict_language("Where is my order? I need help with delivery.")
    assert result["language"] == "en"
    assert result["confidence"] > 0.5


def test_predict_language_empty_string():
    result = predict_language("")
    assert result["language"] == "unknown"
    assert result["confidence"] == 0.0


def test_predict_language_whitespace_only():
    result = predict_language("   ")
    assert result["language"] == "unknown"
    assert result["confidence"] == 0.0


def test_predict_language_confidence_is_valid_probability():
    result = predict_language("Bonjour, comment allez-vous?")
    assert 0.0 <= result["confidence"] <= 1.0