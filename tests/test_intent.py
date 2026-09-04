"""Tests for intent classification: both the rule-based small-talk layer
and the ML classifier fallback.
"""

from src.intent.predictor import detect_small_talk, predict_intent


def test_detect_small_talk_greeting():
    assert detect_small_talk("Hi there!") == "greeting"
    assert detect_small_talk("Hello") == "greeting"
    assert detect_small_talk("Good morning") == "greeting"


def test_detect_small_talk_goodbye():
    assert detect_small_talk("Bye, talk later") == "goodbye"
    assert detect_small_talk("take care") == "goodbye"


def test_detect_small_talk_gratitude():
    assert detect_small_talk("Thanks so much for your help") == "gratitude"
    assert detect_small_talk("thank you!") == "gratitude"


def test_detect_small_talk_no_match_returns_none():
    assert detect_small_talk("Where is my order?") is None


def test_predict_intent_rule_based_greeting():
    result = predict_intent("Hi there!")
    assert result["intent"] == "greeting"
    assert result["source"] == "rule"
    assert result["confidence"] == 1.0


def test_predict_intent_model_based_order_status():
    result = predict_intent("Where is my order?")
    assert result["intent"] == "order_status"
    assert result["source"] == "model"


def test_predict_intent_model_based_billing():
    result = predict_intent("I want a refund for my broken item")
    assert result["intent"] == "billing_and_refunds"
    assert result["source"] == "model"


def test_predict_intent_confidence_is_valid_probability():
    result = predict_intent("How do I reset my password?")
    assert 0.0 <= result["confidence"] <= 1.0