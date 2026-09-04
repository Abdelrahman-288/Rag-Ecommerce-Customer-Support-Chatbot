"""Tests for sentiment classification predictor.

Uses the actual fine-tuned DistilBERT model. Slower than the other test
files (model load + inference), but still runs in seconds, not minutes --
acceptable for a unit test suite of this size.
"""

from src.sentiment.predictor import predict_sentiment


def test_predict_sentiment_returns_expected_keys():
    result = predict_sentiment("Thanks so much for your help!")
    assert "sentiment" in result
    assert "confidence" in result


def test_predict_sentiment_clearly_negative():
    result = predict_sentiment("This is unacceptable, I want a refund now.")
    assert result["sentiment"] == "Negative/Frustrated"


def test_predict_sentiment_clearly_positive():
    result = predict_sentiment("Thanks so much, that solved my problem!")
    assert result["sentiment"] == "Positive/Satisfied"


def test_predict_sentiment_plain_support_question_is_neutral():
    """Regression test for the domain-shift fix from Stage 4 --
    plain transactional questions should not be misread as Negative.
    """
    result = predict_sentiment("Where is my package?")
    assert result["sentiment"] == "Neutral"


def test_predict_sentiment_confidence_is_valid_probability():
    result = predict_sentiment("How do I reset my password?")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_sentiment_negation_handling():
    """Regression test for the negation bug found during Stage 8 live
    testing -- negated positive statements were misclassified as
    Positive/Satisfied before the negation-aware augmentation fix.
    """
    result = predict_sentiment("this is not acceptable")
    assert result["sentiment"] == "Negative/Frustrated"


def test_predict_sentiment_negated_negative_not_overcorrected():
    """Companion test to the negation fix above -- confirms the fix
    didn't overcorrect into 'negation always means negative'.
    """
    result = predict_sentiment("this wasn't a bad experience at all")
    assert result["sentiment"] == "Positive/Satisfied"