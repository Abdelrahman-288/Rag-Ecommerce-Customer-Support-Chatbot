"""Multilingual test coverage.

Two things are tested here, deliberately kept separate:

1. Language detection itself across a sample of the 20 supported
   languages -- this is the one module actually trained to be
   multilingual.
2. The full pipeline's behavior when given non-English input -- this
   documents a real, known limitation: sentiment, intent, and the RAG
   knowledge base are all English-only (dair-ai/emotion and the Bitext
   dataset are both English-language datasets). Language detection
   correctly identifies the language, but downstream stages still process
   the raw (non-English) text through English-trained models, which is
   not linguistically meaningful for non-English input. These tests
   assert the *actual* current behavior, not the ideal behavior --
   they exist to make the limitation explicit and catch any accidental
   regression or crash on non-English input, not to claim correctness
   of non-English sentiment/intent results.
"""

import pytest

from src.chatbot.pipeline import process_message
from src.language_detection.predictor import predict_language

# (text, expected_language_code) -- phrases chosen to be reasonably long
# (not single words) since short text is a known weaker case for the
# language detector, documented in README Limitations.
LANGUAGE_SAMPLES = [
    ("Where is my order and when will it arrive?", "en"),
    ("¿Dónde está mi pedido y cuándo va a llegar?", "es"),
    ("Où est ma commande et quand va-t-elle arriver?", "fr"),
    ("Wo ist meine Bestellung und wann kommt sie an?", "de"),
    ("Dove si trova il mio ordine e quando arriverà?", "it"),
    ("Onde está o meu pedido e quando vai chegar?", "pt"),
    ("Waar is mijn bestelling en wanneer komt die aan?", "nl"),
    ("Где мой заказ и когда он придет?", "ru"),
    ("我的订单在哪里,什么时候到?", "zh"),
    ("私の注文はどこにありますか、いつ届きますか?", "ja"),
    ("طلبي أين هو ومتى سيصل؟", "ar"),
    ("Đơn hàng của tôi ở đâu và khi nào sẽ đến?", "vi"),
]


@pytest.mark.parametrize("text,expected_language", LANGUAGE_SAMPLES)
def test_language_detection_across_languages(text, expected_language):
    result = predict_language(text)
    assert result["language"] == expected_language, (
        f"Expected '{expected_language}' for {text!r}, got '{result['language']}' "
        f"(confidence {result['confidence']:.2f})"
    )


@pytest.mark.parametrize("text,expected_language", LANGUAGE_SAMPLES)
def test_language_detection_confidence_reasonable(text, expected_language):
    """Longer, full-sentence queries should get reasonably confident
    predictions -- this is the case the short-text augmentation (Stage 3)
    was specifically designed to also handle well, not just accept as
    unavoidably low-confidence.
    """
    result = predict_language(text)
    assert result["confidence"] > 0.5, (
        f"Low confidence ({result['confidence']:.2f}) for {text!r} "
        f"predicted as '{result['language']}'"
    )


class TestPipelineNonEnglishBehavior:
    """Documents actual pipeline behavior on non-English input.

    These tests assert what the system currently does, not what an
    ideal multilingual system would do -- see module docstring.
    """

    def test_pipeline_detects_language_correctly_for_non_english_input(self):
        result = process_message("¿Dónde está mi pedido?")
        assert result.language == "es"

    def test_pipeline_does_not_crash_on_non_english_input(self):
        """Regression test: non-English input must not raise an exception
        anywhere in the pipeline, even though sentiment/intent/RAG are not
        linguistically meaningful for it.
        """
        for text, _ in LANGUAGE_SAMPLES:
            result = process_message(text)
            assert result.response != ""
            assert result.route in ("direct", "rag", "rag_escalate", "refusal")

    def test_pipeline_still_returns_structured_response_for_non_english(self):
        result = process_message("Wo ist meine Bestellung?")
        assert result.language == "de"
        assert result.sentiment is not None
        assert result.intent is not None
        assert result.route is not None