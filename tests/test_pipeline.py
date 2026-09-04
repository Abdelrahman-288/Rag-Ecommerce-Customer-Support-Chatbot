"""End-to-end pipeline tests, covering routing behavior for each intent
category. The Groq LLM call is mocked so these tests run instantly and
don't require network access or a live API key (Stage 26 requirement).
"""

from unittest.mock import patch

from src.chatbot.pipeline import process_message


@patch("src.chatbot.pipeline.generate_response")
def test_pipeline_greeting_routes_direct_no_llm_call(mock_generate):
    result = process_message("Hi there!")
    assert result.route == "direct"
    assert result.intent == "greeting"
    mock_generate.assert_not_called()  # small talk must skip the LLM entirely


@patch("src.chatbot.pipeline.generate_response")
def test_pipeline_gratitude_routes_direct_no_llm_call(mock_generate):
    result = process_message("Thanks for your help!")
    assert result.route == "direct"
    assert result.intent == "gratitude"
    mock_generate.assert_not_called()


@patch("src.chatbot.pipeline.generate_response")
def test_pipeline_order_status_routes_to_rag(mock_generate):
    mock_generate.return_value = "Mocked grounded response about your order."
    result = process_message("Where is my order?")
    assert result.route == "rag"
    assert result.intent == "order_status"
    assert result.response == "Mocked grounded response about your order."
    mock_generate.assert_called_once()


@patch("src.chatbot.pipeline.generate_response")
def test_pipeline_complaint_routes_to_rag_escalate(mock_generate):
    mock_generate.return_value = "Mocked empathetic response."
    result = process_message("This is the worst service ever, I want a refund now!")
    assert result.route == "rag_escalate"
    assert result.escalate is True
    assert "flagged this conversation for review" in result.response


@patch("src.chatbot.pipeline.generate_response")
def test_pipeline_low_confidence_routes_to_refusal(mock_generate):
    result = process_message("asdkjaslkdj random gibberish text")
    assert result.route == "refusal"
    mock_generate.assert_not_called()  # refusal must not call the LLM


def test_pipeline_empty_message_handled_gracefully():
    result = process_message("")
    assert result.route == "refusal"
    assert result.response != ""


def test_pipeline_whitespace_only_message_handled_gracefully():
    result = process_message("   ")
    assert result.route == "refusal"


@patch("src.chatbot.pipeline.generate_response")
def test_pipeline_response_structure_has_all_fields(mock_generate):
    mock_generate.return_value = "Mocked response."
    result = process_message("How can I get a refund?")
    assert result.language is not None
    assert result.sentiment is not None
    assert result.intent is not None
    assert result.route is not None
    assert isinstance(result.retrieved_documents, list)