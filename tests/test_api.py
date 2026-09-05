"""Tests for the FastAPI endpoint (/chat and /).

Uses TestClient with mocked LLM calls where appropriate to ensure
fast, network-free execution.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ecommerce-support-chatbot"


def test_chat_endpoint_greeting():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "direct"
    assert data["intent"] == "greeting"
    assert "Hello" in data["response"]


@patch("src.chatbot.pipeline.generate_response")
def test_chat_endpoint_rag_order_status(mock_generate):
    mock_generate.return_value = "Your order is on the way."
    response = client.post("/chat", json={"message": "Where is my order?"})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "rag"
    assert data["intent"] == "order_status"
    assert data["response"] == "Your order is on the way."
    assert isinstance(data["retrieved_documents"], list)


def test_chat_endpoint_empty_message():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "refusal"
    assert "empty" in data["response"]


@patch("api.main.process_message")
def test_chat_endpoint_internal_error(mock_process):
    mock_process.side_effect = RuntimeError("Simulated pipeline failure")
    response = client.post("/chat", json={"message": "Test error"})
    assert response.status_code == 500
    assert "detail" in response.json()
