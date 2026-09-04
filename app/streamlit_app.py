"""Streamlit chat interface for the RAG-based e-commerce support chatbot.

Loads the pipeline once via st.cache_resource and reuses it across reruns.
All NLP/RAG logic lives in src/chatbot/pipeline.py -- this file only
handles UI rendering and conversation state.
"""

import sys
from pathlib import Path

# Streamlit only adds this file's own directory to sys.path, not the
# project root -- add it explicitly so `src`/`configs` imports resolve
# the same way they do when running via `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.chatbot.pipeline import process_message
from src.utils.logging_config import setup_logging

setup_logging()

st.set_page_config(page_title="Support Chatbot", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="Loading models (first message only)...")
def warm_up_pipeline():
    """Trigger model loading once at app startup rather than on first message.

    Each predictor module caches its own model at module level on first
    call, so calling process_message() here once "warms" every model
    (language, sentiment, intent, embeddings) before the user sends
    anything, avoiding a slow first real interaction.
    """
    process_message("hello")
    return True


st.title("🛒 E-commerce Customer Support Chatbot")
st.caption(
    "RAG-based support assistant — language detection, sentiment analysis, "
    "intent classification, and grounded retrieval-augmented responses."
)

warm_up_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "debug": ChatbotResponse|None}

# --- Render conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("debug") is not None:
            debug = msg["debug"]
            with st.expander("🔍 Pipeline details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Language", debug.language, f"{debug.language_confidence:.0%} confidence")
                col2.metric("Sentiment", debug.sentiment, f"{debug.sentiment_confidence:.0%} confidence")
                col3.metric(
                    "Intent",
                    debug.intent,
                    f"{debug.intent_confidence:.0%} ({debug.intent_source})",
                )
                st.write(f"**Route:** `{debug.route}`" + ("  🚩 Escalated to human agent" if debug.escalate else ""))

                if debug.retrieved_documents:
                    st.write("**Retrieved support documents:**")
                    for i, doc in enumerate(debug.retrieved_documents, 1):
                        st.markdown(
                            f"{i}. *(score: {doc.similarity_score:.3f}, "
                            f"intent: {doc.intent})* — {doc.instruction}"
                        )

# --- Chat input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "debug": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = process_message(user_input)
                response_text = result.response
            except Exception as e:
                result = None
                response_text = (
                    "Something went wrong processing your message. "
                    "Please try again, or contact human support."
                )
                st.error(f"Pipeline error: {e}")

        st.markdown(response_text)

        if result is not None:
            with st.expander("🔍 Pipeline details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Language", result.language, f"{result.language_confidence:.0%} confidence")
                col2.metric("Sentiment", result.sentiment, f"{result.sentiment_confidence:.0%} confidence")
                col3.metric(
                    "Intent",
                    result.intent,
                    f"{result.intent_confidence:.0%} ({result.intent_source})",
                )
                st.write(f"**Route:** `{result.route}`" + ("  🚩 Escalated to human agent" if result.escalate else ""))

                if result.retrieved_documents:
                    st.write("**Retrieved support documents:**")
                    for i, doc in enumerate(result.retrieved_documents, 1):
                        st.markdown(
                            f"{i}. *(score: {doc.similarity_score:.3f}, "
                            f"intent: {doc.intent})* — {doc.instruction}"
                        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "debug": result}
    )

# --- Sidebar: reset conversation ---
with st.sidebar:
    st.header("Conversation")
    if st.button("🔄 Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages in this session")