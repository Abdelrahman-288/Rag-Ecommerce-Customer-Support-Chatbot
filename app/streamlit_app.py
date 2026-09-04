"""Streamlit chat interface for the RAG-based e-commerce support chatbot.

Loads the pipeline once via st.cache_resource and reuses it across reruns.
All NLP/RAG logic lives in src/chatbot/pipeline.py -- this file only
handles UI rendering and conversation state.
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Streamlit only adds this file's own directory to sys.path, not the
# project root -- add it explicitly so `src`/`configs` imports resolve
# the same way they do when running via `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from configs.config import REPORTS_DIR
from src.chatbot.pipeline import process_message
from src.utils.logging_config import setup_logging

setup_logging()

st.set_page_config(page_title="Support Chatbot", page_icon="🛒", layout="wide")

FEEDBACK_LOG_PATH = REPORTS_DIR / "evaluation" / "feedback_log.csv"
FEEDBACK_FIELDS = [
    "timestamp",
    "user_message",
    "assistant_response",
    "language",
    "sentiment",
    "intent",
    "route",
    "feedback",
]

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# --- Shared CSS (header, FAQ buttons, chat input, typing indicator) ---
BASE_CSS = """
<style>
.support-header {
    background: linear-gradient(135deg, #6C5CE7 0%, #4E8EF7 100%);
    padding: 28px 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
}
.support-header h1 { color: white; font-size: 1.6rem; margin-bottom: 4px; }
.support-header p { color: rgba(255,255,255,0.9); font-size: 0.95rem; margin: 0; }
.faq-label { font-weight: 600; margin-top: 8px; margin-bottom: 4px; }
div[data-testid="stChatInput"] textarea { border-radius: 24px !important; }

.typing-indicator { display: flex; align-items: center; gap: 4px; padding: 4px 0; }
.typing-indicator span {
    width: 8px; height: 8px; border-radius: 50%;
    background-color: #6C5CE7;
    animation: typing-bounce 1.2s infinite ease-in-out;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
    30% { transform: translateY(-6px); opacity: 1; }
}
</style>
"""

# Covers the main app container AND Streamlit's separate sticky bottom
# container that holds the chat input -- these are distinct DOM regions
# with independent backgrounds, which is why an earlier version of this
# CSS left the input bar light while everything else went dark.
DARK_CSS = """
<style>
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stMain"],
.stApp {
    background-color: #0E1117 !important;
    color: #FAFAFA !important;
}
[data-testid="stSidebar"] {
    background-color: #1A1C24 !important;
}
[data-testid="stSidebar"] * { color: #FAFAFA !important; }
.stApp p, .stApp span, .stApp label, .stApp div, .stApp li { color: #FAFAFA; }
div[data-testid="stChatInput"] {
    background-color: #0E1117 !important;
}
div[data-testid="stChatInput"] textarea {
    background-color: #262730 !important;
    color: #FAFAFA !important;
}
div[data-testid="stChatMessage"] {
    background-color: #1A1C24 !important;
}
div[data-testid="stExpander"] {
    background-color: #1A1C24 !important;
    border-color: #3D3D4D !important;
}
.stButton button {
    background-color: #262730;
    color: #FAFAFA;
    border: 1px solid #3D3D4D;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
    color: #FAFAFA !important;
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)
if st.session_state.dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

FAQ_QUESTIONS = [
    "Where is my order?",
    "How can I get a refund?",
    "I want to change my order",
    "I can't access my account",
]


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


def log_feedback(user_message: str, assistant_response: str, debug, feedback: str) -> None:
    """Append a feedback record to a CSV log for later evaluation.

    Stored under reports/evaluation/ (git-ignored, like other generated
    evaluation artifacts) so real usage feedback doesn't get committed,
    but is available locally for the person running the app to review.
    """
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_LOG_PATH.exists()

    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response,
                "language": debug.language if debug else "",
                "sentiment": debug.sentiment if debug else "",
                "intent": debug.intent if debug else "",
                "route": debug.route if debug else "",
                "feedback": feedback,
            }
        )


# --- Gradient header (Ask AWS-style) ---
st.markdown(
    """
    <div class="support-header">
        <h1>🛒 Ask Support</h1>
        <p>Get help with orders, refunds, and account issues from our AI support assistant.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

warm_up_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "debug": ChatbotResponse|None, "feedback": None|"up"|"down"}
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


def _render_debug_panel(debug) -> None:
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


def _render_feedback_widget(msg_index: int, msg: dict) -> None:
    """Render Helpful/Not Helpful buttons for an assistant message, or a
    confirmation if feedback was already given for this message.
    """
    if msg.get("feedback"):
        label = "You found this helpful 👍" if msg["feedback"] == "up" else "You found this not helpful 👎"
        st.caption(label)
        return

    col1, col2, _ = st.columns([1, 1, 6])
    user_msg_text = st.session_state.messages[msg_index - 1]["content"] if msg_index > 0 else ""

    if col1.button("👍 Helpful", key=f"feedback_up_{msg_index}"):
        st.session_state.messages[msg_index]["feedback"] = "up"
        log_feedback(user_msg_text, msg["content"], msg.get("debug"), "up")
        st.rerun()

    if col2.button("👎 Not Helpful", key=f"feedback_down_{msg_index}"):
        st.session_state.messages[msg_index]["feedback"] = "down"
        log_feedback(user_msg_text, msg["content"], msg.get("debug"), "down")
        st.rerun()


def _handle_user_message(text: str) -> None:
    """Process a message (whether typed or from an FAQ button) through the
    pipeline and append both turns to conversation history.

    Shows an animated "typing" placeholder while process_message runs --
    Streamlit renders elements progressively as the script executes, so
    this actually appears in the browser during the (potentially several
    seconds long) pipeline call, not just at the end.
    """
    st.session_state.messages.append({"role": "user", "content": text, "debug": None, "feedback": None})
    with st.chat_message("user"):
        st.markdown(text)

    thinking_placeholder = st.empty()
    with thinking_placeholder.container():
        with st.chat_message("assistant"):
            st.markdown(
                '<div class="typing-indicator"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

    try:
        result = process_message(text)
        response_text = result.response
    except Exception as e:
        result = None
        response_text = (
            "Something went wrong processing your message. "
            "Please try again, or contact human support."
        )
        st.session_state.last_error = str(e)

    thinking_placeholder.empty()

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "debug": result, "feedback": None}
    )


# --- FAQ quick-action buttons (shown only before the conversation starts) ---
if len(st.session_state.messages) == 0:
    st.markdown('<p class="faq-label">Want help getting started? Try one of these:</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, question in enumerate(FAQ_QUESTIONS):
        if cols[i % 2].button(question, use_container_width=True, key=f"faq_{i}"):
            st.session_state.pending_input = question
            st.rerun()

# --- Process a pending FAQ click ---
if st.session_state.pending_input:
    pending = st.session_state.pending_input
    st.session_state.pending_input = None
    _handle_user_message(pending)

# --- Render conversation history ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("debug") is not None:
                _render_debug_panel(msg["debug"])
            _render_feedback_widget(i, msg)

# --- Chat input ---
user_input = st.chat_input("Ask a question...")

if user_input:
    _handle_user_message(user_input)
    st.rerun()

# --- Sidebar: dark mode toggle + reset conversation ---
with st.sidebar:
    st.header("Conversation")

    toggle_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    if st.button("🔄 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"{len(st.session_state.messages)} messages in this session")
    st.divider()
    st.caption("By chatting, you agree to our support terms.")