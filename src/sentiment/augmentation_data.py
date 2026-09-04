"""Hand-written, naturally-phrased customer-support examples added to the
sentiment training split only (never val/test).

Rationale: dair-ai/emotion is Twitter text about personal emotional states
("i feel..."), so a model trained solely on it misreads plain, neutral
support questions (e.g. "Where is my package?") as Negative/Frustrated with
high confidence -- there's no Twitter-style emotional language to anchor on,
so the model falls back on weak, misleading signal.

Extra weight is given to Neutral, since that's both the smallest class in
the original mapped label space AND the one most responsible for the
misclassification of ordinary support questions.
"""

SENTIMENT_AUGMENTED_EXAMPLES = [
    # --- Neutral: plain, transactional support questions (the main fix) ---
    ("where is my package", "Neutral"),
    ("where is my order", "Neutral"),
    ("when will my order arrive", "Neutral"),
    ("can you check the status of my order", "Neutral"),
    ("how do i reset my password", "Neutral"),
    ("how do i change my shipping address", "Neutral"),
    ("what payment methods do you accept", "Neutral"),
    ("can i cancel my order", "Neutral"),
    ("how long does shipping take", "Neutral"),
    ("i want to update my account details", "Neutral"),
    ("can i get an invoice for my purchase", "Neutral"),
    ("how do i track my order", "Neutral"),
    ("what is your return policy", "Neutral"),
    ("can i change my order before it ships", "Neutral"),
    ("i need help creating an account", "Neutral"),
    ("how do i unsubscribe from your emails", "Neutral"),
    ("can i speak to a human agent", "Neutral"),
    ("do you ship internationally", "Neutral"),
    ("how do i apply a discount code", "Neutral"),
    ("what's the status of my refund", "Neutral"),

    # --- Negative/Frustrated: naturally-phrased support complaints ---
    ("this is really frustrating, my order is late", "Negative/Frustrated"),
    ("i've been waiting for weeks and nobody is helping me", "Negative/Frustrated"),
    ("this is unacceptable, i want my money back", "Negative/Frustrated"),
    ("i'm very unhappy with this service", "Negative/Frustrated"),
    ("my order arrived broken and no one is responding", "Negative/Frustrated"),
    ("i've contacted support three times with no answer", "Negative/Frustrated"),
    ("this is the worst customer service i've ever had", "Negative/Frustrated"),
    ("i'm so annoyed, my refund still hasn't come through", "Negative/Frustrated"),
    ("nobody is answering my questions and it's ridiculous", "Negative/Frustrated"),
    ("i want to file a complaint about my experience", "Negative/Frustrated"),

    # --- Positive/Satisfied: naturally-phrased support praise ---
    ("thanks so much, that solved my problem", "Positive/Satisfied"),
    ("great, my order arrived earlier than expected", "Positive/Satisfied"),
    ("i really appreciate the quick response", "Positive/Satisfied"),
    ("this was resolved perfectly, thank you", "Positive/Satisfied"),
    ("excellent service, i'm very happy with this", "Positive/Satisfied"),
    ("thank you for the fast refund", "Positive/Satisfied"),
    ("i'm glad this got sorted out so quickly", "Positive/Satisfied"),
    ("great support, exactly what i needed", "Positive/Satisfied"),
]