"""Hand-written, naturally-phrased examples added to the training split only.

Rationale: Bitext's instructions are templated, so a model trained solely on
them is confident on template-style test data but underconfident on organic
user phrasing (see Stage 5 confidence-gap finding). These examples are NOT
added to val/test — doing so would let hand-written data influence the
metrics we report, rather than just the model's calibration.
"""

AUGMENTED_EXAMPLES = [
    # order_status
    ("where is my order", "order_status"),
    ("when will my package arrive", "order_status"),
    ("has my order shipped yet", "order_status"),
    ("i haven't received my delivery", "order_status"),
    ("can you tell me the status of my order", "order_status"),
    ("what's taking my order so long", "order_status"),
    ("is my package on the way", "order_status"),
    ("any updates on my shipment", "order_status"),
    ("how long until delivery", "order_status"),
    ("track my package please", "order_status"),
    ("my order hasn't arrived", "order_status"),
    ("when's my stuff getting here", "order_status"),

    # order_management
    ("i need to cancel my order", "order_management"),
    ("can i change my order before it ships", "order_management"),
    ("i want to place a new order", "order_management"),
    ("please cancel order number 12345", "order_management"),
    ("how do i modify my order", "order_management"),
    ("i changed my mind, cancel it", "order_management"),
    ("can i still edit my order", "order_management"),
    ("i'd like to add an item to my order", "order_management"),
    ("is it too late to cancel", "order_management"),
    ("i want to swap an item in my order", "order_management"),
    ("how much does it cost to cancel", "order_management"),
    ("what's the cancellation fee", "order_management"),

    # billing_and_refunds
    ("i want a refund", "billing_and_refunds"),
    ("my payment didn't go through", "billing_and_refunds"),
    ("can i see my invoice", "billing_and_refunds"),
    ("i was charged twice", "billing_and_refunds"),
    ("when will i get my money back", "billing_and_refunds"),
    ("i need a copy of my receipt", "billing_and_refunds"),
    ("what payment methods do you accept", "billing_and_refunds"),
    ("my card got declined", "billing_and_refunds"),
    ("i haven't received my refund yet", "billing_and_refunds"),
    ("can you check on my refund status", "billing_and_refunds"),
    ("there's a billing error on my account", "billing_and_refunds"),
    ("i think i was overcharged", "billing_and_refunds"),

    # account_management
    ("i forgot my password", "account_management"),
    ("how do i create an account", "account_management"),
    ("i want to delete my account", "account_management"),
    ("can you help me log in", "account_management"),
    ("i need to update my email address", "account_management"),
    ("how do i reset my password", "account_management"),
    ("i'm locked out of my account", "account_management"),
    ("can i change my username", "account_management"),
    ("i want to switch to a different account", "account_management"),
    ("having trouble signing up", "account_management"),
    ("how do i change my account details", "account_management"),
    ("i can't access my profile", "account_management"),

    # complaint
    ("this is really frustrating", "complaint"),
    ("i'm very unhappy with this service", "complaint"),
    ("this product arrived broken", "complaint"),
    ("i want to file a complaint", "complaint"),
    ("this is unacceptable", "complaint"),
    ("i'm disappointed with my experience", "complaint"),
    ("terrible customer service", "complaint"),
    ("i received the wrong item and no one is helping", "complaint"),
    ("i've been waiting forever with no response", "complaint"),
    ("this is the third time i've had this issue", "complaint"),
    ("i want to leave a review about my bad experience", "complaint"),
    ("nobody is answering my questions", "complaint"),

    # out_of_scope
    ("can i talk to a real person", "out_of_scope"),
    ("i need to change my shipping address", "out_of_scope"),
    ("how do i unsubscribe from your emails", "out_of_scope"),
    ("connect me to customer service", "out_of_scope"),
    ("i want to update my delivery address", "out_of_scope"),
    ("can i speak to a human agent", "out_of_scope"),
    ("stop sending me newsletters", "out_of_scope"),
    ("i need to update where you ship my orders", "out_of_scope"),
    ("this bot isn't helping, get me a human", "out_of_scope"),
    ("sign me up for your newsletter", "out_of_scope"),
    ("wrong shipping address on file", "out_of_scope"),
    ("i have a question not related to my order", "out_of_scope"),
]