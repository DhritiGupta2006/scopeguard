"""Seeded demo: a 5-page bakery website. Expected labels are hand-written by us for the self-check
in eval_demo.py. They are our own judgement, not an external benchmark."""

SCOPE = """PROJECT: 5-page website for Sweet Crumb Bakery. Fixed price ₹40,000.

1. Deliverables: Five responsive pages (Home, Menu, About, Gallery, Contact) that work on desktop and mobile.
2. Design revisions: Up to 3 rounds of design revisions are included. A round is one consolidated list of changes.
3. Contact form: One contact form that emails the owner is included.
4. Exclusions: E-commerce, online payments or ordering, a blog, SEO work, copywriting, photography and hosting are not included.
5. Content: The client supplies all text and images before development begins.
6. Additional work: Any work outside this scope is quoted separately at the hourly rate and needs written approval before it begins.
7. Timeline: Delivery within 4 weeks of receiving all content."""

RATE = 1500

THREAD = [
    ("Client", "Loved the first draft! Can we change the header color to a warmer cream instead of white?"),
    ("Me", "Great, updating it on the staging link now."),
    ("Client", "Can you make the whole thing feel more premium? Like a high-end patisserie."),
    ("Me", "Happy to. Could you tell me what feels off right now?"),
    ("Client", "Hmm, let's just add online ordering so customers can pay for cakes directly on the site."),
    ("Client", "Also a small blog section for our recipes, it's tiny, shouldn't take long."),
    ("Me", "I'll send you a quote for those."),
    ("Client", "Please change the Menu page font to something rounder."),
    ("Client", "Can you write the About page text for us? We're not good at writing."),
    ("Client", "And make the buttons rounded on all pages please."),
    ("Client", "Now the map on the Contact page should sit a bit lower."),
    ("Client", "We want to be first on Google, please set up the SEO."),
    ("Client", "Thanks so much! Launch is Friday so we're counting on you 🙏"),
    ("Client", "One more thing: add a contact form that sends enquiries to my email."),
]

# index in THREAD -> (expected verdict, expected is_revision)
EXPECTED = {
    0: ("in_scope", True),
    2: ("ambiguous", False),
    4: ("out_of_scope", False),
    5: ("out_of_scope", False),
    7: ("in_scope", True),
    8: ("out_of_scope", False),
    9: ("in_scope", True),
    10: ("out_of_scope", True),   # 4th revision round, over the limit of 3
    11: ("out_of_scope", False),
    12: ("no_request", False),
    13: ("in_scope", False),
}


def thread_text() -> str:
    return "\n".join(f"{who}: {txt}" for who, txt in THREAD)
