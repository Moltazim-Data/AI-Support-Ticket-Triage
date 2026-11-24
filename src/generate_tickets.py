"""Generate a synthetic support-ticket dataset (no real customers).

Texts are built from templates with shared vocabulary, typos and
mislabelled rows so the classification task is realistic, not trivial.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
N_TICKETS = 3000

TEMPLATES = {
    "Billing": [
        "I was charged twice for my {plan} subscription this month",
        "invoice {inv} shows the wrong amount, please correct it",
        "why did my card get charged {amt} euro when I cancelled",
        "need a copy of the invoice for {month} for our accounting",
        "the VAT on my last invoice looks wrong",
        "payment failed but money left my account",
    ],
    "Technical": [
        "the app crashes when I open the {feature} page",
        "error {code} when I try to export the report",
        "dashboard is not loading since the last update",
        "login works but the {feature} button does nothing",
        "sync with {tool} stopped working yesterday",
        "page is very slow and times out after 30 seconds",
    ],
    "Account": [
        "please change the email address on my account",
        "I cannot reset my password, the link expired",
        "how do I add a new user to our team account",
        "please delete my account and my data",
        "my account is locked after too many attempts",
        "need to change the owner of the company account",
    ],
    "Shipping": [
        "my order {order} has not arrived after {days} days",
        "tracking number for order {order} does not work",
        "parcel was delivered to the wrong address",
        "can I change the delivery address for order {order}",
        "the package arrived damaged",
        "when will order {order} be shipped",
    ],
    "Refund": [
        "I want a refund for order {order}, the product is faulty",
        "returned the item {days} days ago and still no refund",
        "please refund the {amt} euro I paid by mistake",
        "how long does a refund take after the return",
        "refund was only partial, {amt} euro are missing",
        "cancel my order {order} and refund the payment",
    ],
}
URGENT_WORDS = ["urgent", "asap", "immediately", "today", "blocked", "cannot work"]
CALM_WORDS = ["when you have time", "no rush", "just a question", "quick question"]
FILLERS = ["Hello,", "Hi team,", "Good morning,", "", "", "Dear support,"]
ENDINGS = ["Thanks", "Best regards", "Thank you", "", "", "Kind regards"]
CHANNELS = ["Email", "Chat", "Web form", "Phone note"]


def _typo(text: str, rng: random.Random) -> str:
    """Swap two neighbouring letters in one word, like a quick typing error."""
    words = text.split()
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 3:
        j = rng.randrange(len(w) - 1)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def generate(n: int = N_TICKETS, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    cats = list(TEMPLATES)
    for i in range(n):
        cat = rng.choices(cats, weights=[22, 26, 18, 20, 14])[0]
        body = rng.choice(TEMPLATES[cat]).format(
            plan=rng.choice(["Basic", "Pro", "Team"]),
            inv=f"INV-{rng.randint(1000, 9999)}",
            amt=rng.choice([19, 49, 99, 120, 240]),
            month=rng.choice(["March", "April", "May", "June"]),
            feature=rng.choice(["reports", "settings", "billing", "analytics"]),
            code=rng.choice([500, 403, 404, 502]),
            tool=rng.choice(["Shopify", "Google Sheets", "Outlook"]),
            order=f"#{rng.randint(10000, 99999)}",
            days=rng.choice([3, 7, 10, 14, 21]),
        )
        r = rng.random()
        if r < 0.25:
            urgency = "High"
            body = f"{body}, {rng.choice(URGENT_WORDS)}"
        elif r < 0.45:
            urgency = "Low"
            body = f"{body}, {rng.choice(CALM_WORDS)}"
        else:
            urgency = "Normal"
        # cross-topic sentence: makes some tickets genuinely ambiguous
        if rng.random() < 0.15:
            other = rng.choice([c for c in cats if c != cat])
            body += ". Also " + rng.choice(TEMPLATES[other]).format(
                plan="Pro", inv="INV-1001", amt=49, month="May", feature="reports",
                code=500, tool="Shopify", order="#12345", days=7)
        if rng.random() < 0.3:
            body = _typo(body, rng)
        text = " ".join(x for x in [rng.choice(FILLERS), body + ".", rng.choice(ENDINGS)] if x)
        # 4 % label noise, as in real helpdesk data
        label = cat if rng.random() > 0.04 else rng.choice(cats)
        rows.append({
            "ticket_id": f"T{100000 + i}",
            "created_at": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=rng.randint(0, 525_600)),
            "channel": rng.choice(CHANNELS),
            "text": text,
            "category": label,
            "urgency": urgency,
        })
    return pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} tickets to {out}")
