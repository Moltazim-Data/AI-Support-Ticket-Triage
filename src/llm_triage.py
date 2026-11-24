"""Optional: classify tickets with a large language model (LLM) and compare
the answers with the classic model.

Needs an API key in the environment (ANTHROPIC_API_KEY) and the
`anthropic` package. Without a key the script explains what is missing and
stops; the rest of the project works without it.

Usage:
    python src/llm_triage.py --sample 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "prompts" / "triage_prompt.md").read_text()
CATEGORIES = {"Billing", "Technical", "Account", "Shipping", "Refund"}
URGENCIES = {"High", "Normal", "Low"}


def parse_answer(raw: str) -> dict:
    """Read the model's JSON answer and reject anything outside the allowed labels."""
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start : end + 1])
    if data.get("category") not in CATEGORIES or data.get("urgency") not in URGENCIES:
        raise ValueError(f"Unexpected labels: {data}")
    return data


def classify(ticket: str, client, model: str) -> dict:
    msg = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.replace("{ticket}", ticket)}],
    )
    return parse_answer(msg.content[0].text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "claude-haiku-4-5"))
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to run the LLM comparison (optional step).")
    import anthropic  # imported here so the main project does not depend on it

    client = anthropic.Anthropic()
    df = pd.read_csv(ROOT / "data" / "tickets.csv").sample(args.sample, random_state=1)
    rows = []
    for _, t in df.iterrows():
        try:
            a = classify(t.text, client, args.model)
            rows.append({"ticket_id": t.ticket_id, "category": t.category, "urgency": t.urgency,
                         "llm_category": a["category"], "llm_urgency": a["urgency"], "reason": a["reason"]})
        except Exception as e:  # keep going; failed answers are counted
            rows.append({"ticket_id": t.ticket_id, "category": t.category, "urgency": t.urgency, "error": str(e)})
    out = pd.DataFrame(rows)
    ok = out.dropna(subset=["llm_category"]) if "llm_category" in out else out.iloc[0:0]
    summary = {
        "sample": len(out),
        "valid_answers": len(ok),
        "llm_category_accuracy": round((ok.llm_category == ok.category).mean(), 3) if len(ok) else None,
        "llm_urgency_accuracy": round((ok.llm_urgency == ok.urgency).mean(), 3) if len(ok) else None,
    }
    out.to_csv(ROOT / "reports" / "llm_sample_results.csv", index=False)
    (ROOT / "reports" / "llm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
