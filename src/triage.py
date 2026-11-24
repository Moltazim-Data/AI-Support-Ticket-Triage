"""Train, evaluate and apply the ticket triage model.

Pipeline: text -> TF-IDF (word + character n-grams) -> logistic regression,
one model for the category and one for the urgency. Tickets the model is
unsure about go to a human review queue instead of being routed blindly.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
REVIEW_THRESHOLD = 0.60          # below this confidence a person checks the ticket
MINUTES_SAVED_PER_TICKET = 1.5   # assumption for the impact estimate, see README
SLA_HOURS = {"High": 4, "Normal": 24, "Low": 72}
QUEUE = {"Billing": "Finance", "Refund": "Finance", "Technical": "Tech support",
         "Account": "Customer success", "Shipping": "Logistics"}


def build_model() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(max_iter=2000, C=4.0))])


def triage(texts: pd.Series, cat_model: Pipeline, urg_model: Pipeline) -> pd.DataFrame:
    """Return category, urgency, confidence, queue, SLA and review flag per ticket."""
    cat_p = cat_model.predict_proba(texts)
    urg_p = urg_model.predict_proba(texts)
    out = pd.DataFrame({
        "pred_category": cat_model.classes_[cat_p.argmax(1)],
        "category_conf": cat_p.max(1).round(3),
        "pred_urgency": urg_model.classes_[urg_p.argmax(1)],
        "urgency_conf": urg_p.max(1).round(3),
    }, index=texts.index)
    out["queue"] = out.pred_category.map(QUEUE)
    out["sla_hours"] = out.pred_urgency.map(SLA_HOURS)
    out["human_review"] = (out.category_conf < REVIEW_THRESHOLD) | (out.urgency_conf < REVIEW_THRESHOLD)
    return out


def run(df: pd.DataFrame, seed: int = 42) -> dict:
    train, test = train_test_split(df, test_size=0.25, random_state=seed, stratify=df.category)
    cat_model = build_model().fit(train.text, train.category)
    urg_model = build_model().fit(train.text, train.urgency)
    res = triage(test.text, cat_model, urg_model).join(test[["ticket_id", "text", "category", "urgency"]])

    auto = res[~res.human_review]
    metrics = {
        "tickets_total": int(len(df)),
        "tickets_test": int(len(test)),
        "category_accuracy": round(accuracy_score(res.category, res.pred_category), 3),
        "category_macro_f1": round(f1_score(res.category, res.pred_category, average="macro"), 3),
        "urgency_accuracy": round(accuracy_score(res.urgency, res.pred_urgency), 3),
        "high_urgency_recall": round(((res.pred_urgency == "High") & (res.urgency == "High")).sum() / (res.urgency == "High").sum(), 3),
        "auto_routed_share": round(len(auto) / len(res), 3),
        "auto_routed_category_accuracy": round(accuracy_score(auto.category, auto.pred_category), 3),
        "review_threshold": REVIEW_THRESHOLD,
        "est_hours_saved_per_1000_tickets": round(1000 * len(auto) / len(res) * MINUTES_SAVED_PER_TICKET / 60, 1),
    }
    return {"metrics": metrics, "results": res, "cat_model": cat_model,
            "report": classification_report(res.category, res.pred_category, digits=3)}


def save_outputs(out: dict) -> None:
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    m, res = out["metrics"], out["results"]
    (rep / "metrics.json").write_text(json.dumps(m, indent=2))
    res.to_csv(rep / "triaged_test_tickets.csv", index=False)

    labels = sorted(res.category.unique())
    cm = confusion_matrix(res.category, res.pred_category, labels=labels)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=120)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#0B1F3F")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Category confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(rep / "confusion_matrix.png")
    plt.close(fig)

    md = [
        "# Results", "",
        "Synthetic data. Figures show how the method works, not results from a real company.", "",
        "| Metric | Value |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in m.items()],
        "", "## Category report (test set)", "", "```", out["report"], "```",
        "", "![Confusion matrix](confusion_matrix.png)",
    ]
    (rep / "results.md").write_text("\n".join(md))


if __name__ == "__main__":
    data = pd.read_csv(ROOT / "data" / "tickets.csv")
    result = run(data)
    save_outputs(result)
    print(json.dumps(result["metrics"], indent=2))
