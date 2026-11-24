import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_tickets import generate  # noqa: E402
from llm_triage import parse_answer  # noqa: E402
from triage import REVIEW_THRESHOLD, run  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return run(generate(n=1200, seed=7))


def test_dataset_shape_and_labels():
    df = generate(n=200, seed=1)
    assert len(df) == 200
    assert df.ticket_id.is_unique
    assert set(df.category) <= {"Billing", "Technical", "Account", "Shipping", "Refund"}
    assert set(df.urgency) == {"High", "Normal", "Low"}
    assert df.text.str.len().min() > 10


def test_model_beats_baseline(result):
    baseline = result["results"].category.value_counts(normalize=True).max()
    assert result["metrics"]["category_accuracy"] > baseline + 0.4


def test_review_flag_matches_threshold(result):
    r = result["results"]
    low = (r.category_conf < REVIEW_THRESHOLD) | (r.urgency_conf < REVIEW_THRESHOLD)
    assert (r.human_review == low).all()


def test_every_ticket_gets_queue_and_sla(result):
    r = result["results"]
    assert r.queue.notna().all()
    assert set(r.sla_hours) <= {4, 24, 72}


def test_llm_parser_accepts_valid_json():
    a = parse_answer('Sure: {"category": "Refund", "urgency": "High", "reason": "faulty"}')
    assert a["category"] == "Refund"


def test_llm_parser_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_answer('{"category": "Sales", "urgency": "High", "reason": "x"}')
