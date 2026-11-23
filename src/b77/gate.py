"""Reject changed or missing cases and require a live 50-query consistency check."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .baselines import Baseline
from .data import digest, read_rows


def regressed(reference: float, candidate: float) -> bool:
    return reference - candidate > 0.01 + 1e-10


def check() -> dict:
    rows = read_rows(Path("evals/gate_subset.jsonl"))
    cached = read_rows(Path("evals/results/tfidf-gate.jsonl"))
    baseline = json.loads(Path("evals/baseline.json").read_text())
    if digest(rows) != baseline["gate_sha256"] or len(rows) != 300:
        raise ValueError("Frozen gate changed; review the dataset version")
    by_id = {row["id"]: row for row in cached}
    if len(by_id) != len(cached) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Cached predictions must cover every gate case exactly once")
    labels = json.loads(Path("data/labels.json").read_text())
    model = Baseline()
    if (
        model.bundle["train_sha256"] != baseline["train_sha256"]
        or digest(labels) != baseline["labels_sha256"]
    ):
        raise ValueError("Model or label identity changed")
    predicted = [by_id[row["id"]]["label"] for row in rows]
    if any(type(label) is not int or not 0 <= label < len(labels) for label in predicted):
        raise ValueError("Cached prediction contains an invalid label")
    score = float(
        f1_score(
            [row["label"] for row in rows],
            predicted,
            labels=np.arange(len(labels)),
            average="macro",
            zero_division=0,
        )
    )
    failures = []
    if regressed(baseline["macro_f1"], score):
        failures.append("macro-F1 dropped more than one percentage point")
    sample = rows[:50]
    actual = model.probabilities([row["text"] for row in sample]).argmax(axis=1).tolist()
    if actual != [by_id[row["id"]]["label"] for row in sample]:
        failures.append("Live 50-query predictions differ from the candidate cache")
    return {
        "status": "BLOCK" if failures else "PASS",
        "macro_f1": score,
        "cases": len(rows),
        "live_check": 50,
        "reasons": failures,
    }
