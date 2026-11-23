"""Check real split identities, normalized leakage and source-only distillation lineage."""

import json
from pathlib import Path

import pytest

from b77.data import digest, disjoint, read_rows
from b77.teacher import validate_distilled


def test_real_partitions_and_stratified_gate():
    train = read_rows(Path("data/prepared/train.jsonl"))
    val = read_rows(Path("data/prepared/validation.jsonl"))
    test = read_rows(Path("data/prepared/test.jsonl"))
    gate = read_rows(Path("evals/gate_subset.jsonl"))
    assert len(test) == 3080 and len(gate) == 300
    assert len({row["label"] for row in gate}) == 77
    assert {row["id"] for row in gate} <= {row["id"] for row in test}
    for a, b in [(train, val), (train, test), (val, test)]:
        disjoint(a, b)
    manifest = json.loads(Path("data/manifest.json").read_text())
    assert digest(gate) == manifest["gate_sha256"]


def test_normalized_leakage_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        disjoint([{"text": "  MY Card  "}], [{"text": "my card"}])


def test_generated_test_copy_is_rejected():
    train = read_rows(Path("data/prepared/train.jsonl"))
    test = read_rows(Path("data/prepared/test.jsonl"))
    names = json.loads(Path("data/labels.json").read_text())
    row = {
        "text": test[0]["text"],
        "label": train[0]["label"],
        "source_id": train[0]["id"],
        "teacher_agreement": 1.0,
        "teacher_votes": [names[train[0]["label"]]] * 3,
    }
    with pytest.raises(ValueError, match="leakage"):
        validate_distilled([row], train, test)


def test_consistent_but_wrong_teacher_votes_are_rejected():
    names = json.loads(Path("data/labels.json").read_text())
    source = [{"id": "train-fixture", "label": 0, "text": "original"}]
    row = {
        "text": "new paraphrase",
        "label": 0,
        "source_id": "train-fixture",
        "teacher_agreement": 1.0,
        "teacher_votes": [names[1]] * 3,
    }
    with pytest.raises(ValueError, match="agreement"):
        validate_distilled([row], source, [])
