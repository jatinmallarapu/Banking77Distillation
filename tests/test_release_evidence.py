"""Exercise release provenance using synthetic files, never claiming fixture weights are trained."""

import json

import pytest
import yaml

from b77.data import digest, write_rows
from b77.publish import release_evidence
from b77.repro import adapter_fingerprint


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence_files(tmp_path):
    train = [
        {"id": "train-0", "text": "source query", "label": 0},
        {"id": "train-1", "text": "another source", "label": 1},
    ]
    validation = [{"id": "validation", "text": "validation query", "label": 0}]
    test = [{"id": f"test-{i}", "text": f"test query {i}", "label": i % 77} for i in range(3080)]
    labels = [f"intent_{i}" for i in range(77)]
    distilled = [
        {
            "text": "paraphrase zero",
            "label": 0,
            "source_id": "train-0",
            "teacher_agreement": 1,
            "teacher_votes": ["intent_0"] * 3,
            "protocol": "fixture_protocol",
        }
    ]
    for split, rows in [("train", train), ("validation", validation), ("test", test)]:
        path = tmp_path / f"data/prepared/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(path, rows)
    write_rows(tmp_path / "data/distilled.jsonl", distilled)
    put(tmp_path, "data/labels.json", labels)
    config = {"distillation_protocol": "fixture_protocol"}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/teacher.yaml").write_text(yaml.safe_dump(config))
    put(
        tmp_path,
        "runs/distill-progress.json",
        {
            "completed_sources": [r["id"] for r in train],
            "unique_kept": 1,
            "proposed": 6,
            "keep_rate": 1 / 6,
            "identity": digest(
                {
                    "config": config,
                    "train": train,
                    "excluded_ids": [r["id"] for r in validation + test],
                }
            ),
        },
    )
    for student in ["a", "b"]:
        for seed in [1, 2, 3]:
            prefix = f"checkpoints/student_{student}/{seed}"
            put(tmp_path, f"{prefix}/adapter/adapter_config.json", {"fixture": True})
            (tmp_path / prefix / "adapter/adapter_model.safetensors").write_bytes(b"fixture only")
            put(
                tmp_path,
                f"{prefix}/training_manifest.json",
                {
                    "status": "trained",
                    "seed": seed,
                    "best_checkpoint": "fixture-checkpoint",
                    "best_validation_loss": seed / 10,
                    "train_sha256": digest(train + distilled if student == "b" else train),
                    "validation_sha256": digest(validation),
                },
            )
    systems = ["zeroshot", "fewshot"] + [
        f"student_{s}-seed{n}" for s in ["a", "b"] for n in [1, 2, 3]
    ]
    for system in systems:
        for batch in [1, 32]:
            fingerprint = None
            if system.startswith("student_"):
                student, seed = system.split("-seed")
                fingerprint = adapter_fingerprint(
                    tmp_path / f"checkpoints/{student}/{seed}/adapter"
                )
            put(
                tmp_path,
                f"evals/results/{system}-batch{batch}.json",
                {
                    "status": "measured",
                    "cases": 3080,
                    "case_sha256": digest(test),
                    "system": system,
                    "hardware": {"batch_size": batch},
                    "accuracy": 0.8,
                    "macro_f1": 0.7,
                    "adapter_fingerprint": fingerprint,
                },
            )
    put(tmp_path, "evals/results/tfidf-test.json", {"cases": 3080, "case_sha256": digest(test)})
    pairs = [
        {
            "id": str(i),
            "clean": f"clean {i}",
            "adversarial": f"attack {i}",
            "forced_label": "intent_1",
        }
        for i in range(50)
    ]
    write_rows(tmp_path / "evals/injection_set.jsonl", pairs)
    put(
        tmp_path,
        "evals/results/student_b-seed1-injections.json",
        {
            "status": "measured",
            "pairs": 50,
            "pair_sha256": digest(pairs),
            "adapter_fingerprint": adapter_fingerprint(
                tmp_path / "checkpoints/student_b/1/adapter"
            ),
        },
    )
    return tmp_path


def test_complete_matching_evidence_passes_selection(evidence_files):
    result = release_evidence(evidence_files)
    assert result["selection"]["selected_seed"] == 1
    assert len(result["reports"]) == 16


def test_modified_adapter_invalidates_its_eval(evidence_files):
    (evidence_files / "checkpoints/student_b/1/adapter/adapter_model.safetensors").write_bytes(
        b"changed fixture"
    )
    with pytest.raises(ValueError, match="adapter bytes differ"):
        release_evidence(evidence_files)


def test_teacher_pilot_cannot_stand_in_for_complete_generation(evidence_files):
    path = evidence_files / "runs/distill-progress.json"
    progress = json.loads(path.read_text())
    progress["completed_sources"] = progress["completed_sources"][:1]
    path.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="pilot cannot be released"):
        release_evidence(evidence_files)
