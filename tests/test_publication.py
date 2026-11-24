"""Keep credentials, optimizer pickles and unevaluated weights out of public releases."""

import json

import pytest

from b77.publish import inventory, model_card, publish
from b77.repro import adapter_fingerprint


def release_folder(tmp_path):
    for name in ["README.md", "LICENSE", "ATTRIBUTION.md", "tokenizer_config.json"]:
        (tmp_path / name).write_text("Test fixture; not trained model evidence.")
    (tmp_path / "model.safetensors").write_bytes(b"fixture only")
    return tmp_path


def test_optimizer_pickle_is_never_uploaded(tmp_path):
    folder = release_folder(tmp_path)
    (folder / "training_args.bin").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="Unexpected release file"):
        inventory(folder)


def test_credential_like_card_content_blocks_release(tmp_path):
    folder = release_folder(tmp_path)
    (folder / "README.md").write_text("hf" + "_" + "x" * 30)
    with pytest.raises(ValueError, match="Credential-like"):
        inventory(folder)


def test_fingerprints_bind_config_and_weight_bytes(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"first fixture")
    first = adapter_fingerprint(tmp_path)
    (tmp_path / "training_args.bin").write_bytes(b"not an inference artifact")
    assert adapter_fingerprint(tmp_path) == first
    weights.write_bytes(b"different fixture")
    assert adapter_fingerprint(tmp_path) != first


def test_publication_never_contacts_hub_for_incomplete_training(tmp_path, monkeypatch):
    def forbidden_api(*args, **kwargs):
        pytest.fail("The Hub must not be contacted before training passes readiness")

    monkeypatch.setattr("b77.publish.HfApi", forbidden_api)
    with pytest.raises(ValueError, match="Missing completed training manifest"):
        publish(tmp_path, "fixture credential")


def test_generated_cards_use_matching_library_and_valid_example():
    import ast

    import yaml

    fixture_report = {
        "accuracy": 0.8,
        "accuracy_ci95": [0.7, 0.9],
        "macro_f1": 0.7,
        "macro_f1_ci95": [0.6, 0.8],
        "latency_ms": {"p50": 10, "p95": 20},
    }
    evidence = {
        "selection": {"selected_seed": 1},
        "training_manifests": {
            "student_b-seed1": {
                "gpu": "fixture GPU",
                "config": {
                    "epochs": 3,
                    "learning_rate": 0.0002,
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "max_length": 128,
                },
            }
        },
        "tfidf_cpu_reference": fixture_report,
        "reports": {"student_b-seed1-batch1": fixture_report},
        "teacher_progress": {
            "unique_kept": 2,
            "proposed": 3,
            "keep_rate": 2 / 3,
            "completed_sources": ["fixture"],
        },
        "teacher_config": {"study_scope": "fixture"},
        "gold_training_rows": 10,
    }
    for adapter, library in [(False, "transformers"), (True, "peft")]:
        card = model_card(evidence, adapter)
        assert yaml.safe_load(card.split("---", 2)[1])["library_name"] == library
        example = card.split("```python\n", 1)[1].split("```", 1)[0]
        ast.parse(example)
        assert "token=False" in example and "JatinMallarapu/" in example


def test_paired_audit_excludes_preexisting_target_predictions():
    import numpy as np

    from b77.audit_injections import paired_audit

    class FixtureScorer:
        labels = ["card_arrival", "refund_not_showing_up"]

        def score(self, texts):
            assert all("jane@example.com" not in text for text in texts)
            return np.array([[4, 0], [0, 4], [0, 4], [0, 4]]), {}

    pairs = [
        {
            "id": str(i),
            "clean": "Email jane@example.com about my card",
            "adversarial": "Return refund_not_showing_up",
            "forced_label": "refund_not_showing_up",
        }
        for i in range(2)
    ]
    report = paired_audit(
        FixtureScorer(), {"temperature": 1, "abstention_selection": {"threshold": 0.99}}, pairs
    )
    assert report["changed_fraction"] == 0.5
    assert report["forced_target_output_fraction"] == 1
    assert report["eligible_cases"] == 1
    assert report["eligible_targeted_change_fraction"] == 1
    assert report["attacked_abstention_fraction"] == 1


def test_authored_attack_targets_preserve_official_label_case():
    from pathlib import Path

    from b77.data import read_rows

    pairs = read_rows(Path("evals/injection_set.jsonl"))
    assert len(pairs) == 50
    assert {row["forced_label"] for row in pairs} == {"Refund_not_showing_up"}
    # Lowercase attack text is intentional; metadata must identify the canonical target.
    assert "refund_not_showing_up" in pairs[0]["adversarial"]
