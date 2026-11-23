"""A reduced release must be explicit and cannot claim unmeasured seed variation."""

import json

import pytest

from b77.release_plan import release_plan
from b77.release import readiness


def test_default_still_requires_three_seeds(tmp_path):
    assert release_plan(tmp_path)["seeds"] == [1, 2, 3]
    with pytest.raises(ValueError, match="Missing completed"):
        readiness(tmp_path)


def test_single_seed_requires_named_plan_and_has_no_seed_std(tmp_path):
    (tmp_path / "configs").mkdir()
    plan = {"version": "v0.1-single-seed-pilot", "seeds": [1], "teacher_systems": ["zeroshot"]}
    path = tmp_path / "configs/release_plan.json"
    path.write_text(json.dumps(plan))
    for student in ["a", "b"]:
        folder = tmp_path / f"checkpoints/student_{student}/1"
        folder.mkdir(parents=True)
        (folder / "training_manifest.json").write_text(
            json.dumps(
                {
                    "status": "trained",
                    "seed": 1,
                    "best_checkpoint": "fixture-only",
                    "best_validation_loss": 1.0,
                }
            )
        )
        report = tmp_path / f"evals/results/student_{student}-seed1-batch1.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "status": "measured",
                    "cases": 3080,
                    "case_sha256": "fixture",
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                }
            )
        )
    result = readiness(tmp_path)
    assert result["release_plan"] == plan
    assert result["summary"]["a"]["accuracy_std"] is None
    assert result["summary"]["b"]["macro_f1_std"] is None
    plan["seeds"] = []
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="explicitly declared"):
        readiness(tmp_path)
