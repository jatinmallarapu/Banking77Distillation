"""The first-release budget remains balanced while keeping three seeds and all gold data."""

import yaml
import pytest

from b77.pilot_pipeline import validate_plan
from b77.teacher import ordered_sources


def test_balanced_200_source_prefix_covers_all_intents():
    train = [{"id": f"{label}-{i}", "label": label} for label in range(77) for i in range(10)]
    selected = ordered_sources(train)[:200]
    counts = [sum(r["label"] == label for r in selected) for label in range(77)]
    assert min(counts) == 2 and max(counts) == 3
    assert len({r["id"] for r in selected}) == 200


def test_pilot_runner_rejects_unapproved_budget(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/teacher.yaml").write_text(
        yaml.safe_dump({"source_limit": 20, "study_scope": "pilot200"})
    )
    with pytest.raises(ValueError, match="200-source pilot"):
        validate_plan(tmp_path)
