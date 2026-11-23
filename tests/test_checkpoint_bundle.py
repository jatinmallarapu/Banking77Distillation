"""A resume archive must preserve one consistent teacher state and omit environment secrets."""

import json
import runpy
import zipfile
from pathlib import Path


def test_bundle_uses_joint_snapshot_and_excludes_environment(tmp_path):
    function = runpy.run_path(str(Path(__file__).parents[1] / "scripts/checkpoint_bundle.py"))[
        "bundle"
    ]
    (tmp_path / "runs").mkdir()
    state = {
        "metadata": {"completed_sources": ["train-1"]},
        "rows": [{"source_id": "train-1", "text": "rewritten query"}],
    }
    (tmp_path / "runs/distill-checkpoint.json").write_text(json.dumps(state))
    (tmp_path / ".env").write_text("PRIVATE_VALUE=must-not-archive")
    destination = tmp_path / "private-bundle.zip"
    result = function(tmp_path, destination)
    assert result["completed_teacher_sources"] == 1
    with zipfile.ZipFile(destination) as archive:
        assert ".env" not in archive.namelist()
        assert json.loads(archive.read("runs/distill-progress.json")) == state["metadata"]
        assert json.loads(archive.read("data/distilled.jsonl")) == state["rows"][0]
