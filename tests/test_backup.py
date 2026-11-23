"""Private recovery must reject public destinations, embedded credentials and unsafe zip paths."""

import json
import runpy
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from b77 import backup


def test_archive_rejects_traversal_and_credentials(tmp_path):
    for index, (name, content) in enumerate(
        [
            ("../outside.txt", "fixture"),
            (".env", "fixture"),
            ("config.json", "hf" + "_" + "x" * 30),
        ]
    ):
        path = tmp_path / f"bad-{index}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name, content)
        with pytest.raises(ValueError):
            backup.check_archive(path)


def test_private_destination_is_required(monkeypatch):
    class FakeAPI:
        def __init__(self, token):
            assert token == "fixture"

        def whoami(self):
            return {"name": "JatinMallarapu"}

        def repo_info(self, repository):
            assert repository == backup.REPOSITORY
            return SimpleNamespace(private=False)

    monkeypatch.setattr(backup, "HfApi", FakeAPI)
    with pytest.raises(ValueError, match="must remain private"):
        backup.private_api("fixture")


def test_bundle_keeps_only_active_optimizer_state(tmp_path):
    function = runpy.run_path(str(Path(__file__).parents[1] / "scripts/checkpoint_bundle.py"))[
        "bundle"
    ]
    trained = tmp_path / "checkpoints/student_a/1"
    (trained / "adapter").mkdir(parents=True)
    (trained / "training_manifest.json").write_text(json.dumps({"status": "trained"}))
    (trained / "adapter/adapter_model.safetensors").write_bytes(b"fixture")
    old = trained / "checkpoint-10"
    old.mkdir()
    (old / "optimizer.pt").write_bytes(b"unneeded")
    active = tmp_path / "checkpoints/student_b/1/checkpoint-20"
    active.mkdir(parents=True)
    for name in [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]:
        (active / name).write_bytes(b"fixture")
    out = tmp_path / "checkpoint.zip"
    function(tmp_path, out)
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        assert "checkpoints/student_a/1/adapter/adapter_model.safetensors" in names
        assert "checkpoints/student_a/1/checkpoint-10/optimizer.pt" not in names
        assert "checkpoints/student_b/1/checkpoint-20/optimizer.pt" in names
        assert "runs/distill-checkpoint.json" not in names
