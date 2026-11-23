"""Prevent resuming optimizer state against changed training data or a partial checkpoint."""

import json

import pytest

from b77.repro import checked_training_resume


def test_resume_rejects_changed_inputs_and_ignores_partial_latest(tmp_path):
    lock = {"train_sha256": "fixture", "seed": 1}
    (tmp_path / "input_manifest.json").write_text(json.dumps(lock))
    complete = tmp_path / "checkpoint-10"
    complete.mkdir()
    for name in [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]:
        (complete / name).write_text("fixture")
    (complete / "trainer_state.json").write_text(json.dumps({"global_step": 10}))
    partial = tmp_path / "checkpoint-20"
    partial.mkdir()
    (partial / "adapter_config.json").write_text("fixture")
    assert checked_training_resume(tmp_path, lock) == str(complete)
    with pytest.raises(ValueError, match="inputs differ"):
        checked_training_resume(tmp_path, {**lock, "train_sha256": "changed"})
