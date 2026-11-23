"""Exercise isolated generation and a merge that removes duplicates across GPU workers."""

import json

import httpx
import pytest

from b77 import distill_shards, teacher
from b77.data import read_rows, write_rows


def test_shards_resume_without_overlap_and_merge_unique_rows(tmp_path, monkeypatch):
    prepared = tmp_path / "data/prepared"
    prepared.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/teacher.yaml").write_text("revision: fixture\n")
    (tmp_path / "data/labels.json").write_text(json.dumps(["card_arrival"]))
    train = [{"id": f"train-{i}", "text": f"source {i}", "label": 0} for i in range(4)]
    write_rows(prepared / "train.jsonl", train)
    write_rows(prepared / "validation.jsonl", [{"id": "val", "text": "heldout query", "label": 0}])
    write_rows(prepared / "test.jsonl", [{"id": "test", "text": "test query", "label": 0}])
    _, _, _, identity = distill_shards.inputs(tmp_path)
    distill_shards.save_state(
        tmp_path,
        {
            "rows": [],
            "metadata": {
                "identity": identity,
                "completed_sources": [],
                "proposed": 0,
                "all_votes_matched": 0,
                "unique_kept": 0,
            },
        },
    )

    class FakeTeacher:
        labels = ["card_arrival"]
        config = {"revision": "fixture"}

        def __init__(self):
            self.client = httpx.Client()

        def paraphrases(self, text, intent):
            return [text + " rewritten", "shared rewritten query", "heldout query"]

        def label(self, text, temperature, seed, examples):
            return "card_arrival", {}

    monkeypatch.setattr(teacher, "Teacher", FakeTeacher)
    workers = distill_shards.prepare(tmp_path)
    with pytest.raises(ValueError, match="already exist"):
        distill_shards.prepare(tmp_path)
    for index, worker in enumerate(workers):
        monkeypatch.chdir(worker)
        result = teacher.distill(
            workers=2, partition=tmp_path / f"runs/teacher-shards/{index}/partition.json"
        )
        assert len(result["completed_sources"]) == 2
        assert (
            teacher.distill(
                workers=2, partition=tmp_path / f"runs/teacher-shards/{index}/partition.json"
            )
            == result
        )
        if index == 0:
            with pytest.raises(ValueError):
                distill_shards.merge(tmp_path)
    final = distill_shards.merge(tmp_path)
    assert final["completed_sources"] == [r["id"] for r in train]
    assert final["proposed"] == 12 and final["all_votes_matched"] == 12
    assert final["unique_kept"] == 5
    rows = read_rows(tmp_path / "data/distilled.jsonl")
    assert sum(r["text"] == "shared rewritten query" for r in rows) == 1
