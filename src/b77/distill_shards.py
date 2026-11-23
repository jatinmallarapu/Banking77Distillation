"""Partition remaining teacher sources into isolated workspaces, then merge verified checkpoints."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from .data import digest, read_rows, write_rows
from .guardrails import mask, normalize
from .teacher import validate_distilled


def inputs(root: Path):
    train = read_rows(root / "data/prepared/train.jsonl")
    excluded = read_rows(root / "data/prepared/validation.jsonl") + read_rows(
        root / "data/prepared/test.jsonl"
    )
    labels = json.loads((root / "data/labels.json").read_text())
    config = yaml.safe_load((root / "configs/teacher.yaml").read_text())
    identity = digest(
        {"config": config, "train": train, "excluded_ids": [r["id"] for r in excluded]}
    )
    return train, excluded, labels, identity


def load_state(root: Path):
    snapshot = root / "runs/distill-checkpoint.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    return {
        "metadata": json.loads((root / "runs/distill-progress.json").read_text()),
        "rows": read_rows(root / "data/distilled.jsonl"),
    }


def save_state(root: Path, state: dict):
    snapshot = root / "runs/distill-checkpoint.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_suffix(".tmp")
    temporary.write_text(json.dumps(state) + "\n")
    temporary.replace(snapshot)
    write_rows(root / "data/distilled.jsonl", state["rows"])
    (root / "runs/distill-progress.json").write_text(json.dumps(state["metadata"], indent=2) + "\n")


def validate_state(state: dict, train, excluded, labels, identity):
    meta = state["metadata"]
    completed = meta["completed_sources"]
    if (
        meta.get("identity") != identity
        or len(completed) != len(set(completed))
        or not set(completed) <= {r["id"] for r in train}
        or meta["proposed"] != 3 * len(completed)
        or meta["unique_kept"] != len(state["rows"])
        or not len(state["rows"]) <= meta["all_votes_matched"] <= meta["proposed"]
        or any(r["source_id"] not in completed for r in state["rows"])
    ):
        raise ValueError("Inconsistent teacher checkpoint")
    validate_distilled(state["rows"], train, excluded, labels)


def prepare(root: Path, count: int = 2) -> list[str]:
    if not 2 <= count <= 4:
        raise ValueError("Use two to four explicitly allocated GPU workers")
    train, excluded, labels, identity = inputs(root)
    base = load_state(root)
    validate_state(base, train, excluded, labels, identity)
    destination = root / "runs/teacher-shards"
    if destination.exists():
        raise ValueError("Shard workspaces already exist; resume them rather than overwriting")
    destination.mkdir(parents=True)
    (destination / "base.json").write_text(json.dumps(base) + "\n")
    paths = []
    for index in range(count):
        worker = destination / str(index)
        for name in [
            "configs/teacher.yaml",
            "data/labels.json",
            "data/prepared/train.jsonl",
            "data/prepared/validation.jsonl",
            "data/prepared/test.jsonl",
        ]:
            target = worker / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        save_state(worker, base)
        (worker / "partition.json").write_text(
            json.dumps({"index": index, "count": count, "base_sha256": digest(base)}) + "\n"
        )
        paths.append(str(worker))
    return paths


def merge(root: Path) -> dict:
    train, excluded, labels, identity = inputs(root)
    folder = root / "runs/teacher-shards"
    base = json.loads((folder / "base.json").read_text())
    validate_state(base, train, excluded, labels, identity)
    workers = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.isdecimal())
    count = len(workers)
    if count < 2:
        raise ValueError("Missing isolated worker checkpoints")
    completed = set(base["metadata"]["completed_sources"])
    states = []
    for index, worker in enumerate(workers):
        state = load_state(worker)
        validate_state(state, train, excluded, labels, identity)
        assignment = {"index": index, "count": count, "base_sha256": digest(base)}
        if state["metadata"].get("partition") != assignment:
            raise ValueError("Worker checkpoint does not match its frozen base or assignment")
        if state["rows"][: len(base["rows"])] != base["rows"]:
            raise ValueError("Worker modified the frozen accepted data")
        if (
            state["metadata"]["completed_sources"][: len(base["metadata"]["completed_sources"])]
            != base["metadata"]["completed_sources"]
        ):
            raise ValueError("Worker changed the frozen completed-source prefix")
        new = set(state["metadata"]["completed_sources"]) - set(
            base["metadata"]["completed_sources"]
        )
        if completed & new:
            raise ValueError("Workers processed overlapping new sources")
        completed.update(new)
        states.append(state)
    if completed != {r["id"] for r in train}:
        raise ValueError("Teacher generation is incomplete; cannot merge")
    groups = [
        [r for r in train if r["label"] == label] for label in sorted({r["label"] for r in train})
    ]
    order = [
        group[index]["id"]
        for index in range(max(map(len, groups)))
        for group in groups
        if index < len(group)
    ]
    rank = {source: index for index, source in enumerate(order)}
    candidates = list(base["rows"])
    for state in states:
        candidates.extend(state["rows"][len(base["rows"]) :])
    candidates.sort(key=lambda r: rank[r["source_id"]])
    seen = {normalize(mask(r["text"])) for r in train + excluded}
    rows = []
    for row in candidates:
        key = normalize(mask(row["text"]))
        if key not in seen:
            rows.append(row)
            seen.add(key)
    metadata = {
        "identity": identity,
        "completed_sources": order,
        "proposed": 3 * len(order),
        "unique_kept": len(rows),
    }
    metadata["all_votes_matched"] = base["metadata"]["all_votes_matched"] + sum(
        state["metadata"]["all_votes_matched"] - base["metadata"]["all_votes_matched"]
        for state in states
    )
    metadata["keep_rate"] = len(rows) / metadata["proposed"]
    final = {"metadata": metadata, "rows": rows}
    validate_state(final, train, excluded, labels, identity)
    save_state(root, final)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "merge"])
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    result = prepare(Path.cwd(), args.count) if args.action == "prepare" else merge(Path.cwd())
    print(json.dumps(result, indent=2))
