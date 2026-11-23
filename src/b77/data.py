"""Freeze disjoint splits and a stratified gate while keeping final-test access explicit."""

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import train_test_split

from .guardrails import mask, normalize


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def disjoint(left: list[dict], right: list[dict]) -> None:
    if {normalize(mask(row["text"])) for row in left} & {
        normalize(mask(row["text"])) for row in right
    }:
        raise ValueError("Normalized text leakage across dataset partitions")


def stratified_gate(rows: list[dict], size: int, seed: int) -> list[dict]:
    labels = sorted({row["label"] for row in rows})
    if size < len(labels) or size > len(rows):
        raise ValueError("Gate must represent every class and fit in the test set")
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for index, label in enumerate(labels):
        pool = [row for row in rows if row["label"] == label]
        count = size // len(labels) + int(index < size % len(labels))
        selected.extend(pool[i] for i in rng.choice(len(pool), count, replace=False))
    return sorted(selected, key=lambda row: row["id"])


def prepare() -> dict:
    config = yaml.safe_load(Path("configs/data.yaml").read_text())
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    official = {}
    for split in ["train", "test"]:
        path = cache / f"{split}.parquet"
        url = f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/data/{split}-00000-of-00001.parquet"
        if not path.exists():
            response = httpx.get(url, timeout=60, follow_redirects=True)
            response.raise_for_status()
            path.write_bytes(response.content)
        table = pq.read_table(path)
        metadata = json.loads(table.schema.metadata[b"huggingface"])
        names = metadata["info"]["features"]["label"]["names"]
        official[split] = [
            {"id": f"{split}-{i:05d}", "text": row["text"], "label": int(row["label"])}
            for i, row in enumerate(table.to_pylist())
        ]
    # Cross-split duplicates would violate the requested boundary: quarantine training copies,
    # never remove or relabel the official test records.
    test_text = {normalize(mask(row["text"])) for row in official["test"]}
    unique: dict[str, dict] = {}
    quarantined = []
    for row in official["train"]:
        text = normalize(mask(row["text"]))
        if text in test_text:
            quarantined.append(row["id"])
        elif text in unique:
            if unique[text]["label"] != row["label"]:
                raise ValueError("Conflicting duplicate training labels require manual resolution")
            quarantined.append(row["id"])
        else:
            unique[text] = row
    cleaned = list(unique.values())
    train, validation = train_test_split(
        cleaned,
        test_size=config["validation_fraction"],
        random_state=config["split_seed"],
        stratify=[row["label"] for row in cleaned],
    )
    for left, right in [
        (train, validation),
        (train, official["test"]),
        (validation, official["test"]),
    ]:
        disjoint(left, right)
    for name, rows in [("train", train), ("validation", validation), ("test", official["test"])]:
        write_rows(Path(f"data/prepared/{name}.jsonl"), sorted(rows, key=lambda row: row["id"]))
    gate = stratified_gate(official["test"], config["gate_size"], config["split_seed"])
    write_rows(Path("evals/gate_subset.jsonl"), gate)
    Path("data/labels.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": config["dataset"],
        "revision": config["revision"],
        "license": config["license"],
        "official_counts": {k: len(v) for k, v in official.items()},
        "prepared_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(official["test"]),
        },
        "quarantined_training_ids": quarantined,
        "gate_sha256": digest(gate),
        "split_seed": config["split_seed"],
        "labels_sha256": digest(names),
        "train_sha256": digest(sorted(train, key=lambda r: r["id"])),
        "validation_sha256": digest(sorted(validation, key=lambda r: r["id"])),
    }
    Path("data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
