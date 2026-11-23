"""Explicitly establish a reviewed gate baseline; never called by CI."""

import json
from pathlib import Path

from b77.data import digest, read_rows

report = json.loads(Path("evals/results/tfidf-gate.json").read_text())
manifest = json.loads(Path("data/manifest.json").read_text())
reference = {
    "system": "tfidf",
    "macro_f1": report["macro_f1"],
    "gate_sha256": digest(read_rows(Path("evals/gate_subset.jsonl"))),
    "train_sha256": manifest["train_sha256"],
    "labels_sha256": manifest["labels_sha256"],
    "max_absolute_f1_drop": 0.01,
    "note": "Review changes to this file separately from candidate code.",
}
Path("evals/baseline.json").write_text(json.dumps(reference, indent=2) + "\n")
print(json.dumps(reference, indent=2))
