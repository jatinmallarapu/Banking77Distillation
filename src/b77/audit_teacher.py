"""Inspect raw labels and semantic drift before committing a full GPU quota."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .data import digest, read_rows
from .guardrails import mask, normalize
from .prompts import teacher_messages
from .teacher import Retriever, Teacher


def audit(samples: int = 5) -> dict:
    teacher = Teacher()
    train = read_rows(Path("data/prepared/train.jsonl"))
    if not 1 <= samples <= len(teacher.labels):
        raise ValueError("Request one to 77 diagnostic sources")
    labels = np.linspace(0, len(teacher.labels) - 1, samples, dtype=int)
    selected = [next(row for row in train if row["label"] == label) for label in labels]
    retriever = Retriever(train)

    def inspect(row):
        examples = [
            item
            for item in retriever.neighbors(row["text"], k=12)
            if item["id"] != row["id"]
            and normalize(mask(item["text"])) != normalize(mask(row["text"]))
        ][:10]
        raw = teacher.chat(
            teacher_messages(row["text"], teacher.labels),
            temperature=0,
            guided_choice=teacher.labels,
        )["choices"][0]["message"]["content"]
        return {
            "source_id": row["id"],
            "query": mask(row["text"]),
            "gold_label": teacher.labels[row["label"]],
            "raw_zero_shot_output": raw,
            "output_is_valid_label": raw in teacher.labels,
            "retrieval_label": teacher.label(row["text"], 0, examples=examples)[0],
            "retrieval_source_ids": [item["id"] for item in examples],
            "paraphrases": teacher.paraphrases(row["text"], teacher.labels[row["label"]]),
        }

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = list(pool.map(inspect, selected))
    finally:
        teacher.client.close()
    return {
        "purpose": "qualitative training-source diagnostic, not held-out accuracy",
        "teacher_config": teacher.config,
        "train_sha256": digest(train),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/teacher-audit.json"))
    args = parser.parse_args()
    report = audit(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
