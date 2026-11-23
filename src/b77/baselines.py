"""Fit a transparent CPU reference; validation alone sets the abstention threshold."""

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import digest, read_rows, write_rows
from .guardrails import mask
from .metrics import choose_threshold, evaluate


class Baseline:
    def __init__(self, path: Path = Path("artifacts/tfidf.joblib")) -> None:
        # Load only our locally produced artifact. Never deserialize an untrusted pickle.
        self.bundle = joblib.load(path)
        self.labels = self.bundle["labels"]
        self.threshold = self.bundle["threshold"]["threshold"]

    def probabilities(self, texts: list[str]) -> np.ndarray:
        return self.bundle["model"].predict_proba([mask(text) for text in texts])


def train() -> dict:
    training = read_rows(Path("data/prepared/train.jsonl"))
    validation = read_rows(Path("data/prepared/validation.jsonl"))
    labels = json.loads(Path("data/labels.json").read_text())
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=70000
                ),
            ),
            ("classifier", LogisticRegression(C=4, max_iter=1000, random_state=42, solver="lbfgs")),
        ]
    )
    started = time.perf_counter()
    model.fit([mask(row["text"]) for row in training], [row["label"] for row in training])
    threshold = choose_threshold(
        np.array([r["label"] for r in validation]),
        model.predict_proba([mask(r["text"]) for r in validation]),
    )
    bundle = {
        "model": model,
        "labels": labels,
        "threshold": threshold,
        "train_sha256": digest(training),
        "validation_sha256": digest(validation),
        "training_seconds": time.perf_counter() - started,
    }
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(bundle, "artifacts/tfidf.joblib")
    report = {k: v for k, v in bundle.items() if k not in {"model", "labels"}}
    Path("evals/results").mkdir(parents=True, exist_ok=True)
    Path("evals/results/tfidf-training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def score(split: str = "gate", final: bool = False) -> dict:
    if split == "test" and not final:
        raise ValueError("Full official test evaluation requires --final-test acknowledgment")
    rows = read_rows(
        Path("evals/gate_subset.jsonl") if split == "gate" else Path(f"data/prepared/{split}.jsonl")
    )
    baseline = Baseline()
    probabilities = []
    latencies = []
    predictions = []
    baseline.probabilities(["warmup transaction query"])
    for row in rows:
        start = time.perf_counter()
        p = baseline.probabilities([row["text"]])[0]
        elapsed = (time.perf_counter() - start) * 1000
        probabilities.append(p)
        latencies.append(elapsed)
        predictions.append(
            {
                "id": row["id"],
                "text": mask(row["text"]),
                "label": int(p.argmax()),
                "confidence": float(p.max()),
                "latency_ms": elapsed,
                "input_tokens": None,
                "output_tokens": None,
                "token_count_status": "not_applicable_tfidf",
            }
        )
    report = evaluate(rows, np.array(probabilities), baseline.labels, baseline.threshold, latencies)
    report.update(
        {
            "system": "tfidf",
            "split": split,
            "case_sha256": digest(rows),
            "train_sha256": baseline.bundle["train_sha256"],
            "label_sha256": digest(baseline.labels),
            "hardware": {"device": "CPU", "platform": platform.platform(), "batch_size": 1},
            "api_cost_per_1000": 0,
            "gpu_cost_per_1000": None,
        }
    )
    write_rows(Path(f"evals/results/tfidf-{split}.jsonl"), predictions)
    Path(f"evals/results/tfidf-{split}.json").write_text(json.dumps(report, indent=2) + "\n")
    # Independent batch-32 throughput measurements; never label CPU numbers as GPU comparisons.
    timing = {}
    for size in [1, 32]:
        start = time.perf_counter()
        for offset in range(0, len(rows), size):
            baseline.probabilities([row["text"] for row in rows[offset : offset + size]])
        duration = time.perf_counter() - start
        timing[str(size)] = {
            "elapsed_seconds": duration,
            "queries_per_second": len(rows) / duration,
        }
    report["batch_throughput_cpu"] = timing
    Path(f"evals/results/tfidf-{split}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
