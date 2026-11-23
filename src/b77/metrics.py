"""Report uncertainty, calibration and abstention without silently changing denominators."""

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def calibration(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict:
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y
    records = []
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        select = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        count = int(select.sum())
        if count:
            average = float(confidence[select].mean())
            accuracy = float(correct[select].mean())
            ece += count / len(y) * abs(average - accuracy)
            records.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": count,
                    "confidence": average,
                    "accuracy": accuracy,
                }
            )
    return {"ece": ece, "bins": records}


def choose_threshold(y: np.ndarray, probability: np.ndarray, target: float = 0.90) -> dict:
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y
    choices = []
    for threshold in np.unique(np.r_[0, confidence]):
        accepted = confidence >= threshold
        if accepted.sum() >= 30 and correct[accepted].mean() >= target:
            choices.append((float(accepted.mean()), float(threshold)))
    threshold = max(choices, key=lambda pair: (pair[0], -pair[1]))[1] if choices else 1.000001
    selected = confidence >= threshold
    return {
        "threshold": threshold,
        "target_accuracy": target,
        "validation_coverage": float(selected.mean()),
        "validation_selective_accuracy": float(correct[selected].mean())
        if selected.any()
        else None,
        "selection": "Maximize validation coverage with >=30 accepted and target empirical accuracy; not a guarantee.",
    }


def evaluate(
    rows: list[dict],
    probability: np.ndarray,
    labels: list[str],
    threshold: float,
    latencies: list[float],
    bootstrap: int = 300,
) -> dict:
    y = np.array([row["label"] for row in rows])
    prediction = probability.argmax(axis=1)
    if probability.shape != (len(rows), len(labels)) or not np.isfinite(probability).all():
        raise ValueError("Invalid probability table")
    if np.any(probability < 0) or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    if len({row["id"] for row in rows}) != len(rows) or not rows:
        raise ValueError("Evaluation requires unique, nonempty case IDs")
    indices = np.arange(len(labels))
    accuracy = float(accuracy_score(y, prediction))
    macro = float(f1_score(y, prediction, labels=indices, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=indices, zero_division=0
    )
    rng = np.random.default_rng(2026)
    scores = []
    per_class = []
    # Stratified bootstrap retains all represented labels in each resample.
    groups = [np.flatnonzero(y == label) for label in indices if np.any(y == label)]
    for _ in range(bootstrap):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        scores.append(
            [
                accuracy_score(y[sample], prediction[sample]),
                f1_score(
                    y[sample], prediction[sample], labels=indices, average="macro", zero_division=0
                ),
            ]
        )
        per_class.append(
            f1_score(y[sample], prediction[sample], labels=indices, average=None, zero_division=0)
        )

    def interval(values):
        return [float(v) for v in np.percentile(values, [2.5, 97.5], axis=0)]

    confidence = probability.max(axis=1)
    accepted = confidence >= threshold
    confusion = Counter((int(a), int(b)) for a, b in zip(y, prediction) if a != b)
    pairs = [
        {
            "actual": labels[a],
            "predicted": labels[b],
            "count": count,
            "example_ids": [
                row["id"] for row, pa in zip(rows, prediction) if row["label"] == a and pa == b
            ][:3],
        }
        for (a, b), count in confusion.most_common(10)
    ]
    return {
        "cases": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro,
        "accuracy_ci95": interval(np.array(scores)[:, 0]),
        "macro_f1_ci95": interval(np.array(scores)[:, 1]),
        "bootstrap": {
            "replicates": bootstrap,
            "seed": 2026,
            "method": "stratified percentile, fixed fitted model",
        },
        "worst_classes": [
            {
                "label": labels[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "f1_ci95": interval(np.array(per_class)[:, i]),
            }
            for i in np.argsort(f1)[:5]
        ],
        "calibration": calibration(y, probability),
        "abstention": {
            "threshold": threshold,
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float((y[accepted] == prediction[accepted]).mean())
            if accepted.any()
            else None,
            "curve": [
                {
                    "threshold": float(t),
                    "coverage": float((confidence >= t).mean()),
                    "accuracy": float((y[confidence >= t] == prediction[confidence >= t]).mean())
                    if (confidence >= t).any()
                    else None,
                }
                for t in np.linspace(0, 1, 21)
            ],
        },
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "top_confused_pairs": pairs,
    }
