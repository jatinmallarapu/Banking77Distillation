"""Run matched GPU comparisons, calibrating and selecting abstention only on validation."""

import json
import time
from pathlib import Path

import numpy as np

from .data import digest, read_rows, write_rows
from .guardrails import mask
from .metrics import choose_threshold, evaluate
from .scoring import RemoteScorer, StudentScorer, fit_temperature, softmax


def gpu_evaluate(
    system: str, adapter: Path | None, batch_size: int, price: float, final: bool = False
) -> dict:
    import torch

    if not final:
        raise ValueError("GPU full-test evaluation requires --final-test acknowledgment")
    if batch_size not in {1, 32} or price < 0:
        raise ValueError(
            "Predeclared batch sizes are 1 and 32; the supplied GPU hourly price must be nonnegative"
        )
    train = read_rows(Path("data/prepared/train.jsonl"))
    validation = read_rows(Path("data/prepared/validation.jsonl"))
    test = read_rows(Path("data/prepared/test.jsonl"))
    labels = json.loads(Path("data/labels.json").read_text())
    scorer = (
        StudentScorer(adapter)
        if adapter is not None
        else RemoteScorer(train, fewshot=system == "fewshot")
    )
    score_parts = []
    for offset in range(0, len(validation), batch_size):
        values, _ = scorer.score([row["text"] for row in validation[offset : offset + batch_size]])
        score_parts.append(values)
    validation_scores = np.concatenate(score_parts)
    temperature = fit_temperature(validation_scores, np.array([r["label"] for r in validation]))
    threshold = choose_threshold(
        np.array([r["label"] for r in validation]), softmax(validation_scores, temperature)
    )
    predictions = []
    probabilities: list = []
    latencies = []
    elapsed_total = 0.0
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for offset in range(0, len(test), batch_size):
        rows = test[offset : offset + batch_size]
        started = time.perf_counter()
        values, usage = scorer.score([row["text"] for row in rows])
        elapsed = time.perf_counter() - started
        elapsed_total += elapsed
        probability = softmax(values, temperature)
        probabilities.extend(probability)
        # Every row receives observed batch latency, not latency divided by batch size.
        latencies.extend([elapsed * 1000] * len(rows))
        for row, p in zip(rows, probability):
            predictions.append(
                {
                    "id": row["id"],
                    "text": mask(row["text"]),
                    "label": int(p.argmax()),
                    "confidence": float(p.max()),
                    "latency_ms": elapsed * 1000,
                    "batch_size": len(rows),
                    "token_usage_scope": "measured per batch in report; not invented per-query counts",
                }
            )
        for key in total_tokens:
            total_tokens[key] += usage[key]
    report = evaluate(test, np.array(probabilities), labels, threshold["threshold"], latencies)
    report.update(
        {
            "status": "measured",
            "system": system,
            "adapter": str(adapter) if adapter else None,
            "case_sha256": digest(test),
            "temperature": temperature,
            "abstention_selection": threshold,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "remote_vllm",
                "batch_size": batch_size,
            },
            "tokens": total_tokens,
            "elapsed_inference_seconds": elapsed_total,
            "gpu_hourly_price_assumption": price,
            "cost_per_1000": price * (elapsed_total / 3600) / len(test) * 1000,
            "decoder": "normalized log-likelihood of all 77 complete label sequences including EOS; not one greedy generation",
        }
    )
    name = f"{system}-batch{batch_size}"
    write_rows(Path(f"evals/results/{name}.jsonl"), predictions)
    Path(f"evals/results/{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
