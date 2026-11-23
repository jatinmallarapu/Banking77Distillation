"""Run matched GPU comparisons, calibrating and selecting abstention only on validation."""

import json
from pathlib import Path

import numpy as np

from .data import digest, read_rows, write_rows
from .guardrails import mask
from .eval_progress import score_checkpointed
from .metrics import choose_threshold, evaluate
from .repro import adapter_fingerprint, file_fingerprint
from .scoring import LocalAWQTeacherScorer, RemoteScorer, StudentScorer, fit_temperature, softmax


def gpu_evaluate(
    system: str,
    adapter: Path | None,
    batch_size: int,
    price: float,
    final: bool = False,
    teacher_backend: str = "vllm",
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
    fingerprint = adapter_fingerprint(adapter) if adapter is not None else None
    if teacher_backend not in {"vllm", "local-awq"}:
        raise ValueError("Unknown teacher scoring backend")
    scorer = (
        StudentScorer(adapter)
        if adapter is not None
        else (LocalAWQTeacherScorer if teacher_backend == "local-awq" else RemoteScorer)(
            train, fewshot=system == "fewshot"
        )
    )
    identity = {
        "schema": 1,
        "system": system,
        "adapter": fingerprint,
        "label_count": len(labels),
        "labels": digest(labels),
        "training": digest(train),
        "backend": "student" if adapter else teacher_backend,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "remote_vllm",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source": {
            name: file_fingerprint(Path(__file__).parent / name)
            for name in ["scoring.py", "prompts.py", "eval_progress.py"]
        },
        "model_config": file_fingerprint(
            Path("configs/student_a.yaml" if adapter else "configs/teacher.yaml")
        ),
    }
    cache = Path(f"evals/results/progress/{system}-batch{batch_size}")
    validation_scores, _, validation_resumed = score_checkpointed(
        validation, scorer, batch_size, cache / "validation", identity
    )
    temperature = fit_temperature(validation_scores, np.array([r["label"] for r in validation]))
    threshold = choose_threshold(
        np.array([r["label"] for r in validation]), softmax(validation_scores, temperature)
    )
    predictions = []
    probabilities: list = []
    latencies = []
    elapsed_total = 0.0
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    test_scores, measured_batches, test_resumed = score_checkpointed(
        test, scorer, batch_size, cache / "test", identity
    )
    for measured in measured_batches:
        offset = measured["offset"]
        rows = test[offset : offset + batch_size]
        values = test_scores[offset : offset + len(rows)]
        usage = measured["tokens"]
        elapsed = measured["elapsed_seconds"]
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
    if adapter is not None and fingerprint != adapter_fingerprint(adapter):
        raise ValueError("Adapter files changed during evaluation; no report was published")
    report = evaluate(test, np.array(probabilities), labels, threshold["threshold"], latencies)
    report.update(
        {
            "status": "measured",
            "evaluation_recovery": {
                "validation_rows_reused": validation_resumed,
                "test_rows_reused": test_resumed,
                "measurement_sessions": sorted(
                    {row["measurement_session"] for row in measured_batches}
                ),
                "timing_scope": "Observed batches on the declared GPU type; may span resumed sessions, not a single uninterrupted process",
            },
            "system": system,
            "adapter": str(adapter) if adapter else None,
            "adapter_fingerprint": fingerprint,
            "case_sha256": digest(test),
            "temperature": temperature,
            "abstention_selection": threshold,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "remote_vllm",
                "batch_size": batch_size,
            },
            "scoring_backend": "transformers-peft-kv-cache" if adapter else teacher_backend,
            "internal_query_microbatch": getattr(scorer, "query_microbatch", batch_size),
            "candidate_label_batch": getattr(scorer, "label_batch_size", 16),
            "candidate_cache_token_budget": getattr(scorer, "cache_budget_tokens", None),
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
