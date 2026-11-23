"""Check real AWQ loading and cached/reference label agreement on validation queries only."""

import json
import time
from pathlib import Path

import numpy as np

from b77.data import read_rows
from b77.scoring import LocalAWQTeacherScorer


def main():
    scorer = LocalAWQTeacherScorer(read_rows(Path("data/prepared/train.jsonl")))
    texts = [r["text"] for r in read_rows(Path("data/prepared/validation.jsonl"))[:2]]
    started = time.perf_counter()
    reference, _ = scorer.score_reference(texts)
    reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cached, _ = scorer.score(texts)
    cached_seconds = time.perf_counter() - started
    if not np.isfinite(cached).all():
        raise ValueError("Nonfinite teacher likelihood scores")
    np.testing.assert_array_equal(cached.argmax(1), reference.argmax(1))
    result = {
        "status": "passed",
        "scope": "two_validation_queries_decoder_smoke_not_test_accuracy",
        "cases": len(texts),
        "same_predictions": True,
        "maximum_absolute_logprob_difference": float(np.max(np.abs(cached - reference))),
        "reference_seconds": reference_seconds,
        "cached_seconds": cached_seconds,
    }
    Path("runs/teacher-decoder-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
