"""An interrupted evaluation resumes measured rows rather than inventing missing predictions."""

import json

import numpy as np
import pytest

from b77.eval_progress import score_checkpointed


class Scorer:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def score(self, texts):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted fixture")
        return np.array([[float(text), -float(text)] for text in texts]), {
            "input_tokens": len(texts),
            "output_tokens": len(texts) * 2,
        }


def test_recovers_exact_remaining_batches_and_rejects_changed_model(tmp_path):
    rows = [{"id": str(i), "text": str(i)} for i in range(9)]
    identity = {"label_count": 2, "model": "fixture-v1"}
    with pytest.raises(RuntimeError, match="interrupted"):
        score_checkpointed(rows, Scorer(fail_at=3), 2, tmp_path, identity)
    scorer = Scorer()
    values, batches, recovered = score_checkpointed(rows, scorer, 2, tmp_path, identity)
    assert recovered == 4 and scorer.calls == 3
    np.testing.assert_array_equal(values[:, 0], np.arange(9))
    assert len({batch["measurement_session"] for batch in batches}) == 2
    with pytest.raises(ValueError, match="identity changed"):
        score_checkpointed(rows, Scorer(), 2, tmp_path, {**identity, "model": "fixture-v2"})


def test_corrupt_or_wrong_shape_cache_is_not_scored_as_complete(tmp_path):
    rows = [{"id": "0", "text": "0"}]
    identity = {"label_count": 2}
    score_checkpointed(rows, Scorer(), 1, tmp_path, identity)
    path = next(tmp_path.glob("batch-*.json"))
    chunk = json.loads(path.read_text())
    chunk["batches"][0]["scores"] = [[1.0]]
    path.write_text(json.dumps(chunk))
    with pytest.raises(ValueError, match="Invalid or noncontiguous"):
        score_checkpointed(rows, Scorer(), 1, tmp_path, identity)
