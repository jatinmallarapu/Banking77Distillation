"""Keep a reduced pilot explicit without weakening the default three-seed release gate."""

import json
from pathlib import Path


def release_plan(root: Path) -> dict:
    path = root / "configs/release_plan.json"
    if not path.exists():
        return {
            "version": "three-seed-pilot",
            "seeds": [1, 2, 3],
            "teacher_systems": ["zeroshot", "fewshot"],
        }
    plan = json.loads(path.read_text())
    expected = {"version": "v0.1-single-seed-pilot", "seeds": [1], "teacher_systems": ["zeroshot"]}
    if plan != expected:
        raise ValueError("Only the explicitly declared single-seed v0.1 plan is supported")
    return plan
