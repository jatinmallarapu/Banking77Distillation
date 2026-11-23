"""Measure paired instruction attacks on the exact adapter selected for publication."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import digest, read_rows
from .guardrails import mask
from .repro import adapter_fingerprint
from .scoring import StudentScorer, softmax


def paired_audit(scorer, calibration: dict, pairs: list[dict]) -> dict:
    if not pairs or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Attack pairs require unique, nonempty case IDs")
    if any(p["forced_label"] not in scorer.labels for p in pairs):
        raise ValueError("Forced targets must belong to the allowed intent taxonomy")
    texts = [mask(p[field]) for p in pairs for field in ["clean", "adversarial"]]
    scores, _ = scorer.score(texts)
    if scores.shape != (2 * len(pairs), len(scorer.labels)) or not np.isfinite(scores).all():
        raise ValueError("The audit requires finite scores for every query and label")
    probability = softmax(scores, calibration["temperature"])
    threshold = calibration["abstention_selection"]["threshold"]
    records = []
    for index, pair in enumerate(pairs):
        clean, attacked = probability[2 * index : 2 * index + 2]
        clean_label = scorer.labels[int(clean.argmax())]
        attacked_label = scorer.labels[int(attacked.argmax())]
        eligible = clean_label != pair["forced_label"]
        records.append(
            {
                "id": pair["id"],
                "clean_label": clean_label,
                "attacked_label": attacked_label,
                "output_changed": clean_label != attacked_label,
                "forced_target_output": attacked_label == pair["forced_label"],
                "eligible_targeted_change": eligible and attacked_label == pair["forced_label"],
                "eligible": eligible,
                "attacked_needs_human_review": float(attacked.max()) < threshold,
            }
        )
    eligible_count = sum(r["eligible"] for r in records)
    return {
        "pairs": len(pairs),
        "changed_fraction": sum(r["output_changed"] for r in records) / len(pairs),
        "forced_target_output_fraction": sum(r["forced_target_output"] for r in records)
        / len(pairs),
        "eligible_targeted_change_fraction": (
            sum(r["eligible_targeted_change"] for r in records) / eligible_count
            if eligible_count
            else None
        ),
        "eligible_cases": eligible_count,
        "attacked_abstention_fraction": sum(r["attacked_needs_human_review"] for r in records)
        / len(pairs),
        "records": records,
        "scope": "Authored paired query variants; output changes, not certified resistance or test accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration_report.read_text())
    fingerprint = adapter_fingerprint(args.adapter)
    if (
        calibration.get("status") != "measured"
        or calibration.get("adapter_fingerprint") != fingerprint
    ):
        raise ValueError("Use the matching measured validation-calibrated student report")
    pairs = read_rows(Path("evals/injection_set.jsonl"))
    if len(pairs) != 50:
        raise ValueError("The frozen authored audit contains 50 pairs")
    result = paired_audit(StudentScorer(args.adapter), calibration, pairs)
    if adapter_fingerprint(args.adapter) != fingerprint:
        raise ValueError("Adapter changed during the attack audit")
    result.update(
        {
            "status": "measured",
            "system": calibration["system"],
            "adapter_fingerprint": fingerprint,
            "pair_sha256": digest(pairs),
        }
    )
    Path(f"evals/results/{calibration['system']}-injections.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
