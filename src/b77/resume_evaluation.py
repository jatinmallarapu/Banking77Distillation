"""Evaluate recovered single-seed adapters on two existing GPUs; do not train or publish."""

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from .colab_pipeline import PILOT
from .data import digest, read_rows
from .pilot_pipeline import Pipeline
from .publish import release_evidence
from .release_plan import release_plan
from .repro import adapter_fingerprint
from .teacher import ordered_sources, validate_distilled


def validate_recovery(root: Path) -> dict:
    if release_plan(root) != PILOT:
        raise ValueError("Restore the declared single-seed pilot; do not redeclare its scope")
    rows = {
        name: read_rows(root / f"data/prepared/{name}.jsonl")
        for name in ["train", "validation", "test"]
    }
    if {name: len(value) for name, value in rows.items()} != {
        "train": 8992,
        "validation": 1000,
        "test": 3080,
    }:
        raise ValueError("Recovered gold split counts differ from the frozen study")
    distilled = read_rows(root / "data/distilled.jsonl")
    labels = json.loads((root / "data/labels.json").read_text())
    config = yaml.safe_load((root / "configs/teacher.yaml").read_text())
    progress = json.loads((root / "runs/distill-progress.json").read_text())
    excluded = rows["validation"] + rows["test"]
    expected = ordered_sources(rows["train"])[:200]
    if (
        len(labels) != 77
        or config.get("source_limit") != 200
        or set(progress["completed_sources"]) != {row["id"] for row in expected}
        or len(progress["completed_sources"]) != 200
        or progress["unique_kept"] != len(distilled)
        or progress["proposed"] != 600
        or not math.isclose(progress["keep_rate"], len(distilled) / 600)
        or progress["identity"]
        != digest(
            {
                "config": config,
                "train": rows["train"],
                "excluded_ids": [row["id"] for row in excluded],
            }
        )
    ):
        raise ValueError("Teacher snapshot does not match the declared 200-source pilot")
    validate_distilled(distilled, rows["train"], excluded, names=labels)
    fingerprints = {}
    for student in ["a", "b"]:
        folder = root / f"checkpoints/student_{student}/1"
        manifest = json.loads((folder / "training_manifest.json").read_text())
        expected_train = rows["train"] + distilled if student == "b" else rows["train"]
        if (
            manifest.get("status") != "trained"
            or manifest["seed"] != 1
            or manifest["train_sha256"] != digest(expected_train)
            or manifest["validation_sha256"] != digest(rows["validation"])
            or not math.isfinite(manifest["best_validation_loss"])
        ):
            raise ValueError("Recovered adapters do not match completed training manifests")
        fingerprints[student] = adapter_fingerprint(folder / "adapter")
    return fingerprints


class EvaluationRecovery(Pipeline):
    def run(self):
        fingerprints = validate_recovery(self.root)
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
        if len(memory) < 2 or any(int(value) < 14336 for value in memory[:2]):
            raise RuntimeError("Allocate the existing free T4 pair; this runner buys no compute")
        try:
            self.record("recovered_adapters_validated_no_retraining", fingerprints=fingerprints)
            self.backup()
            threading.Thread(target=self.backup_loop, daemon=True).start()
            self.wait(
                self.launch([sys.executable, "scripts/verify_decoder.py"], "recovery-decoder", 1),
                "recovery-decoder",
            )
            self.wait(
                self.launch(
                    [sys.executable, "scripts/verify_teacher_scorer.py"],
                    "recovery-teacher-decoder",
                    0,
                ),
                "recovery-teacher-decoder",
            )
            self.record("parallel_recovered_evaluations_teacher_gpu0_students_gpu1")
            teacher = threading.Thread(
                target=self.evaluate_teacher,
                kwargs={"device": 0, "systems": ("zeroshot",)},
                daemon=True,
            )
            teacher.start()
            for student in ["a", "b"]:
                for batch in [1, 32]:
                    key = f"student_{student}-seed1"
                    self.cli(
                        "evaluate-gpu",
                        "--system",
                        key,
                        "--adapter",
                        f"checkpoints/student_{student}/1/adapter",
                        "--batch-size",
                        str(batch),
                        "--gpu-hourly-price",
                        "0",
                        "--final-test",
                        name=f"recovery-{key}-{batch}",
                        device=1,
                    )
                    self.backup()
            self.module(
                "b77.audit_injections",
                "--adapter",
                "checkpoints/student_b/1/adapter",
                "--calibration-report",
                "evals/results/student_b-seed1-batch1.json",
                name="recovery-injection-audit",
                device=1,
            )
            self.backup()
            while teacher.is_alive():
                if self.failed.is_set():
                    raise RuntimeError("Teacher evaluation failed; preserve the latest progress")
                # wait() implements the same deadline and heartbeat for a live process.
                if time.monotonic() > self.deadline:
                    raise TimeoutError("Free session budget reached; progress remains backed up")
                self.heartbeat()
                self.stopped.wait(2)
            if self.failed.is_set():
                raise RuntimeError("Evaluation failed")
            release_evidence(self.root)
            self.record("evaluation_evidence_complete_ready_for_colab_publication")
            self.backup()
        except Exception as error:
            self.record(
                "evaluation_recovery_stopped_not_published",
                error_type=type(error).__name__,
                message=str(error),
            )
            raise
        finally:
            self.stopped.set()
            with self.child_lock:
                for process in self.children:
                    if process.poll() is None:
                        process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-stdin", action="store_true", required=True)
    parser.add_argument("--max-hours", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 10.5:
        parser.error("Use only an existing free session with at most 10.5 hours")
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError("Read the attached secret through stdin only")
    EvaluationRecovery(Path.cwd(), token, args.max_hours).run()


if __name__ == "__main__":
    main()
