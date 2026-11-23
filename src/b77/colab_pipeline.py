"""Sequential v0.1 pilot for one existing free GPU; never provisions paid compute."""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from .pilot_pipeline import Pipeline
from .publish import publish, release_evidence
from .release_plan import release_plan

PILOT = {"version": "v0.1-single-seed-pilot", "seeds": [1], "teacher_systems": ["zeroshot"]}


def declare_pilot(root: Path) -> None:
    # Lock the scope before any experiment. Never relabel existing multi-seed artifacts.
    for name in ["a", "b"]:
        if any((root / f"checkpoints/student_{name}").glob("*/input_manifest.json")):
            raise ValueError("Use a fresh checkout or restore the already declared pilot")
    (root / "configs/release_plan.json").write_text(json.dumps(PILOT, indent=2) + "\n")
    for name in ["a", "b"]:
        path = root / f"configs/student_{name}.yaml"
        config = yaml.safe_load(path.read_text())
        config["seeds"] = [1]
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    teacher = yaml.safe_load((root / "configs/teacher.yaml").read_text())
    if teacher.get("source_limit") != 200 or teacher.get("study_scope") != "pilot200":
        raise ValueError("The single-seed pilot still requires the balanced 200-source teacher")


class ColabPipeline(Pipeline):
    def run(self):
        if release_plan(self.root) != PILOT:
            raise ValueError("Declare the v0.1 scope explicitly before running")
        self.seeds = [1]
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        if not memory.splitlines() or int(memory.splitlines()[0]) < 14336:
            raise RuntimeError("A free allocated GPU with at least 14 GiB is required")
        try:
            self.record("v0.1_preparing_full_gold_data", seeds=[1])
            self.cli("prepare", name="prepare")
            counts = json.loads((self.root / "data/manifest.json").read_text())["prepared_counts"]
            if counts != {"train": 8992, "validation": 1000, "test": 3080}:
                raise ValueError("Full frozen gold data must be retained")
            self.cli("train-baseline", name="baseline-training")
            self.cli("evaluate-baseline", "--split", "test", "--final-test", name="baseline-test")
            self.backup()
            threading.Thread(target=self.backup_loop, daemon=True).start()
            self.wait(
                self.launch([sys.executable, "scripts/verify_decoder.py"], "decoder-math", 0),
                "decoder-math",
            )
            self.cli(
                "train-student",
                "configs/student_a.yaml",
                "--seed",
                "1",
                "--smoke",
                name="gpu-smoke",
                device=0,
            )
            self.record("v0.1_generating_200_source_teacher_data")
            server = self.teacher_server()
            try:
                self.cli("distill", "--workers", "8", name="teacher-pilot", device=0)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
            progress = json.loads((self.root / "runs/distill-progress.json").read_text())
            if len(progress["completed_sources"]) != 200 or not progress["unique_kept"]:
                raise ValueError("A complete nonempty teacher pilot is required")
            self.backup()
            self.wait(
                self.launch(
                    [sys.executable, "scripts/verify_teacher_scorer.py"], "teacher-decoder-smoke", 0
                ),
                "teacher-decoder-smoke",
            )
            for student in ["a", "b"]:
                self.record(f"v0.1_training_student_{student}_seed1")
                self.train(student, 0)
                if self.failed.is_set():
                    raise RuntimeError("Student training failed; inspect its stage log")
                self.backup()
            self.record("v0.1_evaluating_zero_shot_teacher")
            self.evaluate_teacher(device=0, systems=("zeroshot",))
            if self.failed.is_set():
                raise RuntimeError("Teacher evaluation failed")
            self.backup()
            for student in ["a", "b"]:
                for batch in [1, 32]:
                    key = f"student_{student}-seed1"
                    self.record(f"v0.1_evaluating_{key}_batch{batch}")
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
                        name=f"eval-{key}-{batch}",
                        device=0,
                    )
                    self.backup()
            self.module(
                "b77.audit_injections",
                "--adapter",
                "checkpoints/student_b/1/adapter",
                "--calibration-report",
                "evals/results/student_b-seed1-batch1.json",
                name="injection-audit",
                device=0,
            )
            release_evidence(self.root)
            self.backup()
            self.module("b77.release", "--merge", name="merge")
            self.record("v0.1_publishing_measured_pilot")
            publish(self.root, self.token)
            self.module("b77.verify_public", name="anonymous-verification")
            proof = json.loads((self.root / "release/public_verification.json").read_text())
            if proof["status"] != "verified_public_download_and_inference":
                raise ValueError("Anonymous verification failed")
            self.record("v0.1_public_release_verified")
            self.backup()
        except Exception as error:
            self.record(
                "v0.1_stopped_check_logs_before_claiming_release",
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
    parser.add_argument("--declare-single-seed-pilot", action="store_true")
    parser.add_argument("--max-hours", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 10.5:
        parser.error("Use only the existing free session; maximum is 10.5 hours")
    root = Path.cwd()
    if args.declare_single_seed_pilot:
        declare_pilot(root)
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError("Pass the Colab secret through stdin")
    ColabPipeline(root, token, args.max_hours).run()


if __name__ == "__main__":
    main()
