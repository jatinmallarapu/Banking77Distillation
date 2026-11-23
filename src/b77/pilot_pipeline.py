"""Run the declared pilot on existing free GPUs, with verified recovery and gated publication."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import yaml

from .backup import upload as backup_upload
from .publish import NAMESPACE, REPOSITORIES, publish, release_evidence
from .release import readiness


def validate_plan(root: Path) -> None:
    teacher = yaml.safe_load((root / "configs/teacher.yaml").read_text())
    if teacher.get("source_limit") != 200 or teacher.get("study_scope") != "pilot200":
        raise ValueError("This runner is restricted to the explicitly declared 200-source pilot")
    for student in ["a", "b"]:
        config = yaml.safe_load((root / f"configs/student_{student}.yaml").read_text())
        if config["seeds"] != [1, 2, 3]:
            raise ValueError("The pilot retains all three predeclared seeds per student")


class Pipeline:
    def __init__(self, root: Path, token: str, max_hours: float):
        self.root, self.token = root, token
        self.seeds = [1, 2, 3]
        self.deadline = time.monotonic() + max_hours * 3600
        self.stopped = threading.Event()
        self.failed = threading.Event()
        self.backup_lock = threading.Lock()
        self.children: list[subprocess.Popen] = []
        self.child_lock = threading.Lock()
        self.last_heartbeat = 0.0
        self.last_backup = 0.0
        self.phase = "initializing"
        self.env = dict(
            os.environ,
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        self.env.pop("HF_TOKEN", None)
        self.env.pop("TEACHER_API_KEY", None)
        self.env["TEACHER_BASE_URL"] = "http://127.0.0.1:8001/v1"
        (root / "runs").mkdir(exist_ok=True)

    def record(self, phase: str, **details) -> None:
        self.phase = phase
        state = {"phase": phase, "scope": "pilot200", **details}
        path = self.root / "runs/pilot_pipeline_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(state), flush=True)

    def launch(self, command: list[str], name: str, device: int | None = None):
        env = dict(self.env)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        with (self.root / f"runs/{name}.log").open("w") as output:
            process = subprocess.Popen(
                command, cwd=self.root, env=env, stdout=output, stderr=subprocess.STDOUT
            )
        with self.child_lock:
            self.children.append(process)
        return process

    def heartbeat(self):
        if time.monotonic() - self.last_heartbeat < 60:
            return
        self.last_heartbeat = time.monotonic()
        completed = []
        for path in self.root.glob("checkpoints/student_*/*/training_manifest.json"):
            try:
                if json.loads(path.read_text()).get("status") == "trained":
                    completed.append(str(path.parent.relative_to(self.root)))
            except (FileNotFoundError, json.JSONDecodeError):
                # A status observer may race the final manifest write; release gates do not.
                continue
        snapshot = self.root / "runs/distill-checkpoint.json"
        sources = (
            len(json.loads(snapshot.read_text())["metadata"]["completed_sources"])
            if snapshot.exists()
            else 0
        )
        print(
            json.dumps(
                {
                    "phase": self.phase,
                    "trained_runs": completed,
                    "teacher_sources": sources,
                    "teacher_budget": 200,
                }
            ),
            flush=True,
        )

    def wait(self, process, name: str):
        while process.poll() is None:
            if time.monotonic() > self.deadline:
                raise TimeoutError(
                    "Session budget reached; latest verified private backup is retained"
                )
            if self.failed.is_set():
                raise RuntimeError("A worker or repeated backup failure stopped the pipeline")
            self.heartbeat()
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"{name} exited {process.returncode}; inspect runs/{name}.log")

    def module(self, module: str, *arguments: str, name: str, device: int | None = None):
        process = self.launch([sys.executable, "-m", module, *arguments], name, device)
        self.wait(process, name)

    def cli(self, *arguments: str, name: str, device: int | None = None):
        self.module("b77.cli", *arguments, name=name, device=device)

    def backup(self):
        with self.backup_lock:
            result = backup_upload(self.root, self.token)
            self.last_backup = time.monotonic()
            print(
                json.dumps(
                    {
                        "backup": result["status"],
                        "commit": result["commit"],
                        "teacher_sources": result["completed_teacher_sources"],
                    }
                ),
                flush=True,
            )

    def backup_loop(self):
        previous = None
        failures = 0
        while not self.stopped.wait(30):
            try:
                paths = list(
                    self.root.glob("checkpoints/student_*/*/checkpoint-*/trainer_state.json")
                )
                paths += list(self.root.glob("checkpoints/student_*/*/training_manifest.json"))
                paths += list(self.root.glob("evals/results/progress/**/batch-*.json"))
                signature = tuple(sorted((str(p), p.stat().st_mtime_ns) for p in paths))
                if signature != previous or time.monotonic() - self.last_backup >= 900:
                    self.backup()
                    previous, failures = signature, 0
            except Exception as error:
                failures += 1
                print(
                    json.dumps({"backup_retry": failures, "error_type": type(error).__name__}),
                    flush=True,
                )
                if failures >= 3:
                    self.failed.set()
                    return

    def train(self, student: str, device: int):
        try:
            for seed in self.seeds:
                folder = self.root / f"checkpoints/student_{student}/{seed}"
                manifest = folder / "training_manifest.json"
                if (
                    manifest.exists()
                    and json.loads(manifest.read_text()).get("status") == "trained"
                ):
                    continue
                arguments = [
                    "train-student",
                    f"configs/student_{student}.yaml",
                    "--seed",
                    str(seed),
                ]
                if (folder / "input_manifest.json").exists() and any(folder.glob("checkpoint-*")):
                    arguments.append("--resume")
                self.cli(*arguments, name=f"student-{student}-seed{seed}", device=device)
        except Exception as error:
            print(
                json.dumps(
                    {
                        "training_error": student,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                ),
                flush=True,
            )
            self.failed.set()

    def teacher_server(self):
        config = yaml.safe_load((self.root / "configs/teacher.yaml").read_text())
        process = self.launch(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                config["model"],
                "--revision",
                config["revision"],
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
                "--dtype",
                "half",
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.80",
                "--swap-space",
                "0",
                "--enable-prefix-caching",
                "--disable-log-requests",
            ],
            "teacher-server",
            0,
        )
        for _ in range(360):
            if process.poll() is not None:
                raise RuntimeError("Teacher server exited; inspect runs/teacher-server.log")
            try:
                httpx.get("http://127.0.0.1:8001/health", timeout=2).raise_for_status()
                return process
            except httpx.HTTPError:
                time.sleep(1)
        raise TimeoutError("Teacher server readiness timeout")

    def evaluate_teacher(self, device: int = 1, systems: tuple = ("zeroshot", "fewshot")):
        try:
            for system in systems:
                for batch in [1, 32]:
                    path = self.root / f"evals/results/{system}-batch{batch}.json"
                    if path.exists() and json.loads(path.read_text()).get("status") == "measured":
                        continue
                    self.cli(
                        "evaluate-gpu",
                        "--system",
                        system,
                        "--batch-size",
                        str(batch),
                        "--gpu-hourly-price",
                        "0",
                        "--final-test",
                        "--teacher-backend",
                        "local-awq",
                        name=f"eval-{system}-{batch}",
                        device=device,
                    )
        except Exception as error:
            print(json.dumps({"teacher_evaluation_error": str(error)}), flush=True)
            self.failed.set()

    def run(self):
        validate_plan(self.root)
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        )
        if len(devices.splitlines()) < 2 or any(
            int(value.strip()) < 14336 for value in devices.splitlines()[:2]
        ):
            raise RuntimeError(
                "Use the already allocated free pair of GPUs with at least 14 GiB each"
            )
        try:
            self.record("preparing_frozen_data_and_cpu_reference")
            self.cli("prepare", name="prepare")
            counts = json.loads((self.root / "data/manifest.json").read_text())["prepared_counts"]
            if counts != {"train": 8992, "validation": 1000, "test": 3080}:
                raise ValueError("The pilot must retain the full declared gold data split")
            self.cli("train-baseline", name="baseline-training")
            self.cli("evaluate-baseline", "--split", "test", "--final-test", name="baseline-test")
            self.record("verifying_initial_private_recovery_before_training")
            self.backup()
            backup_thread = threading.Thread(target=self.backup_loop, daemon=True)
            backup_thread.start()
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
            self.record("teacher_pilot_and_gold_student_training")
            student_a = threading.Thread(target=self.train, args=("a", 1), daemon=True)
            student_a.start()
            server = self.teacher_server()
            try:
                self.cli("distill", "--workers", "8", name="teacher-pilot", device=0)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
            progress = json.loads((self.root / "runs/distill-progress.json").read_text())
            if len(progress["completed_sources"]) != 200 or progress["unique_kept"] == 0:
                raise ValueError("A complete nonempty 200-source teacher pilot is required")
            self.backup()
            self.wait(
                self.launch(
                    [sys.executable, "scripts/verify_teacher_scorer.py"], "teacher-decoder-smoke", 0
                ),
                "teacher-decoder-smoke",
            )
            self.record("full_gold_plus_pilot_student_training")
            student_b = threading.Thread(target=self.train, args=("b", 0), daemon=True)
            student_b.start()
            while student_a.is_alive():
                if self.failed.is_set() or time.monotonic() > self.deadline:
                    raise RuntimeError("Training stopped")
                self.heartbeat()
                time.sleep(2)
            teacher_eval = threading.Thread(target=self.evaluate_teacher, daemon=True)
            teacher_eval.start()
            while student_b.is_alive():
                if self.failed.is_set() or time.monotonic() > self.deadline:
                    raise RuntimeError("Training stopped")
                self.heartbeat()
                time.sleep(2)
            self.record("six_student_runs_trained_evaluating_all_seeds")
            self.backup()
            for student in ["a", "b"]:
                for seed in [1, 2, 3]:
                    for batch in [1, 32]:
                        key = f"student_{student}-seed{seed}"
                        self.cli(
                            "evaluate-gpu",
                            "--system",
                            key,
                            "--adapter",
                            f"checkpoints/student_{student}/{seed}/adapter",
                            "--batch-size",
                            str(batch),
                            "--gpu-hourly-price",
                            "0",
                            "--final-test",
                            name=f"eval-{key}-{batch}",
                            device=0,
                        )
            while teacher_eval.is_alive():
                if self.failed.is_set() or time.monotonic() > self.deadline:
                    raise RuntimeError("Evaluation stopped")
                self.heartbeat()
                time.sleep(2)
            if self.failed.is_set():
                raise RuntimeError("One evaluation worker failed")
            seed = readiness(self.root)["selected_seed"]
            self.record("auditing_selected_student_before_publication", selected_seed=seed)
            self.module(
                "b77.audit_injections",
                "--adapter",
                f"checkpoints/student_b/{seed}/adapter",
                "--calibration-report",
                f"evals/results/student_b-seed{seed}-batch1.json",
                name="injection-audit",
                device=0,
            )
            release_evidence(self.root)
            self.backup()
            self.module("b77.release", "--merge", name="merge")
            self.record("publishing_measured_pilot_weights")
            publish(self.root, self.token)
            self.module("b77.verify_public", name="anonymous-verification")
            proof = json.loads((self.root / "release/public_verification.json").read_text())
            if proof["status"] != "verified_public_download_and_inference":
                raise ValueError("Public model verification did not pass")
            self.record(
                "public_release_verified",
                models=[f"{NAMESPACE}/{name}" for name in REPOSITORIES.values()],
            )
            self.backup()
        except Exception as error:
            self.record(
                "stopped_not_released_unless_separately_verified",
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
    parser.add_argument("--max-hours", type=float, default=10.5)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 10.5:
        parser.error("Stay within the existing free session; maximum pipeline budget is 10.5 hours")
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError(
            "The attached secret must be piped securely; no credentials belong in source"
        )
    Pipeline(Path.cwd(), token, args.max_hours).run()


if __name__ == "__main__":
    main()
