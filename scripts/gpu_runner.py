"""Run only on an already allocated Linux GPU; never provision, purchase, upload or publish."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml


def command(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "b77.cli", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "teacher", "students", "evaluate"])
    parser.add_argument(
        "--gpu-hourly-price",
        type=float,
        default=0,
        help="Explicit accounting assumption; 0 for a free session",
    )
    args = parser.parse_args()
    import torch

    if (
        sys.platform != "linux"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3
    ):
        raise RuntimeError(
            "Use an existing Linux CUDA GPU with at least 14 GiB; nothing was started"
        )
    command("prepare")
    Path("runs").mkdir(exist_ok=True)
    teacher = yaml.safe_load(Path("configs/teacher.yaml").read_text())
    if args.stage in {"teacher", "evaluate"}:
        os.environ["TEACHER_BASE_URL"] = "http://127.0.0.1:8001/v1"
        with Path("runs/vllm-server.log").open("w") as logfile:
            process = subprocess.Popen(
                [
                    "vllm",
                    "serve",
                    teacher["model"],
                    "--revision",
                    teacher["revision"],
                    "--port",
                    "8001",
                    "--host",
                    "127.0.0.1",
                    "--dtype",
                    "half",
                    "--max-model-len",
                    "4096",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--enable-prefix-caching",
                    "--disable-log-requests",
                ],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(240):
                    if process.poll() is not None:
                        raise RuntimeError("vLLM exited; inspect the local server log")
                    try:
                        httpx.get("http://127.0.0.1:8001/health", timeout=2).raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(1)
                else:
                    raise TimeoutError("vLLM readiness timeout")
                if args.stage == "teacher":
                    command("distill")
                else:
                    for system in ["zeroshot", "fewshot"]:
                        for batch in [1, 32]:
                            command(
                                "evaluate-gpu",
                                "--system",
                                system,
                                "--batch-size",
                                str(batch),
                                "--gpu-hourly-price",
                                str(args.gpu_hourly_price),
                                "--final-test",
                            )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if args.stage == "smoke":
        command("train-student", "configs/student_a.yaml", "--seed", "17", "--smoke")
    elif args.stage == "students":
        for student in ["a", "b"]:
            for seed in [17, 42, 73]:
                command("train-student", f"configs/student_{student}.yaml", "--seed", str(seed))
    elif args.stage == "evaluate":
        # The server is stopped before loading students, so both never compete for T4 VRAM.
        for student in ["a", "b"]:
            for seed in [17, 42, 73]:
                for batch in [1, 32]:
                    command(
                        "evaluate-gpu",
                        "--system",
                        f"student_{student}-seed{seed}",
                        "--adapter",
                        f"checkpoints/student_{student}/{seed}/adapter",
                        "--batch-size",
                        str(batch),
                        "--gpu-hourly-price",
                        str(args.gpu_hourly_price),
                        "--final-test",
                    )


if __name__ == "__main__":
    main()
