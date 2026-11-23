"""Keep final-test access and GPU execution explicit at the command boundary."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("train-baseline")
    commands.add_parser("gate")
    score = commands.add_parser("evaluate-baseline")
    score.add_argument("--split", choices=["gate", "validation", "test"], default="gate")
    score.add_argument("--final-test", action="store_true")
    distillation = commands.add_parser("distill")
    distillation.add_argument("--limit", type=int)
    distillation.add_argument("--workers", type=int, default=1)
    distillation.add_argument("--partition", type=Path)
    student = commands.add_parser("train-student")
    student.add_argument("config", type=Path)
    student.add_argument("--seed", type=int, required=True)
    student.add_argument("--smoke", action="store_true")
    student.add_argument("--resume", action="store_true")
    gpu = commands.add_parser("evaluate-gpu")
    gpu.add_argument("--system", required=True)
    gpu.add_argument("--adapter", type=Path)
    gpu.add_argument("--batch-size", type=int, choices=[1, 32], default=1)
    gpu.add_argument("--gpu-hourly-price", type=float, required=True)
    gpu.add_argument("--final-test", action="store_true")
    gpu.add_argument("--teacher-backend", choices=["vllm", "local-awq"], default="vllm")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        result = prepare()
    elif args.command == "train-baseline":
        from .baselines import train

        result = train()
    elif args.command == "evaluate-baseline":
        from .baselines import score as evaluate

        result = evaluate(args.split, args.final_test)
    elif args.command == "gate":
        from .gate import check

        result = check()
        print(json.dumps(result, indent=2))
        raise SystemExit(result["status"] != "PASS")
    elif args.command == "distill":
        from .teacher import distill

        result = distill(args.limit, args.workers, args.partition)
    elif args.command == "train-student":
        from .train_student import train as train_lora

        result = train_lora(args.config, args.seed, args.smoke, args.resume)
    else:
        from .evaluate import gpu_evaluate

        result = gpu_evaluate(
            args.system,
            args.adapter,
            args.batch_size,
            args.gpu_hourly_price,
            args.final_test,
            args.teacher_backend,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
