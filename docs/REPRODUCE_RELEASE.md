# Reproduce the v0.1 experiment

Use a fresh checkout. These commands consume existing GPU compute; they do not
provision resources or publish repositories. The public artifact's frozen
augmentation reproduces the dataset without re-running stochastic teacher generation.
The original larger notebook also includes batch 32; this protocol uses the
published single-seed, batch-1 scope explicitly.

## Prepare the pinned data and experiment configuration

Install the pinned Python 3.11 extras `.[train,teacher-eval]`, then run `b77 prepare`.
From the repository root, run this hydration script. No token is required:

```python
import json, shutil
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download
from b77.data import digest, read_rows

repo = "JatinMallarapu/qwen2.5-0.5b-banking77-distill-lora"
revision = "d825741218139d7ac28130609e7968157bf904d9"
if Path("checkpoints").exists():
    raise RuntimeError("Use a fresh checkout; preserve existing experiments")
evidence = json.loads(Path(hf_hub_download(
    repo, "release_evidence.json", revision=revision, token=False)).read_text())
if digest(read_rows(Path("data/prepared/test.jsonl"))) != evidence["test_sha256"]:
    raise ValueError("Prepared test split differs from the published experiment")
shutil.copy2(hf_hub_download(repo, "filtered_training_paraphrases.jsonl",
             revision=revision, token=False), "data/distilled.jsonl")
if digest(read_rows(Path("data/distilled.jsonl"))) != evidence["distilled_sha256"]:
    raise ValueError("Frozen augmentation fingerprint differs")
Path("runs").mkdir(exist_ok=True)
Path("runs/distill-progress.json").write_text(json.dumps(evidence["teacher_progress"]))
Path("configs/release_plan.json").write_text(json.dumps(evidence["release_plan"]))
Path("configs/teacher.yaml").write_text(yaml.safe_dump(evidence["teacher_config"]))
for student in ["a", "b"]:
    config = evidence["training_manifests"][f"student_{student}-seed1"]["config"]
    Path(f"configs/student_{student}.yaml").write_text(yaml.safe_dump(config))
```

## Train and evaluate

```sh
b77 train-baseline
b77 evaluate-baseline --split test --final-test
b77 train-student configs/student_a.yaml --seed 1
b77 train-student configs/student_b.yaml --seed 1
b77 evaluate-gpu --system zeroshot --teacher-backend local-awq --batch-size 1 --gpu-hourly-price 0 --final-test
b77 evaluate-gpu --system student_a-seed1 --adapter checkpoints/student_a/1/adapter --batch-size 1 --gpu-hourly-price 0 --final-test
b77 evaluate-gpu --system student_b-seed1 --adapter checkpoints/student_b/1/adapter --batch-size 1 --gpu-hourly-price 0 --final-test
python -m b77.audit_injections --adapter checkpoints/student_b/1/adapter --calibration-report evals/results/student_b-seed1-batch1.json
python scripts/plot_results.py
```

Supply your actual GPU hourly price instead of zero if compute is not free.
Validation alone selects checkpoint, temperature and abstention threshold. Exact
training weights and timings can differ across hardware or numerical backends;
one seed is not evidence of seed stability. Published numbers remain the original
measured reports, not guarantees for a new machine.

For recovery of your own experiment, use the atomic score checkpoints described in
[EVALUATION_RECOVERY.md](EVALUATION_RECOVERY.md). Private checkpoint credentials
are not required to download the public model or its frozen augmentation. Do not
run the publisher when reproducing somebody else's results.
