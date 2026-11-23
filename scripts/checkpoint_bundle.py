"""Save a resumable private experiment bundle without caches, credentials, or base-model weights."""

import argparse
import json
import zipfile
from pathlib import Path


def bundle(root: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    names = [
        "pyproject.toml",
        "README.md",
        "MODEL_CARD.md",
        "LICENSE",
        "ATTRIBUTION.md",
        "evals/baseline.json",
    ]
    for folder in [
        "src",
        "scripts",
        "configs",
        "data/prepared",
        "evals/results",
        "runs/teacher-shards",
    ]:
        names.extend(
            str(p.relative_to(root)).replace("\\", "/")
            for p in (root / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.is_symlink()
            and p.suffix != ".tmp"
        )
    # Completed runs need their final adapter, not duplicate optimizer tensors.
    # An unfinished run keeps its last complete optimizer checkpoint for real resume.
    for student in ["student_a", "student_b"]:
        for seed in [1, 2, 3]:
            run = root / "checkpoints" / student / str(seed)
            for name in ["input_manifest.json", "training_manifest.json"]:
                if (run / name).is_file():
                    names.append(str((run / name).relative_to(root)).replace("\\", "/"))
            manifest = (
                json.loads((run / "training_manifest.json").read_text())
                if (run / "training_manifest.json").exists()
                else {}
            )
            if manifest.get("status") == "trained":
                folders = [run / "adapter"]
            else:
                candidates = [
                    p
                    for p in run.glob("checkpoint-*")
                    if p.name.removeprefix("checkpoint-").isdigit()
                    and (p / "trainer_state.json").is_file()
                    and all(
                        (p / name).is_file()
                        for name in [
                            "optimizer.pt",
                            "scheduler.pt",
                            "rng_state.pth",
                            "adapter_config.json",
                            "adapter_model.safetensors",
                        ]
                    )
                ]
                folders = (
                    [max(candidates, key=lambda p: int(p.name.removeprefix("checkpoint-")))]
                    if candidates
                    else []
                )
            for folder in folders:
                names.extend(
                    str(p.relative_to(root)).replace("\\", "/")
                    for p in folder.rglob("*")
                    if p.is_file() and not p.is_symlink() and p.name != "training_args.bin"
                )
    names.extend(
        [
            "data/labels.json",
            "data/manifest.json",
            "evals/gate_subset.jsonl",
            "evals/injection_set.jsonl",
            "release/release_manifest.json",
            "release/upload_receipt.json",
            "release/public_verification.json",
            "runs/pilot_pipeline_status.json",
            "runs/decoder-math-check.json",
            "runs/teacher-decoder-smoke.json",
        ]
    )
    snapshot = root / "runs/distill-checkpoint.json"
    state = (
        json.loads(snapshot.read_text())
        if snapshot.exists()
        else {
            "metadata": json.loads((root / "runs/distill-progress.json").read_text())
            if (root / "runs/distill-progress.json").exists()
            else {"completed_sources": []},
            "rows": [
                json.loads(line)
                for line in (
                    (root / "data/distilled.jsonl").read_text().splitlines()
                    if (root / "data/distilled.jsonl").exists()
                    else []
                )
            ],
        }
    )
    # Always derive both exported data and counters from a single in-memory snapshot.
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for name in sorted(set(names)):
                source = root / name
                if not source.is_file():
                    continue
                before = source.stat()
                archive.write(source, name)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Checkpoint changed while archiving: {name}; retry later")
            if snapshot.exists() or (root / "runs/distill-progress.json").exists():
                archive.writestr(
                    "data/distilled.jsonl", "".join(json.dumps(r) + "\n" for r in state["rows"])
                )
                archive.writestr(
                    "runs/distill-progress.json", json.dumps(state["metadata"], indent=2)
                )
                archive.writestr("runs/distill-checkpoint.json", json.dumps(state))
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "completed_teacher_sources": len(state["metadata"]["completed_sources"]),
        "retained_paraphrases": len(state["rows"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(Path.cwd(), args.out), indent=2))
