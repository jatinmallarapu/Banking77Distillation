"""Save a resumable private experiment bundle without caches, credentials, or base-model weights."""

import argparse
import json
import zipfile
from pathlib import Path


def bundle(root: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    names = ["pyproject.toml", "README.md", "MODEL_CARD.md"]
    for folder in [
        "src",
        "scripts",
        "configs",
        "data/prepared",
        "checkpoints",
        "evals/results",
        "runs/teacher-shards",
    ]:
        names.extend(
            str(p.relative_to(root)).replace("\\", "/")
            for p in (root / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.is_symlink()
        )
    names.append("data/labels.json")
    snapshot = root / "runs/distill-checkpoint.json"
    state = (
        json.loads(snapshot.read_text())
        if snapshot.exists()
        else {
            "metadata": json.loads((root / "runs/distill-progress.json").read_text()),
            "rows": [
                json.loads(line)
                for line in (root / "data/distilled.jsonl").read_text().splitlines()
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
            archive.writestr(
                "data/distilled.jsonl", "".join(json.dumps(r) + "\n" for r in state["rows"])
            )
            archive.writestr("runs/distill-progress.json", json.dumps(state["metadata"], indent=2))
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
