"""Bind evaluation evidence to adapter bytes rather than a mutable checkpoint path."""

import hashlib
from pathlib import Path

from .data import digest


def file_fingerprint(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("Artifact fingerprinting rejects symlinks")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def adapter_fingerprint(folder: Path) -> str:
    files = sorted(folder.glob("*.safetensors"))
    config = folder / "adapter_config.json"
    if not files or not config.is_file():
        raise ValueError("A complete safetensors adapter and config are required")
    hashes = {}
    for path in [config, *files]:
        hashes[path.name] = file_fingerprint(path)
    return digest(hashes)


def checked_training_resume(folder: Path, expected: dict) -> str:
    """Select a complete optimizer checkpoint only when the original data and config match."""
    import json

    lock = folder / "input_manifest.json"
    if not lock.exists() or json.loads(lock.read_text()) != expected:
        raise ValueError("Resume inputs differ or their fingerprint lock is missing")
    candidates = sorted(
        (p for p in folder.glob("checkpoint-*") if p.name.removeprefix("checkpoint-").isdigit()),
        key=lambda p: int(p.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    for checkpoint in candidates:
        if all((checkpoint / name).is_file() for name in required):
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            if state["global_step"] == int(checkpoint.name.removeprefix("checkpoint-")):
                return str(checkpoint)
    raise ValueError(
        "No complete optimizer checkpoint exists; an adapter alone cannot resume training"
    )
