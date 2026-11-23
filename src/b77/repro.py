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
