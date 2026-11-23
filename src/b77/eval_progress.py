"""Recover measured score batches without mixing models, datasets or GPU configurations."""

import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from .data import digest


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def score_checkpointed(rows, scorer, batch_size: int, folder: Path, identity: dict):
    """Checkpoint every sixteen measured batches; raw query text is never cached here."""
    folder.mkdir(parents=True, exist_ok=True)
    identity = {**identity, "rows_sha256": digest(rows), "batch_size": batch_size}
    manifest = folder / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Evaluation checkpoint identity changed; preserve it and use a new path"
            )
    else:
        atomic_json(manifest, identity)
    records = []
    offset = 0
    for path in sorted(folder.glob("batch-*.json")):
        chunk = json.loads(path.read_text())
        if chunk["identity"] != digest(identity):
            raise ValueError("Evaluation chunk belongs to a different run")
        for record in chunk["batches"]:
            expected = min(batch_size, len(rows) - offset)
            values = np.asarray(record["scores"], dtype=float)
            if (
                record["offset"] != offset
                or expected <= 0
                or values.shape != (expected, identity["label_count"])
                or not np.isfinite(values).all()
                or not math.isfinite(record["elapsed_seconds"])
                or record["elapsed_seconds"] <= 0
            ):
                raise ValueError("Invalid or noncontiguous evaluation checkpoint")
            records.append(record)
            offset += expected
    resumed_rows = offset
    session = uuid.uuid4().hex
    pending: list[dict] = []

    def save_pending():
        if pending:
            atomic_json(
                folder / f"batch-{pending[0]['offset']:08d}.json",
                {"identity": digest(identity), "batches": pending},
            )
            pending.clear()

    try:
        while offset < len(rows):
            batch = rows[offset : offset + batch_size]
            started = time.perf_counter()
            values, usage = scorer.score([row["text"] for row in batch])
            elapsed = time.perf_counter() - started
            values = np.asarray(values, dtype=float)
            if (
                values.shape != (len(batch), identity["label_count"])
                or not np.isfinite(values).all()
            ):
                raise ValueError("Scorer returned invalid label scores")
            record = {
                "offset": offset,
                "scores": values.tolist(),
                "elapsed_seconds": elapsed,
                "tokens": usage,
                "measurement_session": session,
            }
            records.append(record)
            pending.append(record)
            offset += len(batch)
            if len(pending) >= 16:
                save_pending()
                print(
                    json.dumps(
                        {
                            "evaluation_rows": offset,
                            "total_rows": len(rows),
                            "checkpoint": str(folder),
                        }
                    ),
                    flush=True,
                )
    finally:
        save_pending()
    return np.concatenate([np.asarray(row["scores"]) for row in records]), records, resumed_rows
