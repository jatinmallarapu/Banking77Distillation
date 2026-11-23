"""Back up private experiment state atomically; public model publication is a separate gate."""

import argparse
import json
import re
import runpy
import stat
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from .repro import file_fingerprint

REPOSITORY = "JatinMallarapu/Banking77TrainingCheckpoints"
SECRET = re.compile(rb"hf_[A-Za-z0-9]{20,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}")
TEXT = {".py", ".json", ".jsonl", ".yaml", ".yml", ".md", ".toml", ".txt", ".jinja"}


def downloaded_file(path: Path, cache: Path) -> Path:
    """Hub snapshots link to cache blobs; resolve only within this owned download cache."""
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(cache.resolve()) or not resolved.is_file():
        raise ValueError("Downloaded checkpoint escapes the verification cache")
    return resolved


def check_archive(path: Path) -> None:
    """Inspect text for credentials and reject traversal or symlinks before upload or extraction."""
    with zipfile.ZipFile(path) as archive:
        if (
            len(archive.infolist()) > 20000
            or sum(p.file_size for p in archive.infolist()) > 8 * 1024**3
        ):
            raise ValueError("Unexpected experiment archive size")
        for member in archive.infolist():
            name = Path(member.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.filename
                or ":" in member.filename
                or name.name.startswith(".env")
                or ".git" in name.parts
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive member")
            if name.suffix in TEXT and SECRET.search(archive.read(member)):
                raise ValueError("Credential-like text blocks checkpoint upload")


def private_api(token: str) -> HfApi:
    api = HfApi(token=token)
    if api.whoami()["name"].casefold() != "JatinMallarapu".casefold():
        raise ValueError("Checkpoint account must be JatinMallarapu")
    if not api.repo_info(REPOSITORY).private:
        raise ValueError("The checkpoint repository must remain private")
    return api


def upload(root: Path, token: str) -> dict:
    api = private_api(token)
    function = runpy.run_path(str(root / "scripts/checkpoint_bundle.py"))["bundle"]
    with tempfile.TemporaryDirectory(prefix="b77-private-backup-") as directory:
        temporary = Path(directory)
        archive = temporary / "checkpoint.zip"
        state = function(root, archive)
        check_archive(archive)
        checksum = file_fingerprint(archive)
        manifest = {
            "schema": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": checksum,
            "archive_bytes": archive.stat().st_size,
            "completed_teacher_sources": state["completed_teacher_sources"],
            "retained_paraphrases": state["retained_paraphrases"],
            "purpose": "private training recovery; not a public distilled-model release",
        }
        commit = api.create_commit(
            repo_id=REPOSITORY,
            repo_type="model",
            operations=[
                CommitOperationAdd(path_in_repo="checkpoint.zip", path_or_fileobj=str(archive)),
                CommitOperationAdd(
                    path_in_repo="checkpoint_manifest.json",
                    path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode(),
                ),
            ],
            commit_message="Checkpoint measured experiment state",
        )
        # Download the exact committed object; a successful upload response alone is not a backup check.
        downloaded = hf_hub_download(
            REPOSITORY,
            "checkpoint.zip",
            revision=commit.oid,
            token=token,
            cache_dir=temporary / "verification",
            force_download=True,
        )
        if file_fingerprint(downloaded_file(Path(downloaded), temporary / "verification")) != checksum:
            raise ValueError("Private checkpoint download failed SHA256 verification")
    receipt = {
        "status": "verified_private_backup",
        "repository": REPOSITORY,
        "commit": commit.oid,
        "private": True,
        **manifest,
    }
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs/private_backup_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def restore(destination: Path, token: str) -> dict:
    api = private_api(token)
    revision = api.repo_info(REPOSITORY).sha
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore requires an empty destination directory")
    with tempfile.TemporaryDirectory(prefix="b77-restore-") as directory:
        folder = Path(directory)
        metadata = hf_hub_download(
            REPOSITORY, "checkpoint_manifest.json", revision=revision, token=token, cache_dir=folder
        )
        manifest = json.loads(Path(metadata).read_text())
        archive = Path(
            hf_hub_download(
                REPOSITORY,
                "checkpoint.zip",
                revision=revision,
                token=token,
                cache_dir=folder,
                force_download=True,
            )
        )
        archive = downloaded_file(archive, folder)
        if file_fingerprint(archive) != manifest["archive_sha256"]:
            raise ValueError("Restore archive checksum mismatch")
        check_archive(archive)
        destination.mkdir(parents=True, exist_ok=True)
        resolved = destination.resolve()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (resolved / member.filename).resolve()
                if not target.is_relative_to(resolved):
                    raise ValueError("Archive path escapes restore destination")
            zipped.extractall(resolved)
    return {
        "status": "restored_verified_private_backup",
        "commit": revision,
        "destination": str(destination),
        **manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "restore"])
    parser.add_argument("--token-stdin", action="store_true", required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "restore" and args.destination is None:
        parser.error("Restore requires --destination")
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError("Provide the attached secret through stdin, never a command argument")
    result = (
        upload(Path.cwd(), token) if args.action == "upload" else restore(args.destination, token)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
