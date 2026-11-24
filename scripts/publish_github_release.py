"""Publish the fixed v0.1 tag only after public artifacts and this commit's CI pass."""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "jatinmallarapu/Banking77Distillation"
MODELS = {
    "JatinMallarapu/qwen2.5-0.5b-banking77-distill",
    "JatinMallarapu/qwen2.5-0.5b-banking77-distill-lora",
}


def request_json(url, token=None, payload=None, missing_ok=False):
    headers = {"Accept": "application/json", "User-Agent": "Banking77-v01-release"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        if not url.startswith(f"https://api.github.com/repos/{REPOSITORY}/"):
            raise ValueError("GitHub credential destination differs from the authorized repository")
        headers.update(
            {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2026-03-10"}
        )
    request = urllib.request.Request(
        url, headers=headers, data=json.dumps(payload).encode() if payload is not None else None
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        raise RuntimeError(f"Release request failed with HTTP {error.code}") from None


def validate_public_proof(proof, fetch=request_json):
    if (
        proof.get("status") != "verified_public_download_and_inference"
        or set(proof.get("models", {})) != MODELS
    ):
        raise ValueError("Both authorized public model verification receipts are required")
    for repo, record in proof["models"].items():
        revision = record.get("revision", "")
        if (
            not re.fullmatch(r"[0-9a-f]{40}", revision)
            or record.get("anonymous_download") is not True
            or record.get("force_download") is not True
            or record.get("smoke_label") != "card_arrival"
        ):
            raise ValueError("Incomplete anonymous revision and sample-query verification")
        info = fetch(f"https://huggingface.co/api/models/{repo}/revision/{revision}")
        if info.get("private") is not False or info.get("sha") != revision:
            raise ValueError("Published model revision is not anonymously visible")


def main():
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}
    ):
        raise ValueError("Release only from the authorized repository's main branch")
    sha = os.environ.get("GITHUB_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A fixed checkout SHA is required")
    proof = json.loads(Path("release/public_verification.json").read_text())
    validate_public_proof(proof)
    notes = Path("release/NOTES_V01.md").read_text()
    token = os.environ["GITHUB_TOKEN"]
    api = f"https://api.github.com/repos/{REPOSITORY}"
    # No PR code or external artifacts execute with write access. Wait for this exact push's CI.
    for _ in range(40):
        runs = request_json(f"{api}/actions/workflows/ci.yml/runs?head_sha={sha}", token)
        matching = [r for r in runs["workflow_runs"] if r["event"] == "push"]
        if matching and matching[0]["status"] == "completed":
            if matching[0]["conclusion"] != "success":
                raise ValueError("The release commit's CPU CI did not pass")
            break
        time.sleep(30)
    else:
        raise TimeoutError("Timed out waiting for the release commit's CI")
    head = request_json(f"{api}/git/ref/heads/main", token)
    if head["object"]["sha"] != sha:
        raise ValueError("Main advanced; rerun from its current verified commit")
    tag = request_json(f"{api}/git/ref/tags/v0.1", token, missing_ok=True)
    existing = request_json(f"{api}/releases/tags/v0.1", token, missing_ok=True)
    if tag:
        if tag["object"].get("type") != "commit" or tag["object"]["sha"] != sha or not existing:
            raise ValueError("Preserve the existing tag; do not rewrite release history")
        print(existing["html_url"])
        return
    release = request_json(
        f"{api}/releases",
        token,
        {
            "tag_name": "v0.1",
            "target_commitish": sha,
            "name": "v0.1 — Banking77 single-seed distillation pilot",
            "body": notes,
            "draft": False,
            "prerelease": True,
            "make_latest": "false",
        },
    )
    print(release["html_url"])


if __name__ == "__main__":
    main()
