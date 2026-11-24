"""A release must not trust a receipt for a missing, private or different artifact."""

import copy
import runpy

import pytest

MODULE = runpy.run_path("scripts/publish_github_release.py")


def proof():
    return {
        "status": "verified_public_download_and_inference",
        "models": {
            repo: {
                "revision": "a" * 40,
                "anonymous_download": True,
                "force_download": True,
                "smoke_label": "card_arrival",
            }
            for repo in MODULE["MODELS"]
        },
    }


def test_public_revisions_are_checked_without_authentication():
    visited = []

    def fetch(url):
        visited.append(url)
        return {"private": False, "sha": "a" * 40}

    MODULE["validate_public_proof"](proof(), fetch)
    assert len(visited) == 2
    assert all(url.startswith("https://huggingface.co/api/models/") for url in visited)


@pytest.mark.parametrize("field,value", [("anonymous_download", False), ("force_download", False),
                                       ("smoke_label", "wrong_intent"), ("revision", "main")])
def test_incomplete_or_wrong_sample_receipt_blocks_release(field, value):
    evidence = copy.deepcopy(proof())
    next(iter(evidence["models"].values()))[field] = value
    with pytest.raises(ValueError):
        MODULE["validate_public_proof"](evidence)


@pytest.mark.parametrize("private,revision", [(True, "a" * 40), (False, "b" * 40)])
def test_private_or_changed_hub_revision_blocks_release(private, revision):
    with pytest.raises(ValueError, match="anonymously visible"):
        MODULE["validate_public_proof"](
            proof(), lambda url: {"private": private, "sha": revision}
        )


def test_github_credential_is_never_forwarded_to_hugging_face():
    with pytest.raises(ValueError, match="credential destination"):
        MODULE["request_json"]("https://huggingface.co/api/models/example", "fixture")
