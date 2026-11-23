"""Publish measured model artifacts; never create credentials or upload a smoke adapter."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from .data import digest, read_rows
from .release import readiness
from .repro import adapter_fingerprint, file_fingerprint
from .prompts import STUDENT_SYSTEM
from .teacher import ordered_sources, validate_distilled

NAMESPACE = "JatinMallarapu"
REPOSITORIES = {
    "merged": "qwen2.5-0.5b-banking77-distill",
    "adapter": "qwen2.5-0.5b-banking77-distill-lora",
}
ALLOWED_FILES = {
    "README.md",
    "LICENSE",
    "ATTRIBUTION.md",
    "config.json",
    "generation_config.json",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "labels.json",
    "release_evidence.json",
    "filtered_training_paraphrases.jsonl",
}
SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}")


def release_evidence(root: Path) -> dict:
    selected = readiness(root)
    train = read_rows(root / "data/prepared/train.jsonl")
    validation = read_rows(root / "data/prepared/validation.jsonl")
    test = read_rows(root / "data/prepared/test.jsonl")
    labels = json.loads((root / "data/labels.json").read_text())
    distilled = read_rows(root / "data/distilled.jsonl")
    progress = json.loads((root / "runs/distill-progress.json").read_text())
    config = yaml.safe_load((root / "configs/teacher.yaml").read_text())
    identity = digest(
        {"config": config, "train": train, "excluded_ids": [r["id"] for r in validation + test]}
    )
    budget = config.get("source_limit", len(train))
    if not isinstance(budget, int) or not 1 <= budget <= len(train):
        raise ValueError("Invalid declared teacher source budget")
    expected_sources = ordered_sources(train)[:budget]
    if (
        len(test) != 3080
        or len(labels) != 77
        or not distilled
        or set(progress["completed_sources"]) != {r["id"] for r in expected_sources}
        or len(progress["completed_sources"]) != len(expected_sources)
        or progress["identity"] != identity
        or progress["unique_kept"] != len(distilled)
        or progress["proposed"] != 3 * len(expected_sources)
        or not math.isclose(progress["keep_rate"], len(distilled) / progress["proposed"])
        or any(r.get("protocol") != config["distillation_protocol"] for r in distilled)
    ):
        raise ValueError("Matching teacher data for the declared source budget is required")
    validate_distilled(distilled, train, validation + test, names=labels)
    reports = {}
    for system in ["zeroshot", "fewshot"] + [
        f"student_{student}-seed{seed}" for student in ["a", "b"] for seed in [1, 2, 3]
    ]:
        for batch in [1, 32]:
            key = f"{system}-batch{batch}"
            report = json.loads((root / f"evals/results/{key}.json").read_text())
            if (
                report.get("status") != "measured"
                or report["cases"] != len(test)
                or report["case_sha256"] != digest(test)
                or report["system"] != system
                or report["hardware"]["batch_size"] != batch
            ):
                raise ValueError(f"Incomplete or mismatched evaluation: {key}")
            for field in ["accuracy", "macro_f1"]:
                if not math.isfinite(report[field]) or not 0 <= report[field] <= 1:
                    raise ValueError(f"Invalid measured metric: {key}/{field}")
            reports[key] = report
    manifests = {}
    for student in ["a", "b"]:
        for seed in [1, 2, 3]:
            folder = root / f"checkpoints/student_{student}/{seed}"
            manifest = json.loads((folder / "training_manifest.json").read_text())
            expected = train + distilled if student == "b" else train
            if (
                manifest["train_sha256"] != digest(expected)
                or manifest["validation_sha256"] != digest(validation)
                or not math.isfinite(manifest["best_validation_loss"])
                or not (folder / "adapter/adapter_config.json").is_file()
                or not any(p.stat().st_size for p in (folder / "adapter").glob("*.safetensors"))
            ):
                raise ValueError("Training manifests and adapter files must match release data")
            fingerprint = adapter_fingerprint(folder / "adapter")
            for batch in [1, 32]:
                if (
                    reports[f"student_{student}-seed{seed}-batch{batch}"].get("adapter_fingerprint")
                    != fingerprint
                ):
                    raise ValueError("Evaluated adapter bytes differ from release weights")
            manifests[f"student_{student}-seed{seed}"] = manifest
    baseline = json.loads((root / "evals/results/tfidf-test.json").read_text())
    if baseline["case_sha256"] != digest(test) or baseline["cases"] != len(test):
        raise ValueError("The CPU baseline uses different test cases")
    selected_seed = selected["selected_seed"]
    injections = json.loads(
        (root / f"evals/results/student_b-seed{selected_seed}-injections.json").read_text()
    )
    if (
        injections.get("status") != "measured"
        or injections.get("pairs") != 50
        or injections.get("pair_sha256") != digest(read_rows(root / "evals/injection_set.jsonl"))
        or injections.get("adapter_fingerprint")
        != adapter_fingerprint(root / f"checkpoints/student_b/{selected_seed}/adapter")
    ):
        raise ValueError(
            "The selected adapter requires a matching 50-pair instruction-attack audit"
        )
    return {
        "selection": selected,
        "teacher_progress": progress,
        "reports": reports,
        "tfidf_cpu_reference": baseline,
        "labels": labels,
        "test_sha256": digest(test),
        "distilled_sha256": digest(distilled),
        "training_manifests": manifests,
        "teacher_config": config,
        "gold_training_rows": len(train),
        "teacher_source_budget": budget,
        "injections": injections,
    }


def model_card(evidence: dict, adapter: bool) -> str:
    library = "peft" if adapter else "transformers"
    selected_seed = evidence["selection"]["selected_seed"]
    selected_manifest = evidence["training_manifests"][f"student_b-seed{selected_seed}"]
    training = selected_manifest["config"]
    rows = [("TF-IDF (CPU reference)", evidence["tfidf_cpu_reference"])] + [
        (key.removesuffix("-batch1"), value)
        for key, value in evidence["reports"].items()
        if key.endswith("-batch1")
    ]
    lines = [
        "---",
        "license: apache-2.0",
        "base_model: Qwen/Qwen2.5-0.5B-Instruct",
        "datasets:",
        "- PolyAI/banking77",
        "language:",
        "- en",
        f"library_name: {library}",
        "pipeline_tag: text-generation",
        "tags: [distillation, intent-classification, banking, lora]",
        "---",
        "",
        "# Banking77 distilled intent router",
        "",
        "## Model summary and intended use",
        "",
        "A Qwen2.5-0.5B-Instruct student adapted for English banking-intent research. "
        "This personal project is not affiliated with a bank. It is not suitable for "
        "banking decisions, payments, fraud determination, or unreviewed customer deployment.",
        "",
        "## Training data and setup",
        "",
        "PolyAI Banking77 training queries plus intent-preserving Qwen2.5-7B AWQ "
        "paraphrases. Three seeded labels at temperature 0.7 must all match the source "
        "gold label; verification uses training-only retrieval excluding the source ID. "
        "Exact duplicate and validation/test overlap checks are applied. "
        "Teacher agreement is not human verification.",
        "",
        "Frozen augmentation is included as filtered_training_paraphrases.jsonl, licensed "
        "CC BY 4.0 with the Banking77 attribution. After preparing the pinned official split, "
        "use this file as data/distilled.jsonl to reproduce training without regenerating "
        "stochastic teacher rewrites. Its fingerprint is recorded in release_evidence.json.",
        "",
        f"Scope: {evidence['teacher_config'].get('study_scope', 'full')}. Teacher rewrites cover "
        f"{len(evidence['teacher_progress']['completed_sources'])} intent-balanced training sources. "
        f"Both students retain all {evidence['gold_training_rows']} gold training rows. "
        "A pilot release does not complete the larger all-source distillation study.",
        "",
        f"Accepted {evidence['teacher_progress']['unique_kept']} of "
        f"{evidence['teacher_progress']['proposed']} proposals. "
        f"Retention: {evidence['teacher_progress']['keep_rate']:.6%}.",
        "",
        "Student A uses gold-only data; Student B adds accepted teacher data. "
        "Both use seeds 1, 2, 3. The released Student B seed is selected by validation loss, "
        "never test accuracy.",
        "",
        f"Selected seed: {selected_seed}. GPU: {selected_manifest['gpu']}. "
        f"Epochs: {training['epochs']}; learning rate: {training['learning_rate']}; "
        f"LoRA rank/alpha/dropout: {training['lora_rank']}/{training['lora_alpha']}/"
        f"{training['lora_dropout']}; targets: all linear layers; "
        f"maximum length: {training['max_length']}. "
        "T4 execution uses fp16; the training code selects bf16 only on supporting hardware.",
        "",
        "## Measured results",
        "",
        "Each interval is a fixed-model, label-stratified bootstrap interval. "
        "The three seeds are shown separately; their variation is not a query-bootstrap interval.",
        "",
        "| System | Accuracy [95% CI] | Macro-F1 [95% CI] | p50 / p95 ms, batch 1 |",
        "| --- | --- | --- | --- |",
    ]
    for name, report in rows:

        def metric(field):
            lo, hi = report[f"{field}_ci95"]
            return f"{report[field]:.4f} [{lo:.4f}, {hi:.4f}]"

        latency = report["latency_ms"]
        lines.append(
            f"| {name} | {metric('accuracy')} | {metric('macro_f1')} | "
            f"{latency['p50']:.2f} / {latency['p95']:.2f} |"
        )
    lines += [
        "",
        "CPU and GPU timings are different hardware references, not matched speedups. "
        "GPU latency measures all-77-label likelihood scoring, not one greedy generation. "
        "Batch-32 timings, calibration, abstention coverage, token counts, cost assumptions "
        "and per-seed mean/standard deviation are preserved in release_evidence.json. "
        "A zero GPU price denotes free compute, not a production hosting quote.",
        "",
        "## Guardrails and limitations",
        "",
        "Abstention thresholds and temperature scaling are chosen on validation data only. "
        "Finite-label scoring is required for guaranteed label validity; free generation "
        "can still produce invalid text. PII regex masking is partial protection, not "
        "a privacy guarantee. No student-specific prompt-injection success rate is claimed "
        "unless a separate measured report is supplied. Domain shift, label ambiguity and "
        "teacher errors remain limitations.",
        "",
        "## Failure analysis",
        "",
        "The selected Student B report in release_evidence.json contains its five worst "
        "classes and most-confused intent pairs, with source case IDs. The teacher pilot "
        "investigation is documented in the linked GitHub repository.",
        "",
        "## How to use",
        "",
        "Use the pinned base model plus this adapter when library_name is peft; the merged "
        "repository loads directly with AutoModelForCausalLM and AutoTokenizer. "
        "Apply the system prompt and all-label scorer from the GitHub implementation for "
        "the evaluated classification behavior. A generation smoke check is not an evaluation.",
        "",
        "## License and attribution",
        "",
        "Model weights: Apache 2.0, based on the Qwen team's Qwen2.5-0.5B-Instruct. "
        "Training queries: PolyAI Banking77, CC BY 4.0, Casanueva et al. (2020), "
        "Efficient Intent Detection with Dual Sentence Encoders. Generated paraphrases "
        "are derived training data, not new human labels. Preserve ATTRIBUTION.md.",
        "",
        "Dataset: https://huggingface.co/datasets/PolyAI/banking77",
        "",
        "Code and reproduction: https://github.com/jatinmallarapu/Banking77Distillation",
        "",
    ]
    usage = [
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        f"repo = '{NAMESPACE}/{REPOSITORIES['adapter' if adapter else 'merged']}'",
        "tokenizer = AutoTokenizer.from_pretrained(repo, token=False)",
    ]
    if adapter:
        usage += [
            "from peft import PeftModel",
            "base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', "
            "revision='7ae557604adf67be50417f59c2c2f167def9a775', token=False)",
            "model = PeftModel.from_pretrained(base, repo, token=False)",
        ]
    else:
        usage += ["model = AutoModelForCausalLM.from_pretrained(repo, token=False)"]
    usage += [
        f"messages = [{{'role': 'system', 'content': {STUDENT_SYSTEM!r}}}, "
        "{'role': 'user', 'content': \"My new card still hasn't arrived\"}]",
        "inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, "
        "return_tensors='pt')",
        "output = model.generate(inputs, max_new_tokens=16, do_sample=False)",
        "print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))",
        "```",
        "",
    ]
    index = lines.index("## License and attribution")
    lines[index:index] = usage
    if evidence.get("injections"):
        audit = evidence["injections"]
        index = lines.index("## Failure analysis")
        lines[index:index] = [
            f"Selected-adapter instruction-attack audit: {audit['pairs']} authored pairs; "
            f"output-change rate {audit['changed_fraction']:.4%}; forced-target output rate "
            f"{audit['forced_target_output_fraction']:.4%}; attacked abstention rate "
            f"{audit['attacked_abstention_fraction']:.4%}. "
            "These small authored cases do not certify resistance to arbitrary attacks.",
            "",
        ]
    return "\n".join(lines)


def inventory(folder: Path) -> dict[str, str]:
    if folder.is_symlink():
        raise ValueError("Release folders cannot be symlinks")
    hashes = {}
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks cannot enter a public model release")
        if not path.is_file():
            continue
        if path.parent != folder or (
            path.name not in ALLOWED_FILES and path.suffix != ".safetensors"
        ):
            raise ValueError(f"Unexpected release file: {path.name}")
        if path.suffix != ".safetensors" and SECRET.search(path.read_text(encoding="utf-8")):
            raise ValueError(f"Credential-like content in {path.name}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        hashes[path.name] = hasher.hexdigest()
    if not any(name.endswith(".safetensors") for name in hashes):
        raise ValueError("No model weights are present")
    for required in ["README.md", "LICENSE", "ATTRIBUTION.md", "tokenizer_config.json"]:
        if required not in hashes:
            raise ValueError(f"Missing release file: {required}")
    return hashes


def publish(root: Path, token: str) -> dict:
    evidence = release_evidence(root)
    if not token:
        raise ValueError("A write token must be supplied securely")
    selected_seed = evidence["selection"]["selected_seed"]
    source = root / f"checkpoints/student_b/{selected_seed}/adapter"
    source_fingerprint = adapter_fingerprint(source)
    manifest = json.loads((root / "release/release_manifest.json").read_text())
    merged = root / "release/merged"
    if (
        adapter_fingerprint(root / "release/adapter") != source_fingerprint
        or manifest.get("selected_adapter_fingerprint") != source_fingerprint
        or manifest.get("selected_seed") != selected_seed
        or manifest.get("merged_weights_sha256")
        != {path.name: file_fingerprint(path) for path in merged.glob("*.safetensors")}
        or not manifest.get("merged_weights_sha256")
    ):
        raise ValueError("Merged release artifacts do not match the selected evaluated adapter")
    prepared = {}
    for kind in REPOSITORIES:
        folder = root / "release" / kind
        (folder / "README.md").write_text(model_card(evidence, adapter=kind == "adapter"))
        (folder / "labels.json").write_text(json.dumps(evidence["labels"]) + "\n")
        (folder / "release_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        frozen_data = folder / "filtered_training_paraphrases.jsonl"
        frozen_data.write_bytes((root / "data/distilled.jsonl").read_bytes())
        if digest(read_rows(frozen_data)) != evidence["distilled_sha256"]:
            raise ValueError("Frozen public augmentation data differs from the evaluated training data")
        prepared[kind] = inventory(folder)
    api = HfApi(token=token)
    if api.whoami()["name"].casefold() != NAMESPACE.casefold():
        raise ValueError("Token owner does not match the authorized Hugging Face namespace")
    uploaded = {}
    for kind, name in REPOSITORIES.items():
        repo_id = f"{NAMESPACE}/{name}"
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=root / "release" / kind,
            allow_patterns=list(prepared[kind]),
            commit_message="Release measured Banking77 student",
        )
        uploaded[repo_id] = {"commit": commit.oid, "files_sha256": prepared[kind]}
    result = {"status": "uploaded_pending_anonymous_verification", "models": uploaded}
    (root / "release/upload_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--token-stdin",
        action="store_true",
        help="Receive a token through a pipe, never a command-line argument",
    )
    args = parser.parse_args()
    if not args.publish:
        evidence = release_evidence(args.root)
        print(
            json.dumps(
                {
                    "status": "ready_for_merge_and_publication",
                    "selected_seed": evidence["selection"]["selected_seed"],
                }
            )
        )
        return
    release_evidence(args.root)
    # Kaggle owns the credential; never save it to a notebook, login cache or receipt.
    if args.token_stdin:
        token = sys.stdin.read().strip()
    else:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    print(json.dumps(publish(args.root, token), indent=2))


if __name__ == "__main__":
    main()
