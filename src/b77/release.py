"""Merge completed runs locally; publication is handled by the separate evidence gate."""

import argparse
import json
from pathlib import Path

import numpy as np

from .repro import adapter_fingerprint, file_fingerprint
from .release_plan import release_plan


def readiness(root: Path = Path(".")) -> dict:
    plan = release_plan(root)
    runs = []
    for student in ["a", "b"]:
        for seed in plan["seeds"]:
            path = root / f"checkpoints/student_{student}/{seed}/training_manifest.json"
            if not path.exists():
                raise ValueError(f"Missing completed training manifest: {path}")
            manifest = json.loads(path.read_text())
            if (
                manifest["status"] != "trained"
                or manifest["seed"] != seed
                or not manifest["best_checkpoint"]
            ):
                raise ValueError("Smoke or incomplete checkpoints cannot be released")
            report_path = root / f"evals/results/student_{student}-seed{seed}-batch1.json"
            report = json.loads(report_path.read_text())
            if report.get("status") != "measured" or report["cases"] != 3080:
                raise ValueError(
                    "All seeds require complete test reports before release preparation"
                )
            runs.append({"student": student, "seed": seed, "manifest": manifest, "report": report})
    if len({row["report"]["case_sha256"] for row in runs}) != 1:
        raise ValueError("Student evaluations use different test cases")
    summary = {
        student: {
            "accuracy_mean": float(
                np.mean([r["report"]["accuracy"] for r in runs if r["student"] == student])
            ),
            "accuracy_std": None
            if len(plan["seeds"]) == 1
            else float(
                np.std([r["report"]["accuracy"] for r in runs if r["student"] == student], ddof=1)
            ),
            "macro_f1_mean": float(
                np.mean([r["report"]["macro_f1"] for r in runs if r["student"] == student])
            ),
            "macro_f1_std": None
            if len(plan["seeds"]) == 1
            else float(
                np.std([r["report"]["macro_f1"] for r in runs if r["student"] == student], ddof=1)
            ),
        }
        for student in ["a", "b"]
    }
    # Choose the released seed by validation loss, never by test accuracy.
    selected = min(
        [row for row in runs if row["student"] == "b"],
        key=lambda r: r["manifest"]["best_validation_loss"],
    )
    return {
        "status": "ready_for_local_merge",
        "release_plan": plan,
        "summary": summary,
        "selected_seed": selected["seed"],
        "selected_test_report": selected["report"],
        "published": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    report = readiness()
    out = Path("release")
    out.mkdir(exist_ok=True)
    if args.merge:
        import shutil

        import torch
        import yaml
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = yaml.safe_load(Path("configs/student_b.yaml").read_text())
        adapter = Path(f"checkpoints/student_b/{report['selected_seed']}/adapter")
        base = AutoModelForCausalLM.from_pretrained(
            config["base_model"], revision=config["revision"], torch_dtype=torch.float32
        )
        merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        AutoTokenizer.from_pretrained(adapter).save_pretrained(out / "merged")
        card = (
            "---\nlanguage: en\nlicense: apache-2.0\nbase_model: Qwen/Qwen2.5-0.5B-Instruct\n"
            "datasets:\n  - PolyAI/banking77\npipeline_tag: text-generation\n---\n\n"
            "# Banking77 distilled student — local release candidate\n\n"
            "Local merge candidate; NOT PUBLISHED. b77.publish validates complete evidence before upload.\n\n"
            "English intent routing among 77 Banking77 labels. Not suitable for banking decisions, "
            "payment execution, fraud determination, or unreviewed real-customer deployment. "
            "Regex masking and a finite label set do not prove privacy or injection resistance.\n\n"
            "Student B uses pinned Qwen2.5-0.5B-Instruct with LoRA r16/alpha32/dropout0.05, "
            "gold data plus three-vote-filtered teacher paraphrases, and validation-loss checkpoint selection. "
            "The selected seed is chosen by validation loss, not test performance. "
            "Bootstrap intervals concern queries; three-seed standard deviations concern training variation.\n\n"
            "Data credit: PolyAI Banking77, CC BY 4.0, Casanueva et al. (2020), Efficient Intent "
            "Detection with Dual Sentence Encoders. https://huggingface.co/datasets/PolyAI/banking77 "
            "Base weights: Qwen team, Apache 2.0. See ATTRIBUTION.md and LICENSE.\n\n"
            "## Measured, matched release evidence\n\n```json\n"
            + json.dumps(report, indent=2)
            + "\n```\n"
        )
        (out / "merged/README.md").write_text(card)
        license_path = hf_hub_download(config["base_model"], "LICENSE", revision=config["revision"])
        shutil.copy2(license_path, out / "merged/LICENSE")
        shutil.copy2("ATTRIBUTION.md", out / "merged/ATTRIBUTION.md")
        shutil.copytree(
            adapter,
            out / "adapter",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("training_args.bin", "optimizer.pt", "scheduler.pt"),
        )
        (out / "adapter/README.md").write_text(card)
        shutil.copy2(license_path, out / "adapter/LICENSE")
        shutil.copy2("ATTRIBUTION.md", out / "adapter/ATTRIBUTION.md")
        dataset = out / "filtered-dataset"
        dataset.mkdir(exist_ok=True)
        shutil.copy2("data/distilled.jsonl", dataset / "train.jsonl")
        (dataset / "README.md").write_text(
            "---\nlicense: cc-by-4.0\nlanguage: en\n---\n# Filtered training paraphrases\n\nAdapted from PolyAI Banking77 training queries with Qwen teacher paraphrases and three-vote agreement filtering. Not human-verified. Preserve ATTRIBUTION.md. Validation and test sources are excluded.\n"
        )
        shutil.copy2("ATTRIBUTION.md", dataset / "ATTRIBUTION.md")
        report["selected_adapter_fingerprint"] = adapter_fingerprint(adapter)
        report["base_model"] = config["base_model"]
        report["base_revision"] = config["revision"]
        report["merged_weights_sha256"] = {
            path.name: file_fingerprint(path) for path in (out / "merged").glob("*.safetensors")
        }
    (out / "release_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
