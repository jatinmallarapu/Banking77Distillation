"""Force anonymous downloads and inference before recording a usable public release."""

import argparse
import gc
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .prompts import STUDENT_SYSTEM
from .publish import NAMESPACE, REPOSITORIES


def verify(root: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    receipt = json.loads((root / "release/upload_receipt.json").read_text())
    expected = {f"{NAMESPACE}/{name}" for name in REPOSITORIES.values()}
    if set(receipt["models"]) != expected:
        raise ValueError("Verification is restricted to the two authorized model repositories")
    labels = json.loads((root / "data/labels.json").read_text())
    client = HfApi(token=False)
    verified = {}
    for kind, name in REPOSITORIES.items():
        repo_id = f"{NAMESPACE}/{name}"
        revision = receipt["models"][repo_id]["commit"]
        info = client.model_info(repo_id, revision=revision)
        if info.private:
            raise ValueError("The model is not publicly visible")
        # Bypass login tokens and force the network transfer, even with a warm cache.
        folder = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=False,
            force_download=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False
        )
        base = None
        if kind == "merged":
            model = AutoModelForCausalLM.from_pretrained(
                folder,
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
            )
        else:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                revision="7ae557604adf67be50417f59c2c2f167def9a775",
                token=False,
                use_safetensors=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
            )
            model = PeftModel.from_pretrained(base, folder, local_files_only=True)
        model.eval()
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": STUDENT_SYSTEM},
                {"role": "user", "content": "My new card still hasn't arrived"},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            generated = model.generate(inputs, max_new_tokens=16, do_sample=False)
        label = tokenizer.decode(generated[0, inputs.shape[1] :], skip_special_tokens=True).strip()
        if label not in labels:
            raise ValueError("Anonymous inference did not produce a valid Banking77 intent")
        verified[repo_id] = {
            "revision": revision,
            "anonymous_download": True,
            "force_download": True,
            "smoke_label": label,
        }
        del model, base, inputs, generated
        gc.collect()
    result = {
        "status": "verified_public_download_and_inference",
        "models": verified,
        "scope": "one generation smoke per artifact; full quality is in evaluation reports",
    }
    (root / "release/public_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
