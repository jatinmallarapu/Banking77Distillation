"""Train six explicit LoRA runs with completion-only loss and validation checkpoint selection."""

import json
from pathlib import Path

import yaml

from .data import digest, read_rows
from .guardrails import mask
from .prompts import student_messages
from .repro import checked_training_resume
from .teacher import validate_distilled


def train(config_path: Path, seed: int, smoke: bool = False, resume: bool = False) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU training is required; no CPU fallback or paid instance is started")
    config = yaml.safe_load(config_path.read_text())
    if seed not in config["seeds"]:
        raise ValueError("Use a predeclared seed")
    set_seed(seed)
    train_rows = read_rows(Path("data/prepared/train.jsonl"))
    validation = read_rows(Path("data/prepared/validation.jsonl"))
    if config["distilled"]:
        augmented = read_rows(Path("data/distilled.jsonl"))
        validate_distilled(
            augmented, train_rows, validation + read_rows(Path("data/prepared/test.jsonl"))
        )
        if not augmented:
            raise ValueError("Student B requires nonempty filtered teacher data")
        train_rows = train_rows + augmented
    labels = json.loads(Path("data/labels.json").read_text())
    out = Path("checkpoints") / config["name"] / str(seed)
    if smoke:
        out = Path("checkpoints/smoke") / config["name"] / str(seed)
    input_manifest = {
        "config": config,
        "seed": seed,
        "smoke": smoke,
        "train_sha256": digest(train_rows),
        "validation_sha256": digest(validation),
        "labels_sha256": digest(labels),
    }
    resume_checkpoint = checked_training_resume(out, input_manifest) if resume else None
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "input_manifest.json"
    if lock.exists() and json.loads(lock.read_text()) != input_manifest:
        raise ValueError("Training inputs changed; use a separate output directory")
    lock.write_text(json.dumps(input_manifest, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    truncated = {"examples": 0}

    def dataset(rows):
        prepared = []
        for row in rows:
            messages = student_messages(mask(row["text"]))
            completion = [{"role": "assistant", "content": labels[row["label"]]}]
            tokens = tokenizer.apply_chat_template(messages + completion, tokenize=True)
            if len(tokens) > config["max_length"]:
                truncated["examples"] += 1
                query_tokens = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
                while len(tokens) > config["max_length"] and query_tokens:
                    query_tokens = query_tokens[
                        : max(0, len(query_tokens) - (len(tokens) - config["max_length"]) - 1)
                    ]
                    messages[-1]["content"] = tokenizer.decode(
                        query_tokens, skip_special_tokens=True
                    )
                    tokens = tokenizer.apply_chat_template(messages + completion, tokenize=True)
                if not query_tokens:
                    raise ValueError(
                        "The instruction and label leave no room for a query; increase max_length"
                    )
            prepared.append({"prompt": messages, "completion": completion})
        return Dataset.from_list(prepared)

    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        max_length=config["max_length"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        completion_only_loss=True,
        packing=False,
        max_steps=2 if smoke else -1,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset(train_rows[:32] if smoke else train_rows),
        eval_dataset=dataset(validation[:16] if smoke else validation),
        peft_config=LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(out / "adapter")
    manifest = {
        "status": "smoke_only_not_releaseable" if smoke else "trained",
        "config": config,
        "seed": seed,
        "train_sha256": digest(train_rows),
        "validation_sha256": digest(validation),
        "train_rows": len(train_rows),
        "validation_rows": len(validation),
        "truncated_train_and_validation_examples": truncated["examples"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "gpu": torch.cuda.get_device_name(0),
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
