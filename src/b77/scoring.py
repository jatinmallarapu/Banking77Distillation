"""Score complete allowed label sequences with shared query caches and validation calibration."""

import json
from pathlib import Path

import numpy as np
import yaml

from .prompts import student_messages, teacher_messages
from .teacher import Retriever, Teacher


def softmax(scores: np.ndarray, temperature: float = 1) -> np.ndarray:
    scaled = scores / temperature
    values = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, truth: np.ndarray) -> float:
    choices = [0.5, 0.75, 1, 1.5, 2, 3, 4]
    return min(
        choices,
        key=lambda t: float(
            -np.log(softmax(scores, t)[np.arange(len(truth)), truth].clip(1e-12)).mean()
        ),
    )


class RemoteScorer:
    def __init__(self, training: list[dict], fewshot: bool = False) -> None:
        from transformers import AutoTokenizer

        self.api = Teacher()
        self.labels = self.api.labels
        self.fewshot = fewshot
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.api.config["model"], revision=self.api.config["revision"]
        )
        self.retriever = Retriever(training)

    def score(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        candidates = []
        prefix_sizes = []
        for text in texts:
            examples = self.retriever.neighbors(text) if self.fewshot else None
            prefix = self.tokenizer.apply_chat_template(
                teacher_messages(text, self.labels, examples),
                tokenize=True,
                add_generation_prompt=True,
            )
            for label in self.labels:
                ids = (
                    prefix
                    + self.tokenizer.encode(label, add_special_tokens=False)
                    + [self.tokenizer.eos_token_id]
                )
                candidates.append(ids)
                prefix_sizes.append(len(prefix))
        scores = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        for offset in range(0, len(candidates), 16):
            batch = candidates[offset : offset + 16]
            response = self.api.client.post(
                "/completions",
                json={
                    "model": self.api.config["model"],
                    "prompt": batch,
                    "echo": True,
                    "max_tokens": 1,
                    "logprobs": 1,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            data = response.json()
            choices = sorted(data["choices"], key=lambda row: row["index"])
            if len(choices) != len(batch):
                raise ValueError("Incomplete prompt-logprob response")
            for index, choice in enumerate(choices):
                logprobs = choice["logprobs"]["token_logprobs"]
                prefix_size = prefix_sizes[offset + index]
                selected = logprobs[prefix_size : len(batch[index])]
                if len(selected) != len(batch[index]) - prefix_size or any(
                    value is None for value in selected
                ):
                    raise ValueError("Pinned vLLM echo logprob contract was not satisfied")
                scores.append(sum(selected))
            for key in usage:
                field = "prompt_tokens" if key == "input_tokens" else "completion_tokens"
                if field not in data.get("usage", {}):
                    raise ValueError("Missing measured token usage")
                usage[key] += data["usage"][field]
        return np.array(scores).reshape(len(texts), len(self.labels)), usage


class StudentScorer:
    label_batch_size = 16
    query_microbatch = 32
    cache_budget_tokens = 65536

    def messages(self, text: str) -> list[dict]:
        return student_messages(text)

    def __init__(self, adapter: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Student comparison requires a GPU")
        config = yaml.safe_load(Path("configs/student_a.yaml").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], revision=config["revision"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base = AutoModelForCausalLM.from_pretrained(
            config["base_model"], revision=config["revision"], torch_dtype=dtype
        )
        self.model = PeftModel.from_pretrained(base, adapter).to("cuda").eval()
        self.labels = json.loads(Path("data/labels.json").read_text())

    def score_reference(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        import torch

        sequences = []
        starts = []
        for text in texts:
            prefix = self.tokenizer.apply_chat_template(
                self.messages(text), tokenize=True, add_generation_prompt=True
            )
            for label in self.labels:
                sequences.append(
                    prefix
                    + self.tokenizer.encode(label, add_special_tokens=False)
                    + [self.tokenizer.eos_token_id]
                )
                starts.append(len(prefix))
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(sequences), 4):
                batch = sequences[offset : offset + 4]
                length = max(map(len, batch))
                ids = torch.tensor(
                    [row + [self.tokenizer.pad_token_id] * (length - len(row)) for row in batch],
                    device="cuda",
                )
                attention = torch.tensor(
                    [[1] * len(row) + [0] * (length - len(row)) for row in batch], device="cuda"
                )
                logits = self.model(input_ids=ids, attention_mask=attention).logits
                for index, row in enumerate(batch):
                    begin = starts[offset + index]
                    distribution = torch.log_softmax(
                        logits[index, begin - 1 : len(row) - 1].float(), dim=-1
                    )
                    truth = ids[index, begin : len(row)]
                    scores.append(distribution.gather(1, truth[:, None]).sum().item())
                del logits
        torch.cuda.synchronize()
        return np.array(scores).reshape(len(texts), len(self.labels)), {
            "input_tokens": sum(starts),
            "output_tokens": sum(len(row) - start for row, start in zip(sequences, starts)),
        }

    def score(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        """Batch query prefixes and reuse their KV states for complete-label likelihoods."""
        import copy

        import torch

        if len(texts) > self.query_microbatch:
            pieces = [
                self.score(texts[i : i + self.query_microbatch])
                for i in range(0, len(texts), self.query_microbatch)
            ]
            return np.concatenate([p[0] for p in pieces]), {
                key: sum(p[1][key] for p in pieces) for key in ["input_tokens", "output_tokens"]
            }
        prefixes = [
            self.tokenizer.apply_chat_template(
                self.messages(text), tokenize=True, add_generation_prompt=True
            )
            for text in texts
        ]
        labels = [
            self.tokenizer.encode(label, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            for label in self.labels
        ]
        width = max(map(len, prefixes))
        ids = torch.tensor(
            [[self.tokenizer.pad_token_id] * (width - len(p)) + p for p in prefixes], device="cuda"
        )
        attention = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in prefixes], device="cuda"
        )
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        scores = torch.empty((len(texts), len(labels)), device="cuda")
        label_group = min(
            self.label_batch_size, max(1, self.cache_budget_tokens // (len(texts) * width))
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=positions,
                use_cache=True,
                logits_to_keep=1,
            )
            first = torch.log_softmax(output.logits[:, -1].float(), dim=-1)
            for offset in range(0, len(labels), label_group):
                batch = labels[offset : offset + label_group]
                size = max(len(row) - 1 for row in batch)
                # Query-major order matches repeat_interleave of the query KV states.
                expanded_rows = [row for _ in texts for row in batch]
                suffix = torch.tensor(
                    [
                        row[:-1] + [self.tokenizer.pad_token_id] * (size - len(row) + 1)
                        for row in expanded_rows
                    ],
                    device="cuda",
                )
                suffix_attention = torch.tensor(
                    [[1] * (len(row) - 1) + [0] * (size - len(row) + 1) for row in expanded_rows],
                    device="cuda",
                )
                full_attention = torch.cat(
                    [attention.repeat_interleave(len(batch), dim=0), suffix_attention], dim=1
                )
                # Left-padded prefixes need explicit logical positions for each query's suffix.
                suffix_positions = torch.tensor([len(p) for p in prefixes], device="cuda")
                suffix_positions = suffix_positions.repeat_interleave(len(batch))[:, None]
                suffix_positions = suffix_positions + torch.arange(size, device="cuda")[None, :]
                cache = copy.deepcopy(output.past_key_values)
                cache.batch_repeat_interleave(len(batch))
                logits = self.model(
                    input_ids=suffix,
                    attention_mask=full_attention,
                    position_ids=suffix_positions,
                    past_key_values=cache,
                    use_cache=True,
                ).logits
                targets = torch.tensor(
                    [
                        row[1:] + [self.tokenizer.pad_token_id] * (size - len(row) + 1)
                        for row in expanded_rows
                    ],
                    device="cuda",
                )
                # Reduce all candidates together rather than synchronizing once per label.
                selected = logits.gather(2, targets[:, :, None]).squeeze(-1).float()
                log_normalizer = torch.logsumexp(logits.float(), dim=-1)
                suffix_scores = ((selected - log_normalizer) * suffix_attention).sum(-1)
                query_indices = torch.arange(len(texts), device="cuda").repeat_interleave(
                    len(batch)
                )
                first_tokens = torch.tensor([row[0] for row in expanded_rows], device="cuda")
                values = first[query_indices, first_tokens] + suffix_scores
                scores[:, offset : offset + len(batch)] = values.reshape(len(texts), len(batch))
                del cache, logits
        torch.cuda.synchronize()
        return scores.cpu().numpy(), {
            "input_tokens": sum(map(len, prefixes)),
            "output_tokens": len(texts) * sum(map(len, labels)),
        }


class LocalAWQTeacherScorer(StudentScorer):
    """Use the same pinned AWQ weights with query KV reuse instead of repeated HTTP prefill."""

    label_batch_size = 77
    query_microbatch = 1
    cache_budget_tokens = 100000

    def __init__(self, training: list[dict], fewshot: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("The AWQ teacher benchmark requires an already allocated GPU")
        self.config = yaml.safe_load(Path("configs/teacher.yaml").read_text())
        self.labels = json.loads(Path("data/labels.json").read_text())
        self.fewshot = fewshot
        self.retriever = Retriever(training)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"], revision=self.config["revision"], trust_remote_code=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"],
            revision=self.config["revision"],
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
            trust_remote_code=False,
            attn_implementation="sdpa",
        ).eval()

    def messages(self, text: str) -> list[dict]:
        examples = self.retriever.neighbors(text) if self.fewshot else None
        return teacher_messages(text, self.labels, examples)
