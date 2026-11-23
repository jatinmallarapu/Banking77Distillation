"""Check cache equivalence on a tiny random transformer; this is not a trained-model benchmark."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from b77.scoring import StudentScorer


class Tokens:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(char) % 60 for char in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [2, 3] + self.encode(messages[-1]["content"]) + [4]


def main():
    torch.manual_seed(7)
    scorer = StudentScorer.__new__(StudentScorer)
    scorer.tokenizer = Tokens()
    scorer.labels = ["a", "long_label", "xy", "other"]
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    scorer.model = Qwen2ForCausalLM(config).to("cuda").eval()
    texts = ["short", "a longer query", "x"]
    reference, _ = scorer.score_reference(texts)
    batched, _ = scorer.score(texts)
    singles = np.concatenate([scorer.score([text])[0] for text in texts])
    np.testing.assert_allclose(batched, reference, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(singles, reference, atol=2e-5, rtol=1e-6)
    result = {
        "status": "passed",
        "scope": "tiny_random_fp32_transformer_math_only",
        "cases": len(texts),
        "labels": len(scorer.labels),
        "maximum_absolute_difference": float(np.max(np.abs(batched - reference))),
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/decoder-math-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
