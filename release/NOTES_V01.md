# v0.1 — Single-seed Banking77 distillation pilot

Public artifacts:
- [Merged Qwen student](https://huggingface.co/JatinMallarapu/qwen2.5-0.5b-banking77-distill)
- [LoRA adapter](https://huggingface.co/JatinMallarapu/qwen2.5-0.5b-banking77-distill-lora)

All four systems use the same 3,080 official test queries. The teacher generated
600 proposals from 200 balanced training sources; 480 survived the strict
three-vote filter. Both students retain all 8,992 gold training rows, with one seed each.

| System | Accuracy [95% CI] | Macro-F1 [95% CI] | p50 / p95 ms |
| --- | --- | --- | ---: |
| [TF-IDF + logistic regression (CPU)](https://github.com/jatinmallarapu/Banking77Distillation/blob/v0.1/evals/results/tfidf-test.json) | 88.67% [87.61, 89.81] | 88.71% [87.62, 89.85] | 3.21 / 5.54 |
| [Qwen2.5-7B AWQ · zero-shot](https://github.com/jatinmallarapu/Banking77Distillation/blob/v0.1/evals/results/zeroshot-batch1.json) | 61.79% [60.37, 62.94] | 59.68% [58.06, 60.80] | 3684.31 / 3703.69 |
| [Student A · gold only, seed 1](https://github.com/jatinmallarapu/Banking77Distillation/blob/v0.1/evals/results/student_a-seed1-batch1.json) | 93.21% [92.26, 94.09] | 93.21% [92.23, 94.08] | 548.23 / 607.81 |
| [Student B · gold + distilled, seed 1](https://github.com/jatinmallarapu/Banking77Distillation/blob/v0.1/evals/results/student_b-seed1-batch1.json) | 91.98% [90.97, 92.82] | 91.95% [90.94, 92.79] | 548.55 / 607.44 |

Student B's measured accuracy was 1.23 percentage points below
Student A. This pilot does not claim an accuracy improvement from distillation.
Intervals describe query-bootstrap uncertainty, not seed variation. GPU latency
is batch-1, all-77-label likelihood scoring, not HTTP API or greedy-generation
latency. CPU timings are separate hardware references; the GPU price assumption
was zero on free compute, not a production quote.

Both public revisions passed fresh-process anonymous downloads and inference.
The fixed sample query returned `card_arrival` for the merged model and adapter.
[Verification receipt](https://github.com/jatinmallarapu/Banking77Distillation/blob/v0.1/release/public_verification.json).

This is a prerelease research artifact. Additional seeds, full-source distillation,
few-shot comparison and batch-32 scaling remain future work. Code: MIT; weights:
Apache 2.0; Banking77 and derived paraphrases: CC BY 4.0 with attribution.
