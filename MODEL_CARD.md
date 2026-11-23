---
language: en
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - PolyAI/banking77
pipeline_tag: text-generation
tags:
  - intent-classification
  - banking77
  - distillation
---

# Banking77 Distill — release preparation card

**Status: NOT TRAINED / NOT PUBLISHED.** This is a repository-side preparation
card, not evidence of a Hugging Face model release. The owner requested preparation
without GPU spending or Hub publication. Do not upload this draft as a completed
model or claim that the comparisons below have been measured.

## Intended use

Research on English customer-intent routing among the 77 Banking77 intents, with
human review for uncertain queries. Not intended for real banking decisions,
identity verification, fraud determination, payment execution or legal eligibility.
Domain shift, missing intents and multilingual inputs require independent review.

## Planned training

Pinned Qwen2.5-0.5B-Instruct base, LoRA rank 16 / alpha 32 / dropout 0.05, all linear
layers, learning rate 2e-4, three epochs, maximum training length 128. Long query
tails are truncated while retaining the instruction and complete label; counts
must be recorded. Completion-only loss; checkpoint selected by validation loss.
Seeds 1, 2 and 3 for gold-only Student A and augmented Student B.

Teacher: pinned Qwen2.5-7B-Instruct-AWQ. Three paraphrases per training query,
accepted only when three temperature-0.7 labels agree with each other and the
source gold label. Source IDs, teacher revision, votes and exact normalized leakage
checks are required. Keep rate: **not measured**.

## Measured results

The GitHub CPU baseline report is a TF-IDF/logistic model, not this Qwen model.
Qwen quality, bootstrap intervals, three-seed mean ± standard deviation, calibration,
GPU latency, cost and injection robustness: **not measured**. Fill them only from
completed, hash-matched GPU reports. A smoke training run is not releaseable.

## Inference and uncertainty

The planned evaluator scores all 77 complete labels by normalized conditional
log-likelihood, including EOS. Temperature and abstention threshold are chosen on
validation data. This confidence is not a guarantee. Label scoring is substantially
more expensive than a single greedy generation; compare identical decoders and
record batch-1 and batch-32 timings on the same GPU.

## Known limitations and safeguards

Banking77 contains short, English, in-domain queries; it is not a complete banking
taxonomy. Distillation may reinforce teacher errors. Agreement filtering does not
prove label correctness. Regex identifier masking can miss sensitive formats.
Prompt boundaries and a finite label set do not prove injection resistance. No
actual Qwen guardrail results are available in this preparation release.

## Licensing and provenance

Base model: Apache 2.0, Qwen team / Alibaba Cloud. Training data: PolyAI Banking77,
CC BY 4.0; see `ATTRIBUTION.md` for the paper, pinned source and modifications.
Future adapted-data publication must retain attribution. Planned repository name:
`jatinmallarapu/qwen2.5-0.5b-banking77-distill`; no public repository or weights are
claimed to exist. The separate adapter and filtered data remain local until the
owner explicitly authorizes publication after reviewing the completed results.
