# Existing free-GPU runner

**Authorization:** prepare only. No GPU was rented, no cloud instance was created,
and no Hugging Face upload is enabled. The local MX550 has 2 GB VRAM. The teacher,
six LoRA runs, keep rate, GPU latency/cost comparisons and Qwen release are unrun.

Use an existing Linux CUDA session with at least 14 GiB available VRAM, such as a
free T4 session where available. Quotas, availability and session limits vary;
this repository does not promise that the full experiment fits one session.
Persist `checkpoints/`, `data/distilled.jsonl`, and `runs/` between sessions.
Teacher generation involves three paraphrases and nine verification calls per
source query: approximately 108,000 requests for the current training split.
All-label likelihood evaluation also scores 77 candidates per query. Start with
the smoke run and a small explicit `b77 distill --limit 10` to estimate your own
runtime. No training-time or GPU-cost estimates are presented as measurements.

## Environment

Python **3.11** is the CI/Docker target. The CPU code was also checked locally on
Python 3.12. Build a fresh environment; don't install the optional GPU stack over
an unrelated production project. Exact direct versions are in `pyproject.toml`:
Transformers 4.51.3, PEFT 0.15.2, TRL 0.17.0, datasets 3.6.0, torch 2.6.0 and vLLM
0.8.5.post1. The pinned vLLM uses `guided_choice`, not the latest API spelling.
The training API follows [TRL 0.17 documentation](https://huggingface.co/docs/trl/v0.17.0/sft_trainer).
Direct dependency metadata was checked for compatibility; an actual GPU smoke
run has not yet validated CUDA, AWQ or trainer behavior in your notebook image.

```sh
python -m pip install -e '.[train,teacher,dev]'
python scripts/gpu_runner.py smoke
```

The smoke run creates a **non-releaseable** two-step checkpoint. It must not be
listed as a trained Banking77 model. The thin `notebooks/free_gpu_runner.ipynb`
invokes the same source code; all training logic lives in `src/b77/`.

## Full experiment order

1. `python scripts/gpu_runner.py teacher` launches a loopback vLLM server with the
   pinned AWQ revision, filters training-only paraphrases and shuts the server down.
2. Review `runs/distill-progress.json`, source IDs, agreement counts, keep rate,
   representative accepted/rejected examples and leakage checks before training.
   A consistent teacher can still be consistently wrong.
3. `python scripts/gpu_runner.py students` trains Student A and Student B for
   seeds 1, 2, 3, with completion-only loss, validation evaluation each epoch
   and the best validation-loss checkpoint. Long query tails are explicitly
   truncated to fit 128 tokens without cutting the complete label; counts are logged.
4. `python scripts/gpu_runner.py evaluate --gpu-hourly-price 0` measures zero-shot,
   retrieval few-shot and all six student runs at batch sizes 1 and 32. Zero is an
   explicit free-session accounting assumption, not a market GPU price. Substitute
   the actual hourly price for a separately authorized paid environment.
5. `python -m b77.release` validates the six manifests and matched 3,080-case
   reports. `python -m b77.release --merge` prepares local merged and adapter files.
   It never contacts the Hub to create or publish a repository.

## Comparison semantics

The few-shot retriever uses training-only TF-IDF vectors (sparse lexical retrieval,
not dense semantic embeddings). All neural systems use normalized log-likelihood
of the 77 complete labels plus EOS for evaluation, avoiding invented intents and
giving a full distribution for ECE. Sequence likelihood may favor shorter labels.
Validation-only temperature scaling and threshold selection are recorded. This
decoder is more expensive than one guided-choice generation; don't claim its
latency represents a single autoregressive answer. The generation/labeling API
separately uses guided choice during distillation.

Batch latency is the time for the entire classification batch; it is not divided
by batch size and mislabeled as request latency. Throughput and token totals are
reported separately. Teacher timings include loopback API/token scoring overhead;
student timings use local Transformers. Record GPU identity, memory, software
versions and these runtime differences before making a deployment recommendation.
No conclusion that distillation beats prompting is justified until these runs exist.
