# Existing free-GPU runner

**Current scope (November 23, 2026):** the owner selected a first pilot release:
200 intent-balanced teacher sources, all 8,992 gold training queries, and three
seeds per student. Public weights remain pending measured evaluation and
anonymous verification. The earlier session stopped without a verified external
checkpoint; its logged validation losses are historical, not release artifacts.

Use [the pilot release workflow](PILOT_RELEASE.md) and
[`notebooks/free_gpu_runner.ipynb`](../notebooks/free_gpu_runner.ipynb). It runs in
a committed foreground job and verifies private recovery uploads before training.
Environments and model caches stay outside `/kaggle/working`. No paid compute is
provisioned. Availability, quotas and session limits can still interrupt a run;
verified private backups support an explicit restore.

The 200-source pilot nominally makes 2,000 successful teacher requests: one
paraphrase request and nine verification requests per source, excluding retries.
All-label evaluation still scores all 77 complete label sequences. No projected
runtime is reported as a measurement.

## Environment

Python **3.11** is the CI/Docker target. The CPU code was also checked locally on
Python 3.12. Build a fresh environment; don't install the optional GPU stack over
an unrelated production project. Exact direct versions are in `pyproject.toml`:
Transformers 4.51.3, PEFT 0.15.2, TRL 0.17.0, datasets 3.6.0, torch 2.6.0 and vLLM
0.8.5.post1. The pinned vLLM uses `guided_choice`, not the latest API spelling.
The training API follows [TRL 0.17 documentation](https://huggingface.co/docs/trl/v0.17.0/sft_trainer).
The Student A GPU smoke validated CUDA and trainer behavior. AWQ teacher serving
and full evaluation require separate checks. Kaggle defaulted to Python 3.13, so
a dedicated Python 3.11 environment was necessary. The teacher extra includes
setuptools because Triton imports it at runtime in a fresh environment.

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
