# Resume the trained pilot without retraining

`python -m b77.resume_evaluation --token-stdin --max-hours 10` uses an existing
free pair of T4 GPUs. It does not provision compute, train models, merge weights,
or publish public repositories. Restore the verified private checkpoint first;
keep the recovered single-seed configs and frozen data intact.

The runner validates the 200-source teacher identity, three-vote provenance,
training/validation hashes, completed manifests and both adapter fingerprints.
It verifies a private backup before work, then evaluates the teacher on GPU 0
and students on GPU 1. Every system still uses all 3,080 test queries. The explicit
v0.1 release plan declares batch 1; batch-32 scaling is future work. Validation alone selects temperature and abstention threshold.

Raw numeric scores, measured batch latency, token counts and a random measurement
session ID are saved under `evals/results/progress/`. No query text or credentials
are written to those checkpoints. Chunks are atomically replaced every sixteen
batches and on ordinary errors. A forced shutdown can lose the latest incomplete
chunk; completed chunks trigger the private backup worker, checked every 30 seconds.
Successful upload responses are followed by a download and SHA256 check.

Recovery rejects changed datasets, labels, adapter bytes, scoring/prompt source,
model configs, GPU type, Torch or CUDA versions. It does not convert partial caches
into complete reports. All expected rows must exist before metrics are computed.
Each report declares reused row counts and measurement sessions. Timings may span
resumed sessions on that GPU type; they are not advertised as a dedicated single
physical device or an uninterrupted production benchmark.

After all GPU reports and the injection audit pass the publication evidence gate,
the runner backs them up and stops. Restore that final snapshot in Colab for CPU
merging and publishing. Public publishing continues to read Colab Secrets only.
Do not discard the private backup or claim the models are released before anonymous
downloads and inference pass for both public repositories.
