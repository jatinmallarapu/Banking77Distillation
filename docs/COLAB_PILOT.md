# Free single-GPU v0.1 pilot

This is a deliberately reduced study, not a completed three-seed experiment.
It uses 200 balanced teacher sources, all 8,992 gold training queries, the frozen
1,000-query validation set, and all 3,080 official test queries. Student A and B
each use seed 1. Quality bootstrap intervals describe query uncertainty; seed
standard deviation is unavailable and is recorded as null, not zero.

1. Open `notebooks/colab_single_gpu_pilot.ipynb` in Colab. Select the free T4 runtime.
2. Add `HF_TOKEN` using Colab Secrets and enable notebook access. It must authorize
   the two public model destinations and the existing private checkpoint repository.
   The Kaggle secret does not automatically transfer to Colab. Never paste it into code.
3. Run setup, then the foreground pipeline cell. It performs decoder checks, a training
   smoke test, teacher generation, both students, full-test evaluations at batches 1/32,
   the 50-pair injection audit, evidence validation, merging, publishing and anonymous
   download/inference verification. A failure stops publication or verification.

No paid runtime is requested. Availability, runtime length and training duration are
not guaranteed. Do not bypass platform idle limits. Private backups are verified by
downloading the exact committed ZIP and checking SHA256 before training and after
each major stage; complete epoch checkpoints are also backed up during training.

The notebook contains a commented recovery cell. Restore into an empty directory,
reinstall the restored source, and resume without `--declare-single-seed-pilot`.
Training checks input fingerprints before using optimizer checkpoints. Incomplete
evaluation reports are recomputed. Keep the latest backup intact until recovery is
verified; do not launch a fresh job that overwrites it.

The release plan is explicit in `configs/release_plan.json` **inside the Colab run**.
The default repository still requires all three seeds and both teacher comparisons.
The first-launch cell refuses to reinterpret existing training manifests as a new scope.
This pilot includes TF-IDF, zero-shot 7B, gold-only student and distilled student.
Few-shot evaluation, seeds 2/3 and the full teacher-source study are future work.

Only after `release/public_verification.json` records successful anonymous downloads
and inference for both repositories should GitHub results and the resume say released.
The notebook does not have GitHub credentials and does not update GitHub automatically.
