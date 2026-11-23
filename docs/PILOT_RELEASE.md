# Pilot release and durable recovery

The first model declares `source_limit: 200` and `study_scope: pilot200`. Round-robin intent sampling gives two or three sources per Banking77 intent. Student A retains all 8,992 gold training rows; Student B uses the same rows plus filtered paraphrases. Both retain seeds 1, 2 and 3. The full all-source distillation study remains unfinished future work.

## Committed free-GPU job

Open `notebooks/free_gpu_runner.ipynb` with free T4 x2, Internet on and the existing `HF_TOKEN` secret attached. Use **Save Version > Save & Run All (Commit)**. The final foreground subprocess stays active until completion or failure. It never creates or purchases an instance.

The Python environment and public model cache live in `/kaggle/temp`, outside notebook output. The runner verifies private recovery before training. Publication requires all six trained adapters, full official-test comparisons, validation-only calibration, the selected adapter's attack audit and evidence checks. Anonymous downloads and inference then verify both public models.

Teacher generation uses pinned vLLM with unused CPU swap disabled. All-label teacher evaluation optionally uses the same pinned AWQ weights through AutoAWQ 0.2.9 and Triton, with shared query caches. AutoAWQ is archived; this is a frozen experiment dependency, not a production platform recommendation. Its [compatibility notice](https://github.com/casper-hansen/AutoAWQ) matches Torch 2.6.0 and Transformers 4.51.3. A real validation-query decoder probe runs before full comparisons. The [Triton fallback](https://github.com/casper-hansen/AutoAWQ/blob/v0.2.9/awq/modules/linear/gemm.py) avoids adding an incompatible compiled-kernel wheel. Reports identify the backend and internal microbatch limits.

## Private recovery

The owner authorized `JatinMallarapu/Banking77TrainingCheckpoints`, which must remain private. `b77.backup` uploads an archive and manifest in one commit, downloads the exact committed archive and verifies SHA256. A source/data backup must pass before training. Later backups follow completed optimizer checkpoints and at most fifteen-minute intervals. Three consecutive backup errors stop the runner.

Completed runs retain final adapters and manifests. Unfinished runs retain the latest complete optimizer, scheduler and RNG state. Credentials, environments, base-model caches, `.git`, `.env` and unnecessary serialized training arguments are excluded. The private archive is never advertised as a public distilled model.

Pass the existing secret through stdin from the Kaggle parent kernel:

```python
subprocess.run([gpu_python, '-m', 'b77.backup', 'upload', '--token-stdin'],
               cwd=root, env=gpu_env, text=True, check=True,
               input=UserSecretsClient().get_secret('HF_TOKEN'))
```

For recovery, replace the arguments with `b77.backup restore --token-stdin --destination /kaggle/working/b77-restored` through the same secret pipe. The destination must be empty. Restore pins one immutable commit, verifies SHA256 and rejects path traversal and symlinks before extraction. Training resume separately verifies the input fingerprint lock and complete optimizer state. Restoring a smoke checkpoint does not make it a trained release artifact.

## Interrupted attempt

The earlier session logged two completed gold-only seeds and a teacher audit. A Kaggle output upload returned HTTP 502; a replacement session contained no project files. The notebook had used detached background processes rather than an active foreground cell. The precise termination cause was not established. The retry removes that idle-session risk, disables unused CPU swap, and verifies external checkpoints. Historical logs remain available without claiming that the missing adapters were recovered or released.
