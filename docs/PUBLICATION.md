# Publishing a measured model

No model is released merely because the publisher exists. The GPU experiment,
evaluation reports, merged files and anonymous inference must actually pass.
The authorized destinations are `JatinMallarapu/qwen2.5-0.5b-banking77-distill`
and `JatinMallarapu/qwen2.5-0.5b-banking77-distill-lora`.

## Evidence and local merge

From the project root, using the pinned Python 3.11 GPU environment:

```sh
python -m b77.release
# Use the selected_seed reported above; N is a literal placeholder here.
python -m b77.audit_injections --adapter checkpoints/student_b/N/adapter --calibration-report evals/results/student_b-seedN-batch1.json
python -m b77.publish
python -m b77.release --merge
```

The first command is read-only and refuses incomplete runs. It requires all six
trained students, complete teacher-source coverage, matching data fingerprints,
and all eight neural configurations (two teacher modes plus six student seeds)
evaluated on 3,080 cases at batch sizes 1 and 32. Each student report must bind to
the exact adapter bytes. The selected Student B must also have a matching 50-pair
instruction-attack audit. The CPU reference remains explicitly labeled CPU.

The merge command selects Student B by validation loss, writes safetensors,
preserves the base license and data attribution, excludes optimizer pickles, and
records the selected adapter and merged-weight fingerprints. Do not merge while
other processes occupy most host RAM. The publisher regenerates both cards from
measured evidence; it does not upload the repository's untrained draft card.

## Kaggle token bridge

Create and attach `HF_TOKEN` through Kaggle Secrets. Never put its value in code,
an environment file, a command-line argument, notebook output, or chat. The default
Kaggle kernel and the pinned training virtual environment may use different Python
versions. Read the secret in the Kaggle kernel and pass it through a pipe:

```python
import subprocess
from kaggle_secrets import UserSecretsClient

subprocess.run(
    [gpu_python, "-m", "b77.publish", "--publish", "--token-stdin"],
    cwd=root,
    env=gpu_env,
    input=UserSecretsClient().get_secret("HF_TOKEN"),
    text=True,
    check=True,
)
```

`gpu_python`, `root` and `gpu_env` refer to the environment established in the GPU
runbook. The account identity must match `JatinMallarapu`. No login cache is written.
Only whitelisted model/tokenizer files, measured evidence, labels, cards and
attribution are uploaded. Symlinks, pickles and credential-like text block upload.
The upload receipt records public repo IDs, exact commit IDs and file hashes.

## Anonymous verification and release status

```sh
python -m b77.verify_public
```

This forces fresh downloads of both uploaded commits with `token=False`, checks
public visibility and runs a fixed query through each artifact. Both must produce
a valid intent. It uses the training system prompt and does not claim that a single
generation smoke establishes accuracy. The full quality reports remain essential.
Only after `release/public_verification.json` reports success should the GitHub
README, profile and release notes link the public models as released artifacts.
The resume release sentence must wait for that verification as well.
