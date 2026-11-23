# Evaluation protocol and limits

## Business objective

Route an English, in-domain customer query to one of 77 intents. Measure whether a
more expensive or adapted model improves routing enough to justify latency and
compute. Routing is not permission to execute a banking action. Reject uncertain
queries for human review. Banking77 has no comprehensive out-of-domain label;
low-confidence abstention is not a proven out-of-domain detector.

## Source, splits and leakage

The pinned official corpus has 10,003 train and 3,080 test queries. Exact duplicate
and overlapping training copies are quarantined using Unicode-normalized,
casefolded, whitespace-collapsed, identifier-masked text. Eleven training rows are
quarantined, then the remaining rows are split 90/10, stratified: 8,992 train and
1,000 validation. All official test memberships and labels remain intact. This
does not establish the absence of semantic near-duplicates in the public corpus.

The 300-query gate contains every class with three or four examples. It is selected
once with seed 42. It is a regression fixture, not a model-selection validation set.
Its reuse in CI means future full-test scores include 300 previously exposed
queries; report the 2,780-query gate-disjoint remainder separately in any subsequent
research-grade claim. The initial CPU configuration and threshold were fixed before
its full-test measurement. Do not tune prompts or student settings on either test
partition. Teacher prompts may retrieve only the training split.

## Statistics

Accuracy, macro-F1 and per-class F1 are computed on the predeclared split. The
95% intervals are percentile intervals from 300 label-stratified bootstrap
resamples, seed 2026. They measure query-sampling uncertainty for one fixed model;
they do not measure training-seed variation. The students separately require three
seeds and sample standard deviations (ddof=1). Worst-class and top-confusion lists
are descriptive post-evaluation error analysis, not a basis for retroactive tuning.

ECE uses ten equal-width confidence bins, including confidence 1 in the final bin.
TF-IDF probabilities are the original logistic probabilities; their noticeable
underconfidence is shown honestly rather than silently recalibrated after test
inspection. Neural temperature scaling and abstention thresholds use validation
only. The threshold maximizes empirical coverage subject to at least 30 accepted
validation cases and target accuracy 90%; it is not a confidence-bound guarantee.

Prediction logs mask numeric identifiers and email-like strings before writing.
Regex masking is conservative and incomplete, especially for unusual or
alphanumeric identifiers. No runtime request body is logged by the serving command.
The public gate is attributed source data, not uploaded personal customer data.

## Regression gate

Every PR rebuilds the CPU reference from the pinned training split, produces fresh
300-case predictions, checks exact IDs and hashes, and fails if macro-F1 drops
more than **0.01 absolute** below `evals/baseline.json`. A live 50-query check also
checks prediction-cache consistency. `scripts/freeze_gate.py` is a deliberate
baseline update and must be reviewed separately. The gate does not establish
Qwen quality; it is explicitly scoped to the fitted CPU baseline until GPU results
are available. Timing is not gated across heterogeneous runners.

## Security and load-test scope

Fifty authored pairs are five clean prompts crossed with ten instruction-override
wrappers. Report raw-label change and forced-label hit rates. They do not represent
universal attack success, adversarial coverage or a certified defense. A finite
vocabulary prevents made-up labels, not wrong allowed labels or prompt injection.

Locust measurements are short, loopback CPU smoke tests with 0.2–1 second think
time and a 25-user/second ramp. They are not steady-state maximum-throughput claims.
The shutdown CSV snapshot is the report source. CPU client and server share the
machine. CPU latency, GPU latency and end-to-end HTTP latency are distinct metrics.
