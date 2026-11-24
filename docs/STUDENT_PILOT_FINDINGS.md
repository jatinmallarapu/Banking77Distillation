# What the single-seed pilot actually showed

These findings use all 3,080 official test queries. The teacher comparison and
public release were still pending when this analysis was written. The machine-readable
sources are [Student A](../evals/results/student_a-seed1-batch1.json) and
[Student B](../evals/results/student_b-seed1-batch1.json).

| Observation | Gold-only Student A | Augmented Student B |
| --- | ---: | ---: |
| Training rows | 8,992 | 9,472 |
| Best validation loss | 0.0559085980 | 0.0533467755 |
| Test accuracy | 93.2143% | 91.9805% |
| Test macro-F1 | 0.932073 | 0.919533 |
| Validation-selected temperature | 1.5 | 1.0 |
| Test ECE after that calibration | 0.014833 | 0.021686 |

The 480 accepted paraphrases came from 200 intent-balanced training sources. Three
teacher votes agreeing with the gold label filtered out inconsistent labels; it
did not establish that every retained rewrite preserved the original meaning.

## Lower validation loss did not imply better routing

Student B's best validation loss was lower, but its test accuracy was 1.2338
percentage points lower. Training token loss and the evaluated all-label routing
decision are different objectives. This result does not identify a single cause:
augmentation noise, the limited source subset and training variation remain
plausible explanations requiring further experiments. One seed cannot estimate
training-seed variance. Query-bootstrap intervals must not be presented as that
variance.

## Failures concentrate around close intent boundaries

Student B predicted `top_up_reverted` for 14 of the 40 `top_up_failed` test cases;
the latter class had F1 0.7059. It also predicted `exchange_via_app` for eight
`fiat_currency_support` cases and `card_delivery_estimate` for five `card_arrival`
cases. The report includes stable example IDs for inspection. These observations
justify checking whether teacher rewrites blur those boundaries, but do not prove
that the rewrites caused the errors.

## A configured abstention mechanism is not a demonstrated safeguard

Both students selected threshold 0 on validation for the configured 90% empirical
accuracy target, so this operating point abstained on no test cases. The stored
coverage curves show alternatives; choosing a new threshold after inspecting test
results would require a fresh validation protocol. Neither confidence nor the
small authored injection audit certifies safe deployment on real banking traffic.

## An integration failure worth preserving

The attack audit initially stopped because its forced-target metadata used
`refund_not_showing_up`, whereas the official taxonomy contains
`Refund_not_showing_up`. The fix corrected the metadata and retained the lowercase
attack text. The audit continues to reject unknown target labels. This was an
evaluation integration bug, not evidence of robustness or a model-quality failure.

## Practical decision

For this measured pilot, Student A is the stronger student on test accuracy.
Student B remains useful as a reproducible distillation experiment, rather than
an accuracy improvement claim. TF-IDF remains a credible CPU alternative when
its lower measured accuracy is acceptable. CPU TF-IDF timings and GPU all-label
likelihood timings are different execution scopes; their ratio is not a matched
production serving speedup. Teacher results are needed before making the final
teacher-versus-student quality and cost comparison.
