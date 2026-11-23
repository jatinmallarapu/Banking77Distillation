# Teacher pilot investigation

The original ten-source GPU pilot completed in 112.48 seconds and retained zero
of 30 proposed paraphrases. The longer run was stopped after 66 completed sources
with zero accepted examples. These are pilot observations, not a full-data result.

Five raw diagnostic labels were all members of the guided-choice label set. A
format-normalization change would not fix these semantic disagreements:

| Source | Gold intent | Raw zero-shot label |
| --- | --- | --- |
| train-00000 | card_arrival | card_delivery_estimate |
| train-02002 | pending_top_up | pending_top_up |
| train-04007 | get_physical_card | change_pin |
| train-06008 | pending_transfer | pending_top_up |
| train-08010 | top_up_by_cash_or_cheque | top_up_by_cash_or_cheque |

For the arrival query, generated wording introduced issuance and processing facts
absent from the original question. Other examples reveal ambiguous source labels;
the pipeline does not silently relabel Banking77 gold data.

The original prefix is class-grouped, so the early zero keep rate cannot estimate
retention across all 77 intents. Diagnostic retrieval included the source itself;
its predictions are **not** independent accuracy evidence.

## Revised protocol: intent_preserving_retrieval_v2

- Supply the source training intent to the paraphrase generator, explicitly forbid
  adding facts, and forbid placing label text inside generated queries.
- Verify each candidate with ten retrieved **training-only** examples, excluding
  its own source ID. No validation/test examples enter the retrieval index.
- Retain the original three independent seeded label samples at temperature 0.7;
  all must agree with the source gold label. The filter is not weakened to hit a
  desired acceptance percentage.
- Traverse sources in label round-robin order. A 200-source pilot covers every
  label with two or three sources before the full job resumes.
- Change the protocol field in the config and archive v1 output separately. Input
  identity checks prevent silently mixing the two protocols.

Run `b77 distill --limit 200 --workers 8` in the GPU environment. Record actual
retention, latency and per-label coverage. Human inspection is still necessary:
teacher agreement does not prove that a paraphrase preserved meaning. No target
keep-rate range or improvement over the gold-only student is guaranteed.

The revised GPU pilot is pending. Full student comparisons and publication remain
pending; these diagnostics do not justify a released-model claim.
