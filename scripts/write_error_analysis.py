"""Export measured confusions with attributed, redacted query examples after evaluation."""

import json
from pathlib import Path

from b77.data import read_rows

report = json.loads(Path("evals/results/tfidf-test.json").read_text())
predictions = {row["id"]: row for row in read_rows(Path("evals/results/tfidf-test.jsonl"))}
lines = [
    "# Initial CPU error analysis",
    "",
    "Measured on the fixed 3,080-query official test set. Source text is PolyAI Banking77, CC BY 4.0; see ATTRIBUTION.md. Examples are masked. These are descriptive results, not a justification for retuning against this test set.",
    "",
    "## Five lowest-F1 intents",
    "",
    "| Intent | F1 | 95% bootstrap interval | Support |",
    "| --- | ---: | --- | ---: |",
]
for row in report["worst_classes"]:
    lines.append(
        f"| {row['label']} | {row['f1']:.3f} | {row['f1_ci95'][0]:.3f}–{row['f1_ci95'][1]:.3f} | {row['support']} |"
    )
lines.extend(["", "## Ten most frequent directional confusions", ""])
for pair in report["top_confused_pairs"]:
    lines.extend([f"### {pair['actual']} → {pair['predicted']} ({pair['count']})", ""])
    for ident in pair["example_ids"]:
        text = predictions[ident]["text"].replace("\n", " ")
        lines.append(f"- `{ident}`: {text}")
    lines.append("")
lines.extend(
    [
        "## Interpretation and next measurements",
        "",
        "Related transfer-status and card-failure intents remain difficult. Lexical overlap is a plausible contributor, but these examples do not establish a causal diagnosis. The baseline is underconfident (ECE 0.2668); its threshold is validation-selected and should not be mistaken for a calibrated probability guarantee.",
        "",
        "The next predeclared study is the unrun teacher/student comparison with training-only augmentation, three seeds and matched decoders. Do not claim that distillation resolves these errors until the actual reports exist.",
        "",
    ]
)
Path("docs/ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
