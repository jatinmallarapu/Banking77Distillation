"""Render publication-ready reliability and coverage plots from measured CPU outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

report = json.loads(Path("evals/results/tfidf-test.json").read_text())
out = Path("docs")
out.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
bins = report["calibration"]["bins"]
axes[0].plot([0, 1], [0, 1], "--", color="#83909f", label="Ideal calibration")
axes[0].plot(
    [r["confidence"] for r in bins],
    [r["accuracy"] for r in bins],
    "o-",
    color="#126f83",
    label="TF-IDF baseline",
)
axes[0].set(
    xlabel="Mean predicted confidence",
    ylabel="Observed accuracy",
    xlim=(0, 1),
    ylim=(0, 1),
    title=f"Reliability · ECE {report['calibration']['ece']:.3f}",
)
axes[0].legend(frameon=False)
curve = [r for r in report["abstention"]["curve"] if r["accuracy"] is not None]
axes[1].plot([r["coverage"] for r in curve], [r["accuracy"] for r in curve], "o-", color="#126f83")
axes[1].set(
    xlabel="Fraction automatically routed",
    ylabel="Accepted-query accuracy",
    xlim=(0, 1),
    ylim=(0, 1),
    title="Coverage–accuracy trade-off",
)
fig.savefig(out / "calibration.png", dpi=170)
plt.close(fig)
