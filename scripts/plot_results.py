"""Render release figures from complete measured reports, never placeholder numbers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SYSTEMS = [
    ("TF-IDF (CPU)", "tfidf-test.json", "#68737d"),
    ("7B teacher (AWQ)", "zeroshot-batch1.json", "#a65b00"),
    ("Student A (gold)", "student_a-seed1-batch1.json", "#1967a3"),
    ("Student B (distilled)", "student_b-seed1-batch1.json", "#007d68"),
]


def render(results: Path, output: Path) -> None:
    reports = [
        (name, json.loads((results / file).read_text()), color) for name, file, color in SYSTEMS
    ]
    if any(report["cases"] != 3080 for _, report, _ in reports):
        raise ValueError("Figures require all 3080 test queries for every system")
    if len({report["case_sha256"] for _, report, _ in reports}) != 1:
        raise ValueError("Figures cannot compare different test sets")
    if any(report.get("status") != "measured" for _, report, _ in reports[1:]):
        raise ValueError("GPU measurements are incomplete")
    output.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12, 6.5))
    FigureCanvasAgg(figure)
    quality, latency = figure.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.4, 1]})
    figure.subplots_adjust(left=0.23, right=0.96, top=0.82, bottom=0.22, wspace=0.15)
    figure.suptitle("Banking77 · v0.1 single-seed pilot", fontsize=16, fontweight="bold")
    for index, (_, report, color) in enumerate(reports):
        value = report["accuracy"] * 100
        lower, upper = np.array(report["accuracy_ci95"]) * 100
        quality.errorbar(
            value,
            index,
            xerr=[[value - lower], [upper - value]],
            fmt="o",
            capsize=4,
            color=color,
            markersize=7,
        )
        quality.annotate(
            f"{value:.2f}%",
            (value, index),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
        p95 = report["latency_ms"]["p95"]
        latency.plot(p95, index, "o", color=color, markersize=7)
        latency.annotate(
            f"{p95:.1f} ms",
            (p95, index),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
    quality.set_yticks(range(len(reports)), [name for name, _, _ in reports])
    quality.invert_yaxis()
    quality.set_ylim(3.5, -0.6)
    limits = [value * 100 for _, report, _ in reports for value in report["accuracy_ci95"]]
    quality.set_xlim(max(0, min(limits) - 2), min(100, max(limits) + 2))
    quality.set_xlabel("Accuracy (%) · 95% query-bootstrap interval")
    quality.set_title("Quality on the official test set", pad=14)
    latency.set_xscale("log")
    times = [report["latency_ms"]["p95"] for _, report, _ in reports]
    latency.set_xlim(min(times) / 2, max(times) * 5)
    latency.set_xlabel("p95 inference latency (ms, log scale)")
    latency.set_title("Observed inference latency", pad=14)
    latency.tick_params(labelleft=False)
    for axis in [quality, latency]:
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.23,
        0.10,
        "All 3,080 queries. Fixed-model uncertainty; training-seed variance is unmeasured.",
        fontsize=10,
    )
    figure.text(
        0.23,
        0.06,
        "CPU reference and GPU all-77-label scoring are different timing scopes, not matched API speedups.",
        fontsize=9,
    )
    for extension in ["png", "svg"]:
        figure.savefig(output / f"quality_latency.{extension}", dpi=220)

    reliability = Figure(figsize=(8, 6))
    FigureCanvasAgg(reliability)
    axis = reliability.subplots()
    reliability.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.14)
    reliability.suptitle("Banking77 · test-set reliability", fontsize=16, fontweight="bold")
    axis.plot([0, 1], [0, 1], "--", color="#9da4aa", linewidth=1, label="Ideal calibration")
    for name, report, color in reports:
        bins = report["calibration"]["bins"]
        x = [row["confidence"] for row in bins]
        y = [row["accuracy"] for row in bins]
        ece = report["calibration"]["ece"]
        axis.plot(
            x, y, "o-", color=color, linewidth=1.6, markersize=4, label=f"{name} · ECE {ece:.3f}"
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy in each confidence bin",
    )
    axis.grid(alpha=0.16)
    axis.legend(loc="lower right", fontsize=9, framealpha=0.95)
    reliability.text(
        0.12,
        0.04,
        "Ten fixed confidence bins; empty bins omitted. Calibration fitted on validation only.",
        fontsize=9,
    )
    for extension in ["png", "svg"]:
        reliability.savefig(output / f"reliability.{extension}", dpi=220)
    sources = {
        file: hashlib.sha256((results / file).read_bytes()).hexdigest() for _, file, _ in SYSTEMS
    }
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("evals/results"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    render(args.results, args.output)


if __name__ == "__main__":
    main()
