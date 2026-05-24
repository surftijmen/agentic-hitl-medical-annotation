"""
Strict-only trajectory figure with 95% CI error bars, supporting the
"No per-technique strict difference is resolvable at n=50" paragraph
in §5.2. Shows each condition's strict accuracy across iters 1-3 with
per-point error bars derived from the binomial standard error
sqrt(p(1-p)/n), and a shaded grey band marking the iter-1 ± 17pp
two-iteration noise band on the median iter-1 starting point.

If error bars overlap across iters, the visual conclusion is the same
as the prose: within-trajectory swings live inside the noise floor.

Usage:
    PYTHONPATH=src python3 scripts/make_sq2_strict_noisefloor_figure.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N = 50  # samples per iter
CONDITIONS = [
    ("baseline_hitl",    "logs/experiments/baseline_hitl/baseline_hitl_v20260523_*_lenient.json"),
    ("cot_hitl",         "logs/experiments/cot_hitl/cot_hitl_v20260523_*_lenient.json"),
    ("few_shot_hitl",    "logs/experiments/few_shot_hitl/few_shot_hitl_v20260523_*_lenient.json"),
    ("consistency_hitl", "logs/experiments/consistency_hitl/consistency_hitl_v20260523_*_lenient.json"),
    ("rag_hitl",         "logs/experiments/rag_hitl/rag_hitl_v20260523_13[23]*_lenient.json"),
]

COLORS = {
    "baseline_hitl":    "#222222",
    "cot_hitl":         "#1f77b4",
    "few_shot_hitl":    "#2ca02c",
    "consistency_hitl": "#ff7f0e",
    "rag_hitl":         "#d62728",
}

OUT = "docs/figures/fig_sq2_strict_noisefloor.png"


def load_strict(pattern: str):
    paths = sorted(p for p in glob.glob(pattern) if "_lenient" in p)
    if not paths:
        sys.exit(f"ERROR: no files matched {pattern}")
    return [json.load(open(p))["summary"]["strict_accuracy"] for p in paths]


def ci_95(p: float, n: int) -> float:
    """95% CI half-width on a binomial proportion at sample n."""
    return 1.96 * math.sqrt(p * (1 - p) / n)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = [1, 2, 3]

    # Shaded ±17pp two-iteration noise band, centred on the median iter-1 strict.
    iter1_strict = []
    for cond, pat in CONDITIONS:
        iter1_strict.append(load_strict(pat)[0])
    band_centre = sorted(iter1_strict)[len(iter1_strict) // 2]
    halfwidth = 0.17  # ±17pp two-iteration CI at n=50, p≈0.75
    ax.axhspan(band_centre - halfwidth, band_centre + halfwidth,
               color="grey", alpha=0.12,
               label=f"$\\pm 17$pp two-iter noise band (centred on iter-1 median = {band_centre:.2f})")

    for cond, pat in CONDITIONS:
        strict = load_strict(pat)
        errs = [ci_95(p, N) for p in strict]
        c = COLORS[cond]
        ax.errorbar(xs, strict, yerr=errs, marker="o", color=c, label=cond,
                    linewidth=1.6, capsize=3, alpha=0.95)

    ax.set_xlabel("HITL iteration")
    ax.set_ylabel("Strict accuracy (vs MIMIC SEQ\\_NUM=1)")
    ax.set_xticks(xs)
    ax.set_ylim(0.45, 1.00)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_title(f"Per-technique strict trajectories with 95\\% CI bars (n={N} per iter)",
                 fontsize=10)

    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
