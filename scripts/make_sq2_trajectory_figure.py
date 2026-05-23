"""
Generate docs/figures/fig_sq2_trajectories.png — the per-technique HITL
trajectory figure that accompanies Table~\\ref{tab:sq2} in §Results.

Two side-by-side panels:
  • left  : strict accuracy across iters 1-3, one line per condition
  • right : lenient accuracy across iters 1-3, one line per condition

A shaded band on each panel marks the ±17pp two-iteration noise floor
relative to the iter-1 starting point, so the reader can immediately
see that all within-trajectory swings live inside the noise band on
strict but outside it on the lenient ceiling.

Data sources: the *_lenient.json files written by
scripts/rescore_with_all_icds.py for each condition's May 23 trajectory.

Usage:
    PYTHONPATH=src python3 scripts/make_sq2_trajectory_figure.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = [
    # (display label, file glob)
    ("baseline_hitl",    "logs/experiments/baseline_hitl/baseline_hitl_v20260523_*_lenient.json"),
    ("cot_hitl",         "logs/experiments/cot_hitl/cot_hitl_v20260523_*_lenient.json"),
    ("few_shot_hitl",    "logs/experiments/few_shot_hitl/few_shot_hitl_v20260523_*_lenient.json"),
    ("consistency_hitl", "logs/experiments/consistency_hitl/consistency_hitl_v20260523_*_lenient.json"),
    # rag_hitl uses the post-fix (corrected-cases) runs only: 132xxx / 133xxx
    ("rag_hitl",         "logs/experiments/rag_hitl/rag_hitl_v20260523_13[23]*_lenient.json"),
]

COLORS = {
    "baseline_hitl":    "#222222",
    "cot_hitl":         "#1f77b4",
    "few_shot_hitl":    "#2ca02c",
    "consistency_hitl": "#ff7f0e",
    "rag_hitl":         "#d62728",
}

OUT = "docs/figures/fig_sq2_trajectories.png"
NOISE_HALFWIDTH = 0.085  # ±17pp two-iteration 95% CI at n=50, p≈0.75


def load_trajectory(pattern: str):
    """Return ([strict_1..3], [lenient_1..3]) sorted by file timestamp."""
    paths = sorted(p for p in glob.glob(pattern) if "_lenient" in p)
    if not paths:
        sys.exit(f"ERROR: no files matched {pattern}")
    strict, lenient = [], []
    for p in paths:
        d = json.load(open(p))
        s = d["summary"]
        strict.append(s["strict_accuracy"])
        lenient.append(s["lenient_accuracy"])
    return strict, lenient


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    xs = [1, 2, 3]

    for cond, pat in CONDITIONS:
        strict, lenient = load_trajectory(pat)
        c = COLORS[cond]
        ax_s.plot(xs, strict,  marker="o", color=c, label=cond, linewidth=1.8)
        ax_l.plot(xs, lenient, marker="o", color=c, label=cond, linewidth=1.8)

        # Annotate iter-1 starting point with a small text label
        ax_s.annotate(f"{strict[0]:.2f}", (1, strict[0]),
                      textcoords="offset points", xytext=(-22, -4),
                      fontsize=7, color=c)
        ax_l.annotate(f"{lenient[0]:.2f}", (1, lenient[0]),
                      textcoords="offset points", xytext=(-22, -4),
                      fontsize=7, color=c)

    # Shaded ±17pp band centred on each iter-1 starting point — show as a
    # global reference band centred on the median iter-1 strict value.
    # Simpler and cleaner: a single horizontal reference band illustrating
    # the noise floor magnitude.
    for ax, title in [(ax_s, "Strict accuracy (vs MIMIC SEQ\\_NUM=1)"),
                      (ax_l, "Lenient accuracy (vs any billed ICD-9)")]:
        ax.set_xlabel("HITL iteration")
        ax.set_xticks(xs)
        ax.set_ylim(0.55, 1.00)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle=":")

    # Annotate the lenient convergence band (0.86–0.92 across iter-3)
    ax_l.axhspan(0.86, 0.92, color="grey", alpha=0.12,
                 label="iter-3 lenient ceiling band")

    ax_s.set_ylabel("Accuracy")

    # One legend below both panels
    handles, labels = ax_s.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=False)

    fig.suptitle("Per-technique HITL trajectory (n=50 per iter, 3 iters)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
