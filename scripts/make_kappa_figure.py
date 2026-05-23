"""Visualise the judge-vs-reviewer agreement study.

Reads logs/agreement_studies/agreement_*.json and produces:
  - docs/figures/fig_kappa_confusion.png  (2x2 verdict confusion + kappa display)
  - docs/figures/fig_kappa_failure_modes.png  (failure-mode agreement on both-wrong subset)

Usage:
    python3 scripts/make_kappa_figure.py [path/to/agreement_*.json]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        candidates = sorted(glob.glob(os.path.join(ROOT, "logs/agreement_studies/agreement_*.json")))
        if not candidates:
            print("No agreement study log found.")
            sys.exit(1)
        path = candidates[-1]
    print(f"Using {path}")

    data = json.load(open(path))
    s = data["summary"]
    cases = data["cases"]

    # ---------- Figure 1: 2x2 verdict confusion matrix ----------
    confusion = s["confusion"]
    matrix = np.array([
        [confusion["both_correct"], confusion["doctor_only"]],
        [confusion["judge_only"], confusion["both_incorrect"]],
    ])
    n = s["n"]
    kappa = s["cohen_kappa"]
    ci_lo, ci_hi = s["kappa_ci95"]
    agreement = s["agreement"]

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.1)
    for i in range(2):
        for j in range(2):
            v = matrix[i, j]
            color = "white" if v > matrix.max() * 0.55 else "#0C2340"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Judge: correct", "Judge: incorrect"], fontsize=10)
    ax.set_yticklabels(["Reviewer: correct", "Reviewer: incorrect"], fontsize=10)
    ax.set_title(
        f"Judge-vs-reviewer verdict confusion (n={n})\n"
        f"Cohen's $\\kappa$ = {kappa:.3f}  (95% CI [{ci_lo:.3f}, {ci_hi:.3f}]),  "
        f"raw agreement {agreement:.0%}",
        fontsize=11, color="#0C2340", pad=14,
    )
    plt.tight_layout()

    out1 = os.path.join(ROOT, "docs/figures/fig_kappa_confusion.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    # ---------- Figure 2: failure-mode disagreement on both-wrong subset ----------
    both_wrong = [c for c in cases if not c["doctor_correct"] and not c["judge_correct"]]
    if not both_wrong:
        print("No both-wrong cases; skipping failure-mode figure.")
        return

    modes = [
        "missed_entity", "terminology_gap", "unsupported_inference",
        "symptom_diagnosis_confusion", "hallucination", "ambiguous_case",
    ]
    short = {
        "missed_entity": "missed",
        "terminology_gap": "term.",
        "unsupported_inference": "unsupp.",
        "symptom_diagnosis_confusion": "sym↔dx",
        "hallucination": "halluc.",
        "ambiguous_case": "ambig.",
    }
    j_counts = Counter([c["judge_failure"] for c in both_wrong])
    r_counts = Counter([(c["doctor_failure"] or "").lower() for c in both_wrong])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(modes))
    width = 0.38
    ax.bar(x - width / 2, [j_counts.get(m, 0) for m in modes], width,
           label="LLM judge", color="#3b82f6", edgecolor="#0C2340", linewidth=0.6)
    ax.bar(x + width / 2, [r_counts.get(m, 0) for m in modes], width,
           label="Human reviewer", color="#f59e0b", edgecolor="#0C2340", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in modes], fontsize=10)
    ax.set_ylabel("Cases", fontsize=11)
    ax.set_title(
        f"Failure-mode label assigned on the {len(both_wrong)} both-wrong cases\n"
        f"(agreement on mode = {sum(1 for c in both_wrong if (c['judge_failure'] or '').lower() == (c['doctor_failure'] or '').lower())}/{len(both_wrong)})",
        fontsize=11, color="#0C2340", pad=10,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out2 = os.path.join(ROOT, "docs/figures/fig_kappa_failure_modes.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
