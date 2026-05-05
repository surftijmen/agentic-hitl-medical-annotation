"""
Generate the publication figures for the paper.

Produces three PNGs in docs/figures/:
  - fig_clinical_vs_admin.png — bar chart of strict accuracy on clinical
    vs administrative chapters across the 4 configurations.
  - fig_failure_modes.png — stacked bar chart of failure-mode counts per
    configuration.
  - fig_chapter_breakdown.png — per-ICD9-chapter strict vs lenient
    accuracy for the v1 flash baseline (the one with lenient data).

All numbers come from existing experiment logs — no API calls needed.

Usage:
    PYTHONPATH=src python3 scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from modules.sampler_agent import _icd9_chapter, ADMIN_CHAPTERS_DEFAULT  # noqa: E402

OUT_DIR = "docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Use a clean publication-quality style without seaborn dependency
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

CONFIGS = [
    ("v1 flash",          "logs/experiments/baseline/baseline_v20260415_002550.json"),
    ("v2 flash",          "logs/experiments/baseline_v2prompt/baseline_v2prompt_v20260416_231744.json"),
    ("v1 pro",            "logs/experiments/baseline_pro/baseline_pro_v20260417_001808.json"),
    ("v1 pro\nfull_pipeline r1", "logs/experiments/full_pipeline/full_pipeline_v20260417_003218.json"),
]


def _chapter(rec) -> str:
    g = rec.get("gold_reference") or {}
    return _icd9_chapter(g.get("icd9_code") or "") if isinstance(g, dict) else "Unknown"


def _load(path: str) -> List[Dict]:
    if not os.path.exists(path):
        print(f"WARN: {path} not found — skipping")
        return []
    return json.load(open(path))


# -------------------------------------------------------------------------
# Figure 1: Clinical vs admin strict accuracy across configs
# -------------------------------------------------------------------------
def fig_clinical_vs_admin():
    labels, clin_acc, adm_acc, overall = [], [], [], []
    for label, path in CONFIGS:
        recs = _load(path)
        if not recs:
            continue
        clin = [r for r in recs if _chapter(r) not in ADMIN_CHAPTERS_DEFAULT]
        adm  = [r for r in recs if _chapter(r)     in ADMIN_CHAPTERS_DEFAULT]
        labels.append(label)
        clin_acc.append(sum(1 for r in clin if r.get("human_correct") is True) / len(clin) if clin else 0)
        adm_acc.append( sum(1 for r in adm  if r.get("human_correct") is True) / len(adm)  if adm  else 0)
        overall.append(sum(1 for r in recs if r.get("human_correct") is True) / len(recs))

    x = np.arange(len(labels))
    w = 0.27

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    b1 = ax.bar(x - w, clin_acc, w, label="Clinical (n=166)", color="#2a7d46")
    b2 = ax.bar(x,     adm_acc,  w, label="Admin (n=34)",     color="#d97706")
    b3 = ax.bar(x + w, overall,  w, label="Overall (n=200)",  color="#3b82f6")

    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Strict accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Clinical-chapter accuracy is consistent ($\\sim$0.74); admin-chapter accuracy is volatile")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    out = os.path.join(OUT_DIR, "fig_clinical_vs_admin.png")
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# -------------------------------------------------------------------------
# Figure 2: Failure-mode distribution per config (stacked bar)
# -------------------------------------------------------------------------
def fig_failure_modes():
    failure_modes = ["missed_entity", "unsupported_inference",
                     "terminology_gap", "symptom_diagnosis_confusion",
                     "ambiguous_case", "hallucination"]
    palette = ["#dc2626", "#f59e0b", "#a855f7", "#0ea5e9", "#94a3b8", "#1e293b"]

    labels, counts_per_mode = [], {fm: [] for fm in failure_modes}
    for label, path in CONFIGS:
        recs = _load(path)
        if not recs:
            continue
        labels.append(label.replace("\n", " "))
        for fm in failure_modes:
            counts_per_mode[fm].append(
                sum(1 for r in recs if r.get("human_failure_mode") == fm)
            )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    bottom = np.zeros(len(labels))
    for fm, color in zip(failure_modes, palette):
        vals = np.array(counts_per_mode[fm])
        ax.bar(x, vals, 0.55, label=fm, bottom=bottom, color=color)
        bottom = bottom + vals

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Failure count (n=200)")
    ax.set_title("Failure-mode redistribution across prompt and model variants")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    out = os.path.join(OUT_DIR, "fig_failure_modes.png")
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# -------------------------------------------------------------------------
# Figure 3: Per-chapter strict vs lenient (v1 flash baseline)
# -------------------------------------------------------------------------
def fig_chapter_breakdown():
    strict_path  = "logs/experiments/baseline/baseline_v20260415_002550.json"
    lenient_path = "logs/experiments/baseline/baseline_v20260415_002550_lenient.json"
    if not (os.path.exists(strict_path) and os.path.exists(lenient_path)):
        print("WARN: lenient file missing — skipping per-chapter figure")
        return

    recs    = json.load(open(strict_path))
    lenient = json.load(open(lenient_path))["cases"]
    lookup_l = {(c["subject_id"], c["hadm_id"]): c for c in lenient}

    buckets: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "strict": 0, "lenient": 0})
    for rec in recs:
        ch = _chapter(rec)
        b  = buckets[ch]
        b["n"] += 1
        if rec.get("human_correct") is True:
            b["strict"] += 1
        lv = lookup_l.get((rec["subject_id"], rec["hadm_id"]))
        if lv and lv.get("lenient_correct") is True:
            b["lenient"] += 1

    # Sort: clinical chapters first (by number prefix), then admin
    def sort_key(item):
        ch = item[0]
        if ch in ADMIN_CHAPTERS_DEFAULT:
            return (1, ch)
        return (0, ch)

    items = sorted(buckets.items(), key=sort_key)
    chapters  = [k.split(":")[0] if ":" in k else k for k, _ in items]
    strict    = [b["strict"] / b["n"] if b["n"] else 0 for _, b in items]
    lenient_a = [b["lenient"] / b["n"] if b["n"] else 0 for _, b in items]
    is_admin  = [k in ADMIN_CHAPTERS_DEFAULT for k, _ in items]

    x = np.arange(len(chapters))
    w = 0.4

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(x - w/2, strict,    w, label="Strict (SEQ_NUM=1)",      color="#3b82f6")
    ax.bar(x + w/2, lenient_a, w, label="Lenient (any billed ICD)", color="#a855f7")

    # Mark admin chapters with shaded background
    for i, adm in enumerate(is_admin):
        if adm:
            ax.axvspan(i - 0.5, i + 0.5, color="#fef3c7", alpha=0.6, zorder=-1)

    ax.set_xticks(x); ax.set_xticklabels(chapters, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strict vs lenient accuracy per ICD-9 chapter (v1 flash baseline, n=200)\nshaded = administrative chapters (excluded from headline metric)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    out = os.path.join(OUT_DIR, "fig_chapter_breakdown.png")
    plt.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# -------------------------------------------------------------------------

def main():
    print("Generating figures...")
    fig_clinical_vs_admin()
    fig_failure_modes()
    fig_chapter_breakdown()
    print("Done. Files written to docs/figures/")


if __name__ == "__main__":
    main()
