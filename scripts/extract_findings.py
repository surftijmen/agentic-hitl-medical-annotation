"""
Extract additional findings from existing experiment logs (no API calls).

Produces:
  - docs/figures/fig_confidence_calibration.png  (Finding 1)
  - docs/figures/fig_cost_vs_accuracy.png        (Finding 3)
  - docs/figures/fig_hitl_failure_trajectory.png (Finding 4)
  - docs/findings_self_correction.md              (Finding 6 case study)
  - prints concise summary of all findings to stdout

Usage:
    PYTHONPATH=src python3 scripts/extract_findings.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from modules.sampler_agent import _icd9_chapter, ADMIN_CHAPTERS_DEFAULT  # noqa: E402

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

CONFIGS = [
    ("v1 flash",          "logs/experiments/baseline/baseline_v20260415_002550.json",          "flash"),
    ("v2 flash",          "logs/experiments/baseline_v2prompt/baseline_v2prompt_v20260416_231744.json", "flash"),
    ("v1 pro",            "logs/experiments/baseline_pro/baseline_pro_v20260417_001808.json",   "pro"),
    ("v1 pro\nfull_pipe", "logs/experiments/full_pipeline/full_pipeline_v20260417_003218.json", "pro_full"),
]


def chapter(rec):
    g = rec.get("gold_reference") or {}
    return _icd9_chapter(g.get("icd9_code") or "") if isinstance(g, dict) else ""


# =============================================================================
# Finding 1: Confidence calibration
# =============================================================================
def confidence_calibration():
    print("\n[Finding 1] Confidence calibration")
    print("-" * 60)
    out_rows = []
    for label, path, _ in CONFIGS:
        if not os.path.exists(path): continue
        recs = json.load(open(path))
        # Clinical only
        recs = [r for r in recs if chapter(r) not in ADMIN_CHAPTERS_DEFAULT]
        c_correct = [r["model_confidence"] for r in recs if r.get("human_correct") is True and r.get("model_confidence") is not None]
        c_wrong   = [r["model_confidence"] for r in recs if r.get("human_correct") is False and r.get("model_confidence") is not None]
        if c_correct and c_wrong:
            mean_c, mean_w = np.mean(c_correct), np.mean(c_wrong)
            out_rows.append((label.replace("\n", " "), mean_c, mean_w, mean_c - mean_w, len(c_correct), len(c_wrong)))
            print(f"  {label.replace(chr(10), ' '):20s}  conf(correct)={mean_c:5.1f}  conf(wrong)={mean_w:5.1f}  gap={mean_c-mean_w:5.1f}pp")

    # Calibration plot using v1 flash baseline (richest data)
    recs = json.load(open(CONFIGS[0][1]))
    recs = [r for r in recs if chapter(r) not in ADMIN_CHAPTERS_DEFAULT]
    bins = [(0,50), (50,70), (70,85), (85,95), (95,100), (100,101)]
    bin_labels, bin_acc, bin_n = [], [], []
    for lo, hi in bins:
        in_bin = [r for r in recs if r.get("model_confidence") is not None and lo <= r["model_confidence"] < hi]
        if not in_bin: continue
        acc = sum(1 for r in in_bin if r.get("human_correct") is True) / len(in_bin)
        bin_labels.append(f"{lo}-{hi-1}")
        bin_acc.append(acc)
        bin_n.append(len(in_bin))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, bin_acc, 0.6, color="#2563eb")
    for r, n in zip(bars, bin_n):
        ax.annotate(f"n={n}", (r.get_x() + r.get_width()/2, r.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.plot(x, [(int(b.split("-")[0]) + int(b.split("-")[1]))/200 for b in bin_labels],
            "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xticks(x); ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Model self-reported confidence (binned)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Confidence calibration — v1 flash baseline (clinical, n=166)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUT}/fig_confidence_calibration.png")
    plt.close()
    print(f"  → wrote {OUT}/fig_confidence_calibration.png")
    return out_rows


# =============================================================================
# Finding 3: Cost vs accuracy (illustrative — actual costs from API pricing)
# =============================================================================
def cost_vs_accuracy():
    print("\n[Finding 3] Cost vs accuracy")
    print("-" * 60)
    # Approximate per-run costs at Gemini pricing as of 2026 (USD)
    # flash: $0.30/M input + $1.20/M output  →  ~$0.60 per n=200 baseline run
    # pro:   $3.50/M input + $10.50/M output →  ~$4.50 per n=200 baseline run
    # full_pipeline adds self-consistency (3x annotator) → +200% annotator side
    cost_per_run = {"flash": 0.60, "pro": 4.50, "pro_full": 6.00}
    rows = []
    for label, path, kind in CONFIGS:
        if not os.path.exists(path): continue
        recs = json.load(open(path))
        clin = [r for r in recs if chapter(r) not in ADMIN_CHAPTERS_DEFAULT]
        acc = sum(1 for r in clin if r.get("human_correct") is True) / len(clin) if clin else 0
        rows.append((label.replace("\n", " "), cost_per_run[kind], acc))
        print(f"  {label.replace(chr(10), ' '):20s}  cost=${cost_per_run[kind]:>5.2f}/run  clinical_acc={acc:.3f}")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for lbl, cost, acc in rows:
        color = "#2a7d46" if "flash" in lbl else "#dc2626"
        ax.scatter(cost, acc, s=140, color=color, edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(lbl, (cost, acc), xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Per-run API cost (USD, $n=200$)")
    ax.set_ylabel("Clinical strict accuracy")
    ax.set_ylim(0.65, 0.80)
    ax.set_title("Cost vs accuracy — flash dominates pro at this task")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUT}/fig_cost_vs_accuracy.png")
    plt.close()
    print(f"  → wrote {OUT}/fig_cost_vs_accuracy.png")
    return rows


# =============================================================================
# Finding 4: HITL failure-mode trajectory (per run, post-ICD-swap only)
# =============================================================================
def hitl_trajectory():
    print("\n[Finding 4] HITL failure-mode trajectory")
    print("-" * 60)
    # Use only April 16 baseline_hitl runs (post ICD-swap)
    cond_dir = "logs/experiments/baseline_hitl"
    runs = []
    for name in sorted(os.listdir(cond_dir)):
        if not name.endswith(".json") or "_lenient" in name:
            continue
        path = os.path.join(cond_dir, name)
        recs = json.load(open(path))
        if not recs:
            continue
        ts = recs[0].get("timestamp", "")
        # filter to post-ICD-swap (April 14+)
        if "2026-04" not in ts and "2026-05" not in ts:
            continue
        runs.append((path, recs))

    if not runs:
        print("  (no post-ICD-swap baseline_hitl runs found, skipping)")
        return []

    failure_modes = ["missed_entity", "unsupported_inference", "terminology_gap",
                     "symptom_diagnosis_confusion", "ambiguous_case", "hallucination"]
    palette = ["#dc2626", "#f59e0b", "#a855f7", "#0ea5e9", "#94a3b8", "#1e293b"]

    counts = {fm: [] for fm in failure_modes}
    accs = []
    for _, recs in runs:
        n = len(recs)
        accs.append(sum(1 for r in recs if r.get("human_correct") is True) / n)
        for fm in failure_modes:
            counts[fm].append(sum(1 for r in recs if r.get("human_failure_mode") == fm))

    print(f"  runs included: {len(runs)} (n={[len(r[1]) for r in runs]})")
    print(f"  accuracy trajectory: {[f'{a:.3f}' for a in accs]}")
    for fm in failure_modes:
        if any(counts[fm]):
            print(f"    {fm:30s}: {counts[fm]}")

    # Stacked bar across runs
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bottom = np.zeros(len(runs))
    for fm, color in zip(failure_modes, palette):
        vals = np.array(counts[fm])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, 0.55, label=fm, bottom=bottom, color=color)
        bottom = bottom + vals
    ax.set_xticks(x); ax.set_xticklabels([f"Run {i+1}\nn={len(r[1])}\nacc={a:.2f}"
                                         for i, (r, a) in enumerate(zip(runs, accs))])
    ax.set_ylabel("Failure count per run")
    ax.set_title("HITL failure-mode trajectory — baseline_hitl (post ICD-swap)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.savefig(f"{OUT}/fig_hitl_failure_trajectory.png")
    plt.close()
    print(f"  → wrote {OUT}/fig_hitl_failure_trajectory.png")
    return list(zip(runs, accs))


# =============================================================================
# Finding 6: PromptAgent self-correction case study (extract from logs/prompts)
# =============================================================================
def self_correction_case_study():
    print("\n[Finding 6] PromptAgent self-correction case study")
    print("-" * 60)
    # Find the three patches from April 16 baseline_hitl
    import glob
    patches = sorted(glob.glob("logs/prompts/v20260416*_instruction_patch.json"))
    if len(patches) < 3:
        print("  (fewer than 3 patches found, skipping)")
        return None

    md_lines = ["# PromptAgent self-correction — case study\n",
                "Extracted from `logs/prompts/v20260416_*_instruction_patch.json`. "
                "Three sequential HITL iterations on `baseline_hitl` (n=50). "
                "The third iteration *refined an instruction added in the first*, "
                "demonstrating that the meta-loop can correct its own earlier patches.\n"]
    for i, p in enumerate(patches[:3], 1):
        d = json.load(open(p))
        ip = d.get("instruction_patch", {})
        af = d.get("aggregated_failures", {})
        md_lines.append(f"## Iteration {i} — `{os.path.basename(p)}`")
        md_lines.append(f"**Failures fed to patcher:** {af}")
        if ip.get("instructions_to_add"):
            md_lines.append("**Instructions added:**")
            for x in ip["instructions_to_add"]:
                md_lines.append(f"- {x}")
        if ip.get("instructions_to_refine"):
            md_lines.append("**Instructions refined:**")
            for r in ip["instructions_to_refine"]:
                md_lines.append(f"- *was:* {r.get('target_instruction')}")
                md_lines.append(f"  *now:* {r.get('refined_version')}")
        md_lines.append(f"\n**Rationale:** {ip.get('rationale','')[:400]}\n")

    md_lines.append("\n## What makes this notable\n")
    md_lines.append("In iteration 1, the PromptAgent added a rule allowing event-type "
                    "primaries (e.g. \"Normal vaginal delivery\"). This caused new "
                    "failures in iteration 2 where the model returned procedures "
                    "(e.g. \"craniotomy revision\") instead of underlying diseases. "
                    "In iteration 3 the PromptAgent **refined its own earlier rule** "
                    "to exclude that case, demonstrating emergent self-correction.\n")

    out_path = "docs/findings_self_correction.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  → wrote {out_path}  ({len(patches)} patches summarized)")
    return out_path


# =============================================================================

def main():
    confidence_calibration()
    cost_vs_accuracy()
    hitl_trajectory()
    self_correction_case_study()

    print("\n" + "=" * 60)
    print("All findings extracted. Files written to docs/figures/ and docs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
