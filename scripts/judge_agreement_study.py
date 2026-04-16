"""
Judge-vs-doctor agreement harness.

Randomly samples K cases from past experiment runs (or from a fresh run),
presents them to a human reviewer via the existing HumanReviewer Tkinter UI,
and compares the doctor's correct/incorrect verdict against what the LLM
judge already recorded. Outputs:
  - Cohen's kappa (raw and 95% CI)
  - Per-case confusion: judge vs doctor
  - Breakdown by failure_mode

Use this to produce the "reliability" evidence your thesis needs: a
quantitative claim about how often the LLM judge agrees with a clinician.

Usage:
    PYTHONPATH=src python3 scripts/judge_agreement_study.py \\
        --source logs/experiments/baseline/baseline_v20260414_221025.json \\
        --k 50

    # Sample across all runs of a condition:
    PYTHONPATH=src python3 scripts/judge_agreement_study.py \\
        --source-glob "logs/experiments/baseline/*.json" --k 50
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from datetime import datetime

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from modules.human_reviewer import HumanReviewer  # noqa: E402


def load_records(source_paths: list[str]) -> list[dict]:
    records = []
    for path in source_paths:
        with open(path) as f:
            data = json.load(f)
        for rec in data:
            rec["_source_file"] = os.path.basename(path)
            records.append(rec)
    return records


def cohen_kappa(agreements: list[tuple[bool, bool]]) -> tuple[float, float, float]:
    """Return (kappa, 95% CI lower, 95% CI upper) for a 2×2 binary confusion.

    agreements = list of (doctor_verdict, judge_verdict) booleans.
    """
    n = len(agreements)
    if n == 0:
        return 0.0, 0.0, 0.0

    a = sum(1 for d, j in agreements if d and j)           # both correct
    b = sum(1 for d, j in agreements if d and not j)       # doc Y, judge N
    c = sum(1 for d, j in agreements if not d and j)       # doc N, judge Y
    d = sum(1 for d, j in agreements if not d and not j)   # both incorrect

    po = (a + d) / n
    pd_yes = ((a + b) / n) * ((a + c) / n)
    pd_no  = ((c + d) / n) * ((b + d) / n)
    pe = pd_yes + pd_no
    if 1 - pe == 0:
        return 1.0, 1.0, 1.0
    kappa = (po - pe) / (1 - pe)

    # Fleiss–Cohen approximation for kappa SE under a 2×2
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    lo = kappa - 1.96 * se
    hi = kappa + 1.96 * se
    return kappa, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Single experiment JSON file to sample from.")
    parser.add_argument("--source-glob", help="Glob of experiment JSON files.")
    parser.add_argument("--k", type=int, default=50, help="Cases to review (default 50).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--out-dir", default="logs/agreement_studies",
        help="Where the result JSON lands.",
    )
    args = parser.parse_args()

    if not (args.source or args.source_glob):
        parser.error("Provide --source or --source-glob.")

    source_paths = [args.source] if args.source else sorted(glob.glob(args.source_glob))
    if not source_paths:
        parser.error(f"No files matched: {args.source_glob}")

    all_records = load_records(source_paths)
    if len(all_records) < args.k:
        print(f"WARNING: only {len(all_records)} cases available; using all of them.")
    random.Random(args.seed).shuffle(all_records)
    sample = all_records[: args.k]

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"agreement_{ts}.json")

    print(f"Will review {len(sample)} cases. Results → {out_path}")
    print("Instructions: mark each case correct/incorrect as a CLINICIAN, based on")
    print("the note and the gold ICD-9. Don't look at the judge's verdict first.")
    print()

    reviewer = HumanReviewer()
    per_case = []

    for i, rec in enumerate(sample, start=1):
        annotation = {
            "diagnosis": rec["model_diagnosis"],
            "is_diagnosis_given": 1 if rec["model_diagnosis"] and rec["model_diagnosis"] != "none" else 0,
            "confidence_level": rec.get("model_confidence", 0),
        }
        doctor_fb = reviewer.review(
            annotation=annotation,
            medical_note=rec["note_text"],
            gold=rec["gold_reference"],
            case_num=i,
            total_cases=len(sample),
        )
        per_case.append({
            "subject_id":       rec["subject_id"],
            "hadm_id":          rec["hadm_id"],
            "source_file":      rec["_source_file"],
            "model_diagnosis":  rec["model_diagnosis"],
            "gold":             rec["gold_reference"],
            "judge_correct":    rec["human_correct"],
            "judge_failure":    rec["human_failure_mode"],
            "judge_comment":    rec["human_comment"],
            "doctor_correct":   doctor_fb["correct"],
            "doctor_failure":   doctor_fb.get("failure_mode"),
            "doctor_comment":   doctor_fb.get("comment"),
            "doctor_confidence": doctor_fb.get("confidence"),
        })

    reviewer.close()

    pairs = [(c["doctor_correct"], c["judge_correct"]) for c in per_case]
    kappa, lo, hi = cohen_kappa(pairs)
    agree = sum(1 for d, j in pairs if d == j)

    # confusion
    matrix = {
        "both_correct":      sum(1 for d, j in pairs if d and j),
        "doctor_only":       sum(1 for d, j in pairs if d and not j),
        "judge_only":        sum(1 for d, j in pairs if not d and j),
        "both_incorrect":    sum(1 for d, j in pairs if not d and not j),
    }

    summary = {
        "n": len(pairs),
        "agreement": agree / len(pairs) if pairs else 0.0,
        "cohen_kappa": kappa,
        "kappa_ci95":  [lo, hi],
        "confusion":   matrix,
        "sources":     source_paths,
        "seed":        args.seed,
    }

    print()
    print("=" * 60)
    print("AGREEMENT RESULTS")
    print("=" * 60)
    print(f"  n:                     {summary['n']}")
    print(f"  raw agreement:         {summary['agreement']:.1%}")
    print(f"  Cohen's kappa:         {summary['cohen_kappa']:.3f} "
          f"(95% CI {summary['kappa_ci95'][0]:.3f}, {summary['kappa_ci95'][1]:.3f})")
    print(f"  both correct:          {matrix['both_correct']}")
    print(f"  doctor only correct:   {matrix['doctor_only']}    (judge false-neg)")
    print(f"  judge only correct:    {matrix['judge_only']}    (judge false-pos)")
    print(f"  both incorrect:        {matrix['both_incorrect']}")
    print()

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "cases": per_case}, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
