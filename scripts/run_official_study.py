"""
Official thesis experiment runner.

Executes the longitudinal study: 6 conditions × 3 runs × n=200, with auto
review via Gemini. Results land in logs/experiments/ as per-run dataframes
plus one top-level summary JSON for the cross-condition comparison table.

Usage:
    PYTHONPATH=src python3 scripts/run_official_study.py

Options:
    --conditions a,b,c   Restrict to named conditions
    --runs N             Override number of runs (default 3)
    --n N                Override sample size per run (default 200)
    --judge MODEL        Override judge model (default gemini-2.5-flash)
"""

from __future__ import annotations

import argparse
import os
import sys

# main.py lives in src/, so we need src on the path before importing
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from main import EXPERIMENTS, run_longitudinal_study  # noqa: E402


DEFAULT_CONDITIONS = [
    "baseline",
    "rag_only",
    "cot_only",
    "few_shot_only",
    "cot_plus_few_shot",
    "full_pipeline",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated condition names. Default: the 6 core conditions.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per condition.")
    parser.add_argument("--n",    type=int, default=200, help="Samples per run.")
    parser.add_argument(
        "--judge",
        default="gemini-2.5-flash",
        help="Gemini judge model (structured output enforced in AutoReviewer).",
    )
    parser.add_argument(
        "--annotator-model",
        default="gemini-2.5-flash",
        help="Gemini model used by the annotator. Default gemini-2.5-flash matches existing baseline; use gemini-2.5-pro for the Layer-3 upgrade study.",
    )
    parser.add_argument(
        "--prompt",
        default="logs/prompts/v1_initial.json",
        help="Starting prompt (v1_initial.json has the whitelist removed; v2_primary_focus.json is the Layer-2 improved prompt).",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in EXPERIMENTS]
    if unknown:
        print(f"ERROR: unknown conditions: {unknown}", file=sys.stderr)
        print(f"Available: {sorted(EXPERIMENTS)}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("OFFICIAL STUDY")
    print("=" * 70)
    print(f"  conditions      : {conditions}")
    print(f"  runs            : {args.runs}")
    print(f"  n per run       : {args.n}")
    print(f"  annotator model : {args.annotator_model}")
    print(f"  judge model     : {args.judge}")
    print(f"  prompt          : {args.prompt}")
    print(f"  total           : {len(conditions) * args.runs * args.n} annotator calls "
          f"+ {len(conditions) * args.runs * args.n} judge calls")
    print("=" * 70)
    print()

    results = run_longitudinal_study(
        num_runs=args.runs,
        sample_size=args.n,
        prompt_path=args.prompt,
        experiments_to_run=conditions,
        reviewer_mode="auto",
        judge_model=args.judge,
        annotator_model=args.annotator_model,
    )

    print()
    print("Done. Summary JSON is in logs/experiments/longitudinal_*.json")
    print("Per-run DataFrames are in logs/experiments/<condition>/*.json")
    return results


if __name__ == "__main__":
    main()
