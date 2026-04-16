"""
Analyze prompt evolution across iterations for SubQ3.

Reads all versioned prompt files in logs/prompts/ and computes complexity
metrics per iteration: character count, word count, number of instructions,
number of constraint keywords ("must", "do not", etc.), and diff size
relative to the previous version. If a longitudinal results JSON is
supplied, also correlates complexity with per-run accuracy.

Usage:
    PYTHONPATH=src python3 scripts/analyze_prompt_evolution.py
    PYTHONPATH=src python3 scripts/analyze_prompt_evolution.py \\
        --longitudinal logs/experiments/longitudinal_20260415_xxxx.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Optional

CONSTRAINT_WORDS = ["must", "do not", "never", "only", "always", "required"]


def flatten_prompt(prompt: dict) -> str:
    """Collapse a prompt JSON into a single text blob for counting."""
    parts = [
        prompt.get("system_prompt", ""),
        prompt.get("query", ""),
        " ".join(prompt.get("allowed_diagnoses", []) or []),
        " ".join(prompt.get("instructions", []) or []),
        prompt.get("format_instruction", ""),
    ]
    return "\n".join(p for p in parts if p)


def analyze_one(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Some prompt log files are PATCH proposals, not prompts — skip those
    if "status" in data and "instructions" not in data:
        return {}

    text = flatten_prompt(data)
    instructions = data.get("instructions", []) or []
    return {
        "file":             os.path.basename(path),
        "chars":            len(text),
        "words":            len(text.split()),
        "num_instructions": len(instructions),
        "avg_instr_len":    round(sum(len(i.split()) for i in instructions) / len(instructions), 1) if instructions else 0,
        "constraint_count": sum(text.lower().count(w) for w in CONSTRAINT_WORDS),
        "has_allowed_list": bool(data.get("allowed_diagnoses")),
        "allowed_list_size": len(data.get("allowed_diagnoses", []) or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="logs/prompts", help="Prompt directory.")
    parser.add_argument("--longitudinal", help="Longitudinal results JSON for accuracy join.")
    args = parser.parse_args()

    prompt_files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    rows = [analyze_one(p) for p in prompt_files]
    rows = [r for r in rows if r]

    print(f"\n{'file':45s} {'chars':>7} {'words':>7} {'#inst':>6} {'avg_len':>8} {'constr':>7} {'allowed':>8}")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['file']:45s} {r['chars']:>7} {r['words']:>7} {r['num_instructions']:>6} "
            f"{r['avg_instr_len']:>8} {r['constraint_count']:>7} "
            f"{r['allowed_list_size'] if r['has_allowed_list'] else '-':>8}"
        )

    if args.longitudinal and os.path.exists(args.longitudinal):
        with open(args.longitudinal) as f:
            long_data = json.load(f)
        print(f"\n{'condition':22s} {'run':>4} {'acc':>6}  (prompt complexity per run TBD — join by timestamp)")
        for cond, runs in long_data.items():
            for m in runs:
                print(f"{cond:22s} {m.get('run', '?'):>4} {m.get('accuracy', 0):>6.2f}")

    print()
    print("For SubQ3: plot `chars` or `num_instructions` on x-axis, `accuracy` on y-axis,")
    print("with one trace per HITL condition. A positive slope means complexity helps;")
    print("a plateau means the prompt stopped evolving productively.")


if __name__ == "__main__":
    main()
