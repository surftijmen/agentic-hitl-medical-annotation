"""
Run the three HITL conditions still missing from the SQ2 comparison:
  • cot_hitl              — CoT + prompt patching + feedback routing
  • consistency_hitl      — self-consistency + prompt patching + routing
  • baseline_hitl         — re-run to extend its current 2-iter trajectory to 3

All three at n=50, 3 iters, auto reviewer (gemini-2.5-flash). Saves the
usual per-condition longitudinal JSON under logs/experiments/longitudinal_*.json
and per-run dataframes under logs/experiments/<condition>/.

Usage:
    PYTHONPATH=src python3 scripts/fill_sq2_gaps.py
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from main import run_longitudinal_study  # noqa: E402


if __name__ == "__main__":
    run_longitudinal_study(
        num_runs=3,
        sample_size=50,
        debug=False,
        prompt_path="logs/prompts/v1_initial.json",
        experiments_to_run=["cot_hitl", "consistency_hitl", "baseline_hitl"],
        reviewer_mode="auto",
        judge_model="gemini-2.5-flash",
        annotator_model="gemini-2.5-flash",
    )
