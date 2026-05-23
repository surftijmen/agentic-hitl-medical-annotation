"""
Round 2 of the SQ2 gap-fill: re-run few_shot_hitl and rag_hitl with the same
methodology as the May 23 baseline_hitl / cot_hitl / consistency_hitl
trajectories (n=50, 3 iters, auto reviewer on gemini-2.5-flash, fresh start
from v1_initial.json).

Notes:
  • rag_hitl uses a persistent RAG store. To start with an empty store
    (matching the other conditions' fresh starts), delete the pointer file
    *before* running this script:
        rm logs/rag_stores/rag_hitl.txt
    Otherwise the new run resumes the existing store and the trajectory is
    confounded by accumulated cases from old experiments.

Usage:
    PYTHONPATH=src python3 scripts/fill_sq2_round2.py
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
        experiments_to_run=["few_shot_hitl", "rag_hitl"],
        reviewer_mode="auto",
        judge_model="gemini-2.5-flash",
        annotator_model="gemini-2.5-flash",
    )
