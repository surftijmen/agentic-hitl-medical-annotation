"""
HITL learning-trajectory analyzer.

Reads all runs of a condition (e.g. baseline_hitl has 3 sequential runs under
logs/experiments/baseline_hitl/) and shows how the failure-mode mix changes
across runs — the actual mechanism of HITL working or not.

Also tracks prompt complexity across runs (chars, instruction count) from the
prompt files referenced in each run's records.

Usage:
    PYTHONPATH=src python3 scripts/analyze_hitl_trajectory.py baseline_hitl
    PYTHONPATH=src python3 scripts/analyze_hitl_trajectory.py full_pipeline
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)


def _load_runs(condition: str) -> List[Dict]:
    """Return runs sorted by timestamp, each with {path, records, prompt_path}."""
    cond_dir = os.path.join("logs/experiments", condition)
    if not os.path.isdir(cond_dir):
        sys.exit(f"ERROR: {cond_dir} not found")

    runs = []
    for name in sorted(os.listdir(cond_dir)):
        if not name.endswith(".json") or name.endswith("_lenient.json"):
            continue
        path = os.path.join(cond_dir, name)
        with open(path) as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            continue
        prompt_path = records[0].get("prompt_version", "unknown")
        runs.append({
            "path": path,
            "records": records,
            "prompt_path": prompt_path,
            "timestamp": records[0].get("timestamp", name),
        })
    runs.sort(key=lambda r: r["timestamp"])
    return runs


def _prompt_stats(prompt_path: str) -> Dict:
    if not os.path.exists(prompt_path):
        return {"chars": 0, "num_instructions": 0, "exists": False}
    try:
        with open(prompt_path) as f:
            p = json.load(f)
    except Exception:
        return {"chars": 0, "num_instructions": 0, "exists": False}
    text = " ".join([
        p.get("system_prompt", ""),
        p.get("query", ""),
        " ".join(p.get("instructions", []) or []),
        p.get("format_instruction", ""),
    ])
    return {
        "chars": len(text),
        "num_instructions": len(p.get("instructions", []) or []),
        "exists": True,
    }


def _summarize(records: List[Dict]) -> Dict:
    n = len(records)
    correct = sum(1 for r in records if r.get("human_correct") is True)
    fms = Counter(
        r.get("human_failure_mode") for r in records
        if r.get("human_correct") is False and r.get("human_failure_mode")
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "failure_modes": dict(fms),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    condition = sys.argv[1]
    runs = _load_runs(condition)
    if not runs:
        sys.exit(f"No run logs found for condition '{condition}'")

    print(f"\n{'=' * 90}")
    print(f"  HITL TRAJECTORY — {condition}  ({len(runs)} runs)")
    print(f"{'=' * 90}\n")

    # Per-run summary
    all_fm_keys = set()
    summaries = []
    for run in runs:
        s = _summarize(run["records"])
        s["prompt_path"] = run["prompt_path"]
        s["prompt_stats"] = _prompt_stats(run["prompt_path"])
        s["timestamp"] = run["timestamp"][:19]
        summaries.append(s)
        all_fm_keys.update(s["failure_modes"].keys())

    # Header
    fm_ordered = sorted(all_fm_keys)
    print(f"  {'run':<4} {'timestamp':<19}  {'n':>4} {'acc':>6}  "
          f"{'prompt_chars':>12} {'#inst':>5}  "
          + "  ".join(f"{fm[:18]:>18}" for fm in fm_ordered))
    print(f"  {'-' * (42 + 20 * len(fm_ordered))}")

    for i, s in enumerate(summaries, 1):
        fm_counts = "  ".join(f"{s['failure_modes'].get(fm, 0):>18}" for fm in fm_ordered)
        print(
            f"  {i:<4} {s['timestamp']:<19}  {s['n']:>4} {s['accuracy']:>6.3f}  "
            f"{s['prompt_stats']['chars']:>12} {s['prompt_stats']['num_instructions']:>5}  "
            f"{fm_counts}"
        )

    # Deltas run 1 → final
    if len(summaries) >= 2:
        first, last = summaries[0], summaries[-1]
        print(f"\n  {'-' * 50}")
        print(f"  DELTA run 1 → run {len(summaries)}")
        print(f"    accuracy: {first['accuracy']:.3f} → {last['accuracy']:.3f} "
              f"({(last['accuracy']-first['accuracy'])*100:+.1f}pp)")
        print(f"    prompt chars: {first['prompt_stats']['chars']} → "
              f"{last['prompt_stats']['chars']} "
              f"({last['prompt_stats']['chars']-first['prompt_stats']['chars']:+d})")
        print(f"    # instructions: {first['prompt_stats']['num_instructions']} → "
              f"{last['prompt_stats']['num_instructions']}")
        for fm in fm_ordered:
            d = last["failure_modes"].get(fm, 0) - first["failure_modes"].get(fm, 0)
            if d:
                print(f"    {fm:<30} {first['failure_modes'].get(fm, 0):>3} → "
                      f"{last['failure_modes'].get(fm, 0):>3} ({d:+d})")

    # What SHIFTED between runs? Same subjects across runs?
    # Build subject_id → list of correct/incorrect per run
    subj_trajectory: Dict[int, List[bool]] = defaultdict(list)
    for s_idx, run in enumerate(runs):
        for rec in run["records"]:
            subj_trajectory[rec["subject_id"]].append(rec.get("human_correct") is True)

    # Cases that appeared in ALL runs and changed verdict
    all_runs_subjects = [
        sid for sid, trajectory in subj_trajectory.items()
        if len(trajectory) == len(runs)
    ]
    if all_runs_subjects:
        recovered = [s for s in all_runs_subjects
                     if subj_trajectory[s][0] is False and subj_trajectory[s][-1] is True]
        regressed = [s for s in all_runs_subjects
                     if subj_trajectory[s][0] is True and subj_trajectory[s][-1] is False]
        print(f"\n  CASES APPEARING IN ALL {len(runs)} RUNS: {len(all_runs_subjects)}")
        print(f"    recovered (❌→✓ across HITL): {len(recovered)}")
        print(f"    regressed (✓→❌ across HITL): {len(regressed)}")
        print(f"    net: {len(recovered) - len(regressed):+d}")


if __name__ == "__main__":
    main()
