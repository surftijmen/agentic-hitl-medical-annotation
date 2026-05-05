"""
Per-ICD9-chapter accuracy breakdown.

Reads experiment JSON logs, joins each case's gold ICD-9 code to its chapter,
and prints an accuracy table per chapter — so you can see WHERE the strict
metric fails clinically (obstetric/admin vs real clinical diagnoses).

If *_lenient.json companions exist, also reports lenient accuracy per chapter.

Usage:
    PYTHONPATH=src python3 scripts/analyze_by_chapter.py \\
        logs/experiments/baseline/baseline_v20260415_002550.json \\
        logs/experiments/baseline_pro/baseline_pro_v20260417_001808.json \\
        logs/experiments/baseline_v2prompt/baseline_v2prompt_v20260416_231744.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from modules.sampler_agent import _icd9_chapter  # noqa: E402


def _extract_gold_code(rec: dict) -> Optional[str]:
    g = rec.get("gold_reference")
    if isinstance(g, dict):
        return g.get("icd9_code")
    return None


def _load_with_optional_lenient(path: str) -> List[dict]:
    """Load primary log; if a *_lenient.json sibling exists, merge its verdicts."""
    with open(path) as f:
        records = json.load(f)

    lenient_path = path.replace(".json", "_lenient.json")
    if os.path.exists(lenient_path):
        with open(lenient_path) as f:
            lenient = json.load(f).get("cases", [])
        # Align by (subject_id, hadm_id) in order — both files have the same records
        for rec, lrec in zip(records, lenient):
            rec["_lenient_correct"] = lrec.get("lenient_correct")
    return records


def _per_chapter(records: List[dict]) -> Dict[str, Dict]:
    buckets: Dict[str, Dict] = defaultdict(lambda: {
        "n": 0, "strict_correct": 0, "lenient_correct": 0,
        "has_lenient": False, "failure_modes": defaultdict(int),
    })
    for rec in records:
        code = _extract_gold_code(rec) or ""
        chapter = _icd9_chapter(code)
        b = buckets[chapter]
        b["n"] += 1
        if rec.get("human_correct") is True:
            b["strict_correct"] += 1
        else:
            fm = rec.get("human_failure_mode") or "unlabeled"
            b["failure_modes"][fm] += 1
        if "_lenient_correct" in rec:
            b["has_lenient"] = True
            if rec["_lenient_correct"] is True:
                b["lenient_correct"] += 1
    return buckets


def _print_table(label: str, buckets: Dict[str, Dict]) -> None:
    any_lenient = any(b["has_lenient"] for b in buckets.values())
    header = f"\n{'=' * 90}\n  {label}\n{'=' * 90}"
    print(header)
    if any_lenient:
        print(f"  {'chapter':<34} {'n':>4} {'strict':>8} {'lenient':>8} {'Δpp':>5}  top-failure")
    else:
        print(f"  {'chapter':<34} {'n':>4} {'strict':>8}  top-failure")
    print(f"  {'-' * 84}")

    # Sort chapters by size (biggest first)
    ordered = sorted(buckets.items(), key=lambda x: -x[1]["n"])
    total_n = sum(b["n"] for _, b in ordered)
    total_strict = sum(b["strict_correct"] for _, b in ordered)
    total_lenient = sum(b["lenient_correct"] for _, b in ordered if b["has_lenient"])
    total_lenient_n = sum(b["n"] for _, b in ordered if b["has_lenient"])

    for chap, b in ordered:
        strict_acc = b["strict_correct"] / b["n"] if b["n"] else 0
        top_fail = max(b["failure_modes"].items(), key=lambda x: x[1], default=("—", 0))
        fail_str = f"{top_fail[0]}:{top_fail[1]}" if top_fail[1] else "—"
        if b["has_lenient"]:
            lenient_acc = b["lenient_correct"] / b["n"] if b["n"] else 0
            delta = (lenient_acc - strict_acc) * 100
            print(
                f"  {chap:<34} {b['n']:>4} {strict_acc:>8.3f} {lenient_acc:>8.3f} "
                f"{delta:>+5.1f}  {fail_str}"
            )
        else:
            print(f"  {chap:<34} {b['n']:>4} {strict_acc:>8.3f}  {fail_str}")

    print(f"  {'-' * 84}")
    if any_lenient and total_lenient_n:
        print(
            f"  {'TOTAL':<34} {total_n:>4} "
            f"{total_strict/total_n:>8.3f} {total_lenient/total_lenient_n:>8.3f} "
            f"{(total_lenient/total_lenient_n - total_strict/total_n)*100:>+5.1f}"
        )
    else:
        print(f"  {'TOTAL':<34} {total_n:>4} {total_strict/total_n:>8.3f}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: analyze_by_chapter.py <log.json> [<log.json> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"SKIP {path} (not found)")
            continue
        records = _load_with_optional_lenient(path)
        buckets = _per_chapter(records)
        _print_table(os.path.basename(path), buckets)


if __name__ == "__main__":
    main()
