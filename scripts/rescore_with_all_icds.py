"""
Dual-metric re-scoring: strict (primary ICD-9 only) vs lenient (any billed
ICD-9 for the admission).

Loads an existing experiment JSON (the per-run dataframe written by
run_pipeline) and re-judges every case against the FULL list of ICD-9 codes
for that admission. Writes <original>_lenient.json alongside the input.

Rationale: ~30% of errors in the flash baseline are cases where the annotator
extracted a clinically legitimate diagnosis from the note but it didn't match
the single billed SEQ_NUM=1 code (MIMIC billing artifacts — V-codes, obstetric
admin labels, "Other/Unspecified" catch-alls). A clinician would call those
"correct, got one of the billed diagnoses." Reporting strict AND lenient
accuracy is the honest clinical picture.

Usage:
    PYTHONPATH=src python3 scripts/rescore_with_all_icds.py \\
        --source logs/experiments/baseline/baseline_v20260415_002550.json

    # Re-score every run of a condition:
    PYTHONPATH=src python3 scripts/rescore_with_all_icds.py \\
        --source-glob "logs/experiments/baseline/*.json"

    # Only recompute the summary, don't re-call the judge (fast):
    PYTHONPATH=src python3 scripts/rescore_with_all_icds.py \\
        --source ... --summary-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
import toml
from google import genai
from google.genai import types

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)

from modules.auto_reviewer import FAILURE_MODES, _JUDGMENT_SCHEMA  # noqa: E402


DIAGNOSES_ICD_CSV = "data/new/DIAGNOSES_ICD.csv"
D_ICD_DIAGNOSES_CSV = "data/new/D_ICD_DIAGNOSES.csv"


# Judge prompt for lenient scoring — reuses the same failure-mode taxonomy
# but tells the model the gold is a LIST and any semantic match counts.
_LENIENT_SYSTEM_PROMPT = """\
You are a senior clinician evaluating an LLM medical-text annotation system.

The annotator produced a single free-text diagnosis for an admission. For
evaluation, you receive the FULL set of ICD-9 codes billed for this admission
(not just the primary). Your task: decide whether the annotator's diagnosis
is semantically consistent with ANY of the billed ICD-9 codes for this
admission, given what the note says.

━━━ INPUTS ━━━
• MEDICAL NOTE — the (possibly redacted) discharge summary.
• ANNOTATOR OUTPUT — free-text `diagnosis` plus confidence + optional reasoning.
• BILLED ICD-9 DIAGNOSES — ranked list of codes for this admission (SEQ_NUM,
  ICD9, short_title, long_title). SEQ_NUM=1 is the primary/principal diagnosis.

━━━ WHAT COUNTS AS CORRECT (LENIENT) ━━━
Mark CORRECT if the annotator's diagnosis is semantically consistent with
ANY one of the billed ICD-9 long_titles — primary or secondary. The same
rules as the strict judge apply per-code:
• Accept synonyms, abbreviations, sub-types, clinical equivalents.
• Reject different organ systems, symptom-as-diagnosis, unsupported inferences.
• The annotator's answer must still be supported by the note text.

In your `comment`, name which billed code matched and whether it was
SEQ_NUM=1 (primary) or a secondary billed code.

━━━ FAILURE MODES (when INCORRECT) ━━━
Same taxonomy as strict scoring: hallucination, missed_entity,
terminology_gap, unsupported_inference, symptom_diagnosis_confusion,
ambiguous_case. Use "none" when correct.

━━━ OUTPUT ━━━
Structured JSON matching the enforced schema: correct (bool), failure_mode,
comment, confidence (0-100).

The `comment` field must be ONE short sentence (max ~25 words). Name the
matching billed code and its SEQ_NUM if correct; name the failure cause if
incorrect. Do not quote long passages.
"""


def _load_admission_icds() -> Dict[int, List[Dict[str, Any]]]:
    """Return {HADM_ID: [{icd9_code, short_title, long_title, seq_num}, ...]}."""
    if not os.path.exists(DIAGNOSES_ICD_CSV):
        sys.exit(f"ERROR: {DIAGNOSES_ICD_CSV} not found.")
    if not os.path.exists(D_ICD_DIAGNOSES_CSV):
        sys.exit(f"ERROR: {D_ICD_DIAGNOSES_CSV} not found.")

    dict_df = pd.read_csv(D_ICD_DIAGNOSES_CSV, dtype=str, keep_default_na=False)
    dict_df["ICD9_CODE"] = dict_df["ICD9_CODE"].str.strip()
    lookup = {
        r.ICD9_CODE: (r.SHORT_TITLE.strip(), r.LONG_TITLE.strip())
        for r in dict_df.itertuples(index=False)
    }

    diag_df = pd.read_csv(DIAGNOSES_ICD_CSV, dtype={"ICD9_CODE": str})
    diag_df = diag_df.dropna(subset=["HADM_ID", "SEQ_NUM", "ICD9_CODE"])
    diag_df["ICD9_CODE"] = diag_df["ICD9_CODE"].str.strip()
    diag_df = diag_df[diag_df["ICD9_CODE"].isin(lookup)]
    diag_df = diag_df.sort_values(["HADM_ID", "SEQ_NUM"])

    admissions: Dict[int, List[Dict[str, Any]]] = {}
    for r in diag_df.itertuples(index=False):
        hadm = int(r.HADM_ID)
        short, long_ = lookup[r.ICD9_CODE]
        admissions.setdefault(hadm, []).append({
            "seq_num":     int(r.SEQ_NUM),
            "icd9_code":   r.ICD9_CODE,
            "short_title": short,
            "long_title":  long_,
        })
    return admissions


def _format_gold_list(icds: List[Dict[str, Any]]) -> str:
    """Render the full admission ICD list for the judge prompt."""
    if not icds:
        return "BILLED ICD-9 DIAGNOSES: none available"
    lines = ["BILLED ICD-9 DIAGNOSES (all codes for this admission):"]
    for d in icds:
        marker = "  [PRIMARY]" if d["seq_num"] == 1 else ""
        lines.append(
            f"  SEQ_NUM={d['seq_num']:<3} {d['icd9_code']:<8} {d['long_title']}{marker}"
        )
    return "\n".join(lines)


def _build_user_prompt(rec: Dict, icds: List[Dict]) -> str:
    ann = {
        "diagnosis":          rec.get("model_diagnosis"),
        "confidence_level":   rec.get("model_confidence"),
    }
    return "\n\n".join([
        f"MEDICAL NOTE:\n{rec.get('note_text', '')}",
        f"ANNOTATOR OUTPUT:\n{json.dumps(ann, indent=2)}",
        _format_gold_list(icds),
    ])


def _parse_judgment(text: Optional[str]) -> Dict:
    """Parse judge output. Response schema + enough tokens makes clean JSON the
    norm; fallback handles truncation or rare malformed cases by extracting at
    least the `correct` verdict via regex."""
    if not text:
        return {"correct": False, "failure_mode": "ambiguous_case",
                "comment": "empty response", "confidence": 0}
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    data = None
    for candidate in (text, cleaned):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            pass
    # raw_decode from first brace — tolerates trailing garbage after valid JSON
    if data is None:
        start = cleaned.find("{")
        if start != -1:
            try:
                data, _ = json.JSONDecoder().raw_decode(cleaned, start)
            except json.JSONDecodeError:
                pass
    # Regex fallback for truncated output — extract what we can
    if data is None:
        m_correct = re.search(r'"correct"\s*:\s*(true|false)', cleaned, re.I)
        if m_correct:
            m_fm = re.search(r'"failure_mode"\s*:\s*"([^"]+)"', cleaned)
            m_conf = re.search(r'"confidence"\s*:\s*(\d+)', cleaned)
            data = {
                "correct": m_correct.group(1).lower() == "true",
                "failure_mode": m_fm.group(1) if m_fm else None,
                "comment": "(recovered from truncated judge output)",
                "confidence": int(m_conf.group(1)) if m_conf else 50,
            }
    if data is None:
        return {"correct": False, "failure_mode": "ambiguous_case",
                "comment": f"unparseable: {cleaned[:100]!r}", "confidence": 0}

    fm = data.get("failure_mode")
    if fm == "none" or (data.get("correct") is True and fm in (None, "none")):
        data["failure_mode"] = None
    elif fm not in FAILURE_MODES + [None]:
        data["failure_mode"] = "ambiguous_case"
    return data


def _judge_one(client, model: str, rec: Dict, icds: List[Dict]) -> Dict:
    user_prompt = _build_user_prompt(rec, icds)
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_LENIENT_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=_JUDGMENT_SCHEMA,
                ),
            )
            return _parse_judgment(resp.text)
        except Exception as e:
            err = str(e)
            transient = "503" in err or "429" in err or "UNAVAILABLE" in err
            if transient and attempt < 5:
                suggested = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)s", err)
                base = float(suggested.group(1)) if suggested else 10 * (2 ** attempt)
                time.sleep(base + random.uniform(1, 5))
            else:
                raise
    return {"correct": False, "failure_mode": "ambiguous_case",
            "comment": "judge exhausted retries", "confidence": 0}


def rescore_file(
    path: str,
    admissions: Dict[int, List[Dict]],
    client,
    judge_model: str,
    max_workers: int = 8,
) -> Dict:
    with open(path) as f:
        records = json.load(f)

    # Build tasks, one per record
    lenient_results: Dict[int, Dict] = {}

    def _task(i_rec):
        i, rec = i_rec
        hadm = int(rec["hadm_id"])
        icds = admissions.get(hadm, [])
        if not icds:
            return i, {"correct": None, "failure_mode": "ambiguous_case",
                       "comment": "no billed ICDs found for HADM_ID",
                       "confidence": 0, "admission_icd_count": 0}
        verdict = _judge_one(client, judge_model, rec, icds)
        verdict["admission_icd_count"] = len(icds)
        return i, verdict

    print(f"  re-scoring {len(records):,} cases (parallel workers={max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_task, (i, r)): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futs):
            i, verdict = fut.result()
            lenient_results[i] = verdict
            done += 1
            if done % 25 == 0 or done == len(records):
                print(f"    {done}/{len(records)} done", flush=True)

    # Merge strict + lenient side-by-side
    out_records = []
    for i, rec in enumerate(records):
        lv = lenient_results[i]
        merged = {
            **rec,
            "strict_correct":       rec.get("human_correct"),
            "strict_failure_mode":  rec.get("human_failure_mode"),
            "strict_comment":       rec.get("human_comment"),
            "lenient_correct":      lv.get("correct"),
            "lenient_failure_mode": lv.get("failure_mode"),
            "lenient_comment":      lv.get("comment"),
            "admission_icd_count":  lv.get("admission_icd_count", 0),
        }
        out_records.append(merged)

    # Summary
    strict_correct  = sum(1 for r in out_records if r["strict_correct"] is True)
    lenient_correct = sum(1 for r in out_records if r["lenient_correct"] is True)
    flipped = sum(
        1 for r in out_records
        if r["strict_correct"] is False and r["lenient_correct"] is True
    )

    summary = {
        "n":                  len(out_records),
        "strict_accuracy":    round(strict_correct / len(out_records), 4) if out_records else 0,
        "lenient_accuracy":   round(lenient_correct / len(out_records), 4) if out_records else 0,
        "flipped_to_correct": flipped,
        "delta":              round((lenient_correct - strict_correct) / len(out_records), 4) if out_records else 0,
    }

    # Write output
    out_path = path.replace(".json", "_lenient.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "cases": out_records}, f, indent=2)

    print(
        f"  → strict={summary['strict_accuracy']:.3f}  "
        f"lenient={summary['lenient_accuracy']:.3f}  "
        f"flipped={summary['flipped_to_correct']} "
        f"(delta=+{summary['delta']*100:.1f}pp)"
    )
    print(f"  → wrote {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Single experiment JSON file.")
    parser.add_argument("--source-glob", help="Glob of experiment JSON files.")
    parser.add_argument("--judge-model", default="gemini-2.5-flash",
                        help="Gemini judge model for lenient re-scoring.")
    parser.add_argument("--secrets", default="./secrets.toml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip API calls; recompute summary from an existing *_lenient.json.")
    args = parser.parse_args()

    if not (args.source or args.source_glob):
        parser.error("Provide --source or --source-glob.")

    paths = [args.source] if args.source else sorted(glob.glob(args.source_glob))
    paths = [p for p in paths if not p.endswith("_lenient.json")]
    if not paths:
        parser.error("No input files matched (after excluding *_lenient.json).")

    if args.summary_only:
        print("Summary-only mode — reading existing *_lenient.json outputs.")
        for p in paths:
            lp = p.replace(".json", "_lenient.json")
            if not os.path.exists(lp):
                print(f"  {p} → no {lp} yet, skipping")
                continue
            with open(lp) as f:
                d = json.load(f)
            s = d["summary"]
            print(
                f"  {os.path.basename(p):55s} strict={s['strict_accuracy']:.3f} "
                f"lenient={s['lenient_accuracy']:.3f} delta=+{s['delta']*100:.1f}pp"
            )
        return

    print("Loading admission ICD lookup...", flush=True)
    admissions = _load_admission_icds()
    print(f"  {len(admissions):,} admissions indexed")

    secrets = toml.load(args.secrets)
    client = genai.Client(api_key=secrets["api"]["key"])

    all_summaries = []
    for p in paths:
        print(f"\n=== {p} ===")
        s = rescore_file(p, admissions, client, args.judge_model, args.workers)
        all_summaries.append({"file": os.path.basename(p), **s})

    if len(all_summaries) > 1:
        print("\n" + "=" * 72)
        print(f"{'file':55s} {'strict':>7} {'lenient':>8} {'Δpp':>6}")
        print("-" * 72)
        for s in all_summaries:
            print(
                f"{s['file']:55s} {s['strict_accuracy']:>7.3f} "
                f"{s['lenient_accuracy']:>8.3f} {s['delta']*100:>6.1f}"
            )


if __name__ == "__main__":
    main()
