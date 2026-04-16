"""
One-time ETL for the MIMIC-III source tables in data/new/.

Produces data/new/processed/notes_with_gold.parquet: one row per admission,
discharge summary text joined with the primary ICD-9 diagnosis (SEQ_NUM=1)
and its human-readable titles. This parquet is what the pipeline reads at
runtime.

Usage:
    python scripts/preprocess_new_data.py            # skip if output exists
    python scripts/preprocess_new_data.py --force    # rebuild
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


DATA_DIR = "data/new"
OUT_DIR = os.path.join(DATA_DIR, "processed")
OUT_PATH = os.path.join(OUT_DIR, "notes_with_gold.parquet")

NOTES_CSV = os.path.join(DATA_DIR, "NOTEEVENTS.csv")
DIAG_CSV = os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv")
DICT_CSV = os.path.join(DATA_DIR, "D_ICD_DIAGNOSES.csv")

CHUNK_SIZE = 200_000


def icd9_chapter(code: str) -> str:
    """Return the ICD-9-CM chapter label for a code.

    Covers the 17 numeric chapters plus V- and E-codes. Unparseable codes
    fall back to "Unknown" (shouldn't happen after the preprocess filters).
    """
    if not code:
        return "Unknown"
    c = str(code).strip().upper()
    if c.startswith("V"):
        return "V: Supplementary (V-codes)"
    if c.startswith("E"):
        return "E: External causes (E-codes)"
    # First three characters are the numeric category (codes may include a
    # decimal, but the raw DIAGNOSES_ICD.csv stores them zero-padded).
    head = c[:3]
    try:
        n = int(head)
    except ValueError:
        return "Unknown"
    if   1   <= n <= 139: return "01: Infectious/parasitic"
    elif 140 <= n <= 239: return "02: Neoplasms"
    elif 240 <= n <= 279: return "03: Endocrine/metabolic"
    elif 280 <= n <= 289: return "04: Blood/immune"
    elif 290 <= n <= 319: return "05: Mental"
    elif 320 <= n <= 389: return "06: Nervous/sense"
    elif 390 <= n <= 459: return "07: Circulatory"
    elif 460 <= n <= 519: return "08: Respiratory"
    elif 520 <= n <= 579: return "09: Digestive"
    elif 580 <= n <= 629: return "10: Genitourinary"
    elif 630 <= n <= 679: return "11: Pregnancy/childbirth"
    elif 680 <= n <= 709: return "12: Skin/subcutaneous"
    elif 710 <= n <= 739: return "13: Musculoskeletal"
    elif 740 <= n <= 759: return "14: Congenital"
    elif 760 <= n <= 779: return "15: Perinatal"
    elif 780 <= n <= 799: return "16: Symptoms/ill-defined"
    elif 800 <= n <= 999: return "17: Injury/poisoning"
    return "Unknown"


def load_icd_dictionary() -> dict:
    print(f"[1/4] Loading ICD-9 dictionary from {DICT_CSV} ...", flush=True)
    df = pd.read_csv(DICT_CSV, dtype=str, keep_default_na=False)
    # Normalize to strings, strip whitespace
    df["ICD9_CODE"] = df["ICD9_CODE"].astype(str).str.strip()
    lookup = {
        row.ICD9_CODE: (row.SHORT_TITLE.strip(), row.LONG_TITLE.strip())
        for row in df.itertuples(index=False)
    }
    print(f"      {len(lookup):,} ICD-9 codes")
    return lookup


def load_primary_dx(icd_lookup: dict) -> pd.DataFrame:
    print(f"[2/4] Loading primary diagnoses from {DIAG_CSV} ...", flush=True)
    df = pd.read_csv(DIAG_CSV, dtype={"ICD9_CODE": str})
    n0 = len(df)

    # Primary diagnosis = SEQ_NUM == 1
    df = df[df["SEQ_NUM"] == 1].copy()
    n1 = len(df)

    # Deduplicate on HADM_ID (shouldn't happen — but guard against bad data)
    df = df.drop_duplicates(subset=["HADM_ID"], keep="first")
    n2 = len(df)

    df["ICD9_CODE"] = df["ICD9_CODE"].astype(str).str.strip()
    df = df[df["ICD9_CODE"].isin(icd_lookup)]
    n3 = len(df)

    df["short_title"] = df["ICD9_CODE"].map(lambda c: icd_lookup[c][0])
    df["long_title"]  = df["ICD9_CODE"].map(lambda c: icd_lookup[c][1])
    df = df.rename(columns={"ICD9_CODE": "icd9_code"})
    df = df[["SUBJECT_ID", "HADM_ID", "icd9_code", "short_title", "long_title"]]

    print(
        f"      {n0:,} rows → {n1:,} primary (SEQ_NUM=1) → {n2:,} unique HADM_ID "
        f"→ {n3:,} with valid ICD-9 dictionary entry"
    )
    return df


def stream_discharge_notes(primary_dx: pd.DataFrame) -> pd.DataFrame:
    """Stream NOTEEVENTS.csv in chunks, keep only discharge summaries whose
    admission has a primary ICD-9 diagnosis."""
    if not os.path.exists(NOTES_CSV):
        print(f"ERROR: {NOTES_CSV} not found", file=sys.stderr)
        sys.exit(1)

    size_gb = os.path.getsize(NOTES_CSV) / (1024 ** 3)
    print(
        f"[3/4] Streaming {NOTES_CSV} ({size_gb:.1f} GB) in chunks of "
        f"{CHUNK_SIZE:,} rows ...",
        flush=True,
    )

    # Fast HADM_ID filter set — skip rows outside our join set before the
    # heavier string ops. Enables early dropout on each chunk.
    valid_hadm = set(primary_dx["HADM_ID"].dropna().astype("int64"))

    use_cols = [
        "ROW_ID", "SUBJECT_ID", "HADM_ID",
        "CATEGORY", "CHARTDATE", "ISERROR", "TEXT",
    ]
    dtype = {
        "ROW_ID": "Int64",
        "SUBJECT_ID": "Int64",
        "HADM_ID": "Int64",
        "CATEGORY": "string",
        "CHARTDATE": "string",
        "ISERROR": "string",   # usually empty; kept as str so NaN detection is reliable
        "TEXT": "string",
    }

    total_in = 0
    total_after_discharge = 0
    total_after_iserror = 0
    total_after_hadm = 0
    kept_chunks = []

    for chunk_idx, chunk in enumerate(pd.read_csv(
        NOTES_CSV,
        usecols=use_cols,
        dtype=dtype,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ), start=1):
        total_in += len(chunk)

        cat = chunk["CATEGORY"].astype("string").str.strip().str.lower()
        chunk = chunk[cat == "discharge summary"]
        total_after_discharge += len(chunk)
        if chunk.empty:
            print(f"      chunk {chunk_idx}: 0 discharge rows", flush=True)
            continue

        # Drop rows flagged as errors. ISERROR is nullable; empty/NaN = not-error.
        err = chunk["ISERROR"].astype("string").fillna("").str.strip()
        chunk = chunk[err == ""]
        total_after_iserror += len(chunk)

        chunk = chunk.dropna(subset=["HADM_ID", "TEXT"])
        chunk = chunk[chunk["HADM_ID"].astype("int64").isin(valid_hadm)]
        total_after_hadm += len(chunk)

        if not chunk.empty:
            kept_chunks.append(chunk.copy())

        print(
            f"      chunk {chunk_idx}: scanned={total_in:,} "
            f"discharge={total_after_discharge:,} "
            f"not_error={total_after_iserror:,} "
            f"in_hadm_set={total_after_hadm:,}",
            flush=True,
        )

    if not kept_chunks:
        print("ERROR: no discharge summaries matched. Check input files.", file=sys.stderr)
        sys.exit(1)

    notes = pd.concat(kept_chunks, ignore_index=True)
    print(f"      total kept discharge rows: {len(notes):,}")

    # Deduplicate multiple discharge summaries per admission: keep the LATEST
    # CHARTDATE (addenda / final corrections); ROW_ID as tiebreaker.
    # CHARTDATE is "YYYY-MM-DD" / "YYYY-MM-DD HH:MM:SS" — lexicographic sort
    # preserves chronological order, avoiding slow pd.to_datetime.
    notes = notes.sort_values(
        ["HADM_ID", "CHARTDATE", "ROW_ID"],
        ascending=[True, False, False],
        na_position="last",
    )
    n_before_dedup = len(notes)
    notes = notes.drop_duplicates(subset=["HADM_ID"], keep="first")
    print(
        f"      deduped multiple discharge summaries per admission: "
        f"{n_before_dedup:,} → {len(notes):,}"
    )

    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the output parquet exists.")
    args = parser.parse_args()

    if os.path.exists(OUT_PATH) and not args.force:
        print(f"{OUT_PATH} already exists. Re-run with --force to rebuild.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    icd_lookup = load_icd_dictionary()
    primary_dx = load_primary_dx(icd_lookup)
    notes = stream_discharge_notes(primary_dx)

    print("[4/4] Joining notes with primary diagnoses ...", flush=True)
    merged = notes.merge(primary_dx, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    # Final sanity: long_title and TEXT both non-null
    merged = merged[merged["long_title"].notna() & merged["TEXT"].notna()]
    merged = merged[merged["long_title"].astype(str).str.len() > 0]

    merged["icd9_chapter"] = merged["icd9_code"].map(icd9_chapter)

    final_cols = [
        "ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "TEXT",
        "icd9_code", "short_title", "long_title", "icd9_chapter",
    ]
    out = merged[final_cols].reset_index(drop=True)

    out.to_parquet(OUT_PATH, index=False)
    print(f"      wrote {len(out):,} rows to {OUT_PATH}")

    print()
    print("Summary:")
    print(f"  total admissions with discharge summary + primary ICD: {len(out):,}")
    print(f"  unique patients:   {out['SUBJECT_ID'].nunique():,}")
    print(f"  unique admissions: {out['HADM_ID'].nunique():,}")
    print(f"  unique ICD-9 codes: {out['icd9_code'].nunique():,}")
    print("  top-10 chapters:")
    for chap, cnt in out["icd9_chapter"].value_counts().head(10).items():
        print(f"    {chap:<34} {cnt:>7,}")


if __name__ == "__main__":
    main()
