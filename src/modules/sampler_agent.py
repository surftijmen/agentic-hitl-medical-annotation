"""Data sampler for the HITL pipeline.

Reads the preprocessed parquet produced by scripts/preprocess_new_data.py
(one discharge summary per admission with its primary ICD-9 diagnosis) and
returns stratified batches where the gold label is the ICD-9 code + titles.
"""

import pandas as pd


def _icd9_chapter(code: str) -> str:
    """ICD-9-CM chapter bucket — fallback for rows that lack a precomputed one."""
    if not code:
        return "Unknown"
    c = str(code).strip().upper()
    if c.startswith("V"):
        return "V: Supplementary (V-codes)"
    if c.startswith("E"):
        return "E: External causes (E-codes)"
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


DEFAULT_PROCESSED_PATH = "data/new/processed/notes_with_gold.parquet"
HOLDOUT_SIZE = 200
# Fixed seed so the 200-row holdout is the same across runs — lets accuracy
# numbers be compared iteration-to-iteration under the new gold.
DEFAULT_SEED = 20260414

# ICD-9 chapters whose primary-code admissions are driven by hospital billing
# conventions rather than clinical diagnosis. Excluding these scopes the
# sampler to the clinical-annotation task under study:
#
#   V-codes : "Supplementary Classification of Factors Influencing Health
#             Status and Contact with Health Services" — e.g. V3000 "Single
#             liveborn, born in hospital" — describes encounter reason, not
#             disease.
#   ch. 15  : Perinatal codes (760-779) — administrative newborn categories
#             (birth-weight brackets, neonatal jaundice of prematurity).
#   ch. 16  : Symptoms, signs, ill-defined conditions (780-799) — used as
#             primary when no disease was confirmed (e.g. "Nausea with
#             vomiting" even when underlying cancer is billed as secondary).
#
# These cases test conformance with billing conventions rather than clinical
# annotation ability; excluding them is standard in clinical-NLP benchmarks
# and is documented in the Methods section.
ADMIN_CHAPTERS_DEFAULT = frozenset({
    "V: Supplementary (V-codes)",
    "15: Perinatal",
    "16: Symptoms/ill-defined",
})


class DataSampler:
    def __init__(
        self,
        processed_path: str = DEFAULT_PROCESSED_PATH,
        seed: int = DEFAULT_SEED,
        exclude_admin_chapters: bool = True,
    ):
        df = pd.read_parquet(processed_path)

        # Minimal sanity guard (preprocess already did the real filtering)
        df = df[df["TEXT"].notna() & df["long_title"].notna()].copy()
        df = df[df["long_title"].astype(str).str.len() > 0]

        if "icd9_chapter" not in df.columns:
            df["icd9_chapter"] = df["icd9_code"].map(_icd9_chapter)

        # Scope to clinical-diagnosis chapters only (V-codes, Perinatal,
        # Symptoms/ill-defined are administrative billing categories, not
        # clinical annotation targets). See ADMIN_CHAPTERS_DEFAULT.
        n_before = len(df)
        if exclude_admin_chapters:
            df = df[~df["icd9_chapter"].isin(ADMIN_CHAPTERS_DEFAULT)].copy()
            excluded = n_before - len(df)
            print(
                f"  [DataSampler] excluded {excluded:,} administrative-chapter rows "
                f"({', '.join(sorted(ADMIN_CHAPTERS_DEFAULT))})"
            )

        # Build the gold-label dict column once so downstream code reads row["gold"].
        df["gold"] = df[["icd9_code", "short_title", "long_title"]].to_dict(orient="records")

        print(
            f"  [DataSampler] {len(df):,} usable rows from {processed_path} "
            f"({df['icd9_chapter'].nunique()} ICD-9 chapters, "
            f"{df['icd9_code'].nunique():,} unique codes)"
        )

        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Reserve the last HOLDOUT_SIZE rows as a fixed held-out test set.
        self.holdout_df = shuffled.iloc[-HOLDOUT_SIZE:].reset_index(drop=True)
        self.df        = shuffled.iloc[:-HOLDOUT_SIZE].reset_index(drop=True)

        self._rng = pd.core.common.random_state(seed)

    def sample_batch(self, n: int) -> pd.DataFrame:
        """Stratified random sample across ICD-9 chapters.

        When n >= number of chapters, draws n // num_chapters per chapter so
        every chapter is represented. When n is smaller, falls back to a
        plain random sample of n rows so the requested size is respected.
        """
        chapters = self.df["icd9_chapter"].unique()

        if n < len(chapters):
            return self.df.sample(n=n, random_state=self._rng).reset_index(drop=True)

        per_class = n // len(chapters)
        remainder = n - per_class * len(chapters)

        frames = []
        for chapter in chapters:
            pool = self.df[self.df["icd9_chapter"] == chapter]
            k = min(per_class, len(pool))
            if k > 0:
                frames.append(pool.sample(n=k, random_state=self._rng))

        sampled_ids = pd.concat(frames).index if frames else pd.Index([])
        leftover = self.df.drop(index=sampled_ids)
        if remainder > 0 and len(leftover) > 0:
            frames.append(leftover.sample(n=min(remainder, len(leftover)), random_state=self._rng))

        return pd.concat(frames).sample(frac=1, random_state=self._rng).reset_index(drop=True)

    def sample_holdout(self) -> pd.DataFrame:
        """Return the fixed held-out test set (same rows every call)."""
        return self.holdout_df
