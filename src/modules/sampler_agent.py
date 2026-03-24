import pandas as pd


# Maps substrings of the MIMIC-III admission DIAGNOSIS field to the
# allowed_diagnoses labels used by the annotator prompt.
DIAGNOSIS_LABEL_MAP = {
    "PNEUMONIA": "Pneumonia",
    "SEPSIS": "Sepsis",
    "CONGESTIVE HEART FAILURE": "CHF",
    "CORONARY ARTERY DISEASE": "CAD",
    "HYPERTENSION": "Hypertension",
    "DIABETES": "Diabetes",
    "CHRONIC OBSTRUCTIVE PULMONARY DISEASE": "COPD",
    "COPD": "COPD",
    "ATRIAL FIBRILLATION": "Atrial Fibrillation",
    " AFIB": "Atrial Fibrillation",
    "CHRONIC KIDNEY DISEASE": "CKD",
    "ASTHMA": "Asthma",
    "CANCER": "Cancer",
    "CARCINOMA": "Cancer",
    "MALIGNANCY": "Cancer",
    "TUMOR": "Cancer",
    "NEOPLASM": "Cancer",
    "STROKE": "Stroke",
    "CEREBROVASCULAR ACCIDENT": "Stroke",
    "CVA": "Stroke",
    "MYOCARDIAL INFARCTION": "MI",
    " MI ": "MI",
    "STEMI": "MI",
    "NSTEMI": "MI",
    "TUBERCULOSIS": "Tuberculosis",
    "ANEMIA": "Anemia",
    "CIRRHOSIS": "Cirrhosis",
    "HIV": "HIV",
    "AIDS": "HIV",
    "OBESITY": "Obesity",
    "URINARY TRACT INFECTION": "UTI",
    " UTI": "UTI",
    "DEEP VEIN THROMBOSIS": "DVT",
    " DVT": "DVT",
    "PULMONARY EMBOLISM": "PE",
}


def _map_diagnosis(raw: str) -> str | None:
    """Return the normalized label only if the raw string unambiguously maps to
    exactly one allowed diagnosis.  Multi-diagnosis strings like 'CVA; CHF PNEUMONIA'
    that match multiple distinct labels are excluded (return None) because the sampler
    cannot reliably determine which is the primary admission diagnosis.
    """
    upper = str(raw).upper()
    matched_labels = {
        label
        for key, label in DIAGNOSIS_LABEL_MAP.items()
        if key in upper
    }
    if len(matched_labels) == 1:
        return next(iter(matched_labels))
    return None


class DataSampler:
    def __init__(
        self,
        notes_path: str,
        adm_path: str,
        seed: int = 122,
    ):
        notes_df = pd.read_parquet(
            notes_path,
            columns=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        )

        adm_df = pd.read_parquet(
            adm_path,
            columns=["SUBJECT_ID", "HADM_ID", "DIAGNOSIS"],
        )

        merged = notes_df.merge(adm_df, on=["SUBJECT_ID", "HADM_ID"], how="inner")

        # Filter 1: only discharge summaries — these reliably contain confirmed diagnoses.
        # Nursing notes, radiology reports, etc. often don't state the primary diagnosis.
        discharge_mask = merged["CATEGORY"].str.lower().str.strip() == "discharge summary"

        # Filter 2: drop "rule out" cases — DIAGNOSIS values starting with "R/O" mean
        # the condition was NOT confirmed. Keeping them produces unachievable gold labels
        # (e.g. "R/O LYMPHOMA" → gold="Cancer" but the note says rule-out lymphoma).
        ruleout_mask = ~merged["DIAGNOSIS"].str.upper().str.startswith("R/O")

        # Filter 3: only rows where the admission DIAGNOSIS maps to one of the 21
        # allowed labels. Other rows have unanswerable ground truth (e.g. "NEWBORN",
        # "LOWER GI BLEED") that will always score as incorrect regardless of AI output.
        merged["label"] = merged["DIAGNOSIS"].apply(_map_diagnosis)
        label_mask = merged["label"].notna()

        filtered = merged[discharge_mask & ruleout_mask & label_mask].copy()

        print(
            f"  [DataSampler] {len(filtered):,} usable rows after filtering "
            f"(discharge summaries with mappable labels) "
            f"from {len(merged):,} total merged rows."
        )

        self.df = (
            filtered
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        self.cursor = 0

    def sample_batch(self, n: int) -> pd.DataFrame:
        batch = self.df.iloc[self.cursor : self.cursor + n]
        self.cursor += n
        return batch
