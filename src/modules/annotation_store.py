import pandas as pd
from typing import Dict, Optional


class AnnotationStore:
    def __init__(self):
        self.records = []

    def add(
        self,
        subject_id: int,
        hadm_id: int,
        annotation: Dict,
        gold: Optional[str] = None,
    ):
        self.records.append({
            "SUBJECT_ID": subject_id,
            "HADM_ID": hadm_id,
            "annotation": annotation,
            "gold_label": gold,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
