from typing import Dict, List
import pandas as pd
from datetime import datetime


class AnnotationStore:
    def __init__(self):
        self.records: List[Dict] = []

    def add(
        self,
        subject_id: int,
        hadm_id: int,
        note_text: str,
        annotation: Dict,
        gold: str,
        human_feedback: Dict,
        prompt_version: str,
    ):
        self.records.append({
            "timestamp": datetime.utcnow().isoformat(),
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "note_text": note_text,
            "model_diagnosis": annotation.get("diagnosis"),
            "model_confidence": annotation.get("confidence_level"),
            "gold_reference": gold,
            "human_correct": human_feedback.get("correct"),
            "human_confidence": human_feedback.get("confidence"),
            "error_type": human_feedback.get("error_type"),
            "prompt_version": prompt_version,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
