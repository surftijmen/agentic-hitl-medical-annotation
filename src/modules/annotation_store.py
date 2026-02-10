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
            "human_feedback": human_feedback.get("reason"),
            "human_confidence": human_feedback.get("confidence"),
            "error_type": human_feedback.get("error_type"),
            "prompt_version": prompt_version,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def build_retrieval_text(self, note: str, final_dx: str) -> str:
        note_short = note[:400]
        return f"""
        Patient case diagnosed with {final_dx}.
        Key context: {note_short}
        """.strip()

    def to_rag_record(self, record):

        if record["human_correct"] is False:
            return None

        final_dx = record["model_diagnosis"]

        retrieval_text = self.build_retrieval_text(
            record["note_text"],
            final_dx
        )

        return {
            "retrieval_text": retrieval_text,
            "final_diagnosis": final_dx,
            "model_diagnosis": record["model_diagnosis"],
            "correct": record["human_correct"],
            "failure_mode": record["error_type"],
            "doctor_confidence": record["human_confidence"],
            "model_confidence": record["model_confidence"],
            "note_text": record["note_text"],
        }