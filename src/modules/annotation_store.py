from typing import Any, Dict, List, Optional, Union
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
        gold: Union[Dict[str, Any], str, None],
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
            "human_failure_mode": human_feedback.get("failure_mode"),  # from UI directly
            "human_comment": human_feedback.get("comment"),
            "human_confidence": human_feedback.get("confidence"),
            "model_confidence_raw": human_feedback.get("model_confidence"),
            "prompt_version": prompt_version,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def build_retrieval_text(self, note: str, final_dx: str) -> str:
        note_short = note[:400]
        return f"Patient case diagnosed with {final_dx}.\nKey context: {note_short}".strip()

    def to_rag_record(self, record: Dict) -> Optional[Dict]:
        """
        Returns a RAG record for validated (correct) cases only.
        Includes subject_id and hadm_id so RAG deduplication works.
        """
        if record.get("human_correct") is not True:
            return None

        final_dx = record["model_diagnosis"]
        if not final_dx or final_dx.lower() == "none":
            return None

        retrieval_text = self.build_retrieval_text(record["note_text"], final_dx)

        return {
            "retrieval_text": retrieval_text,
            "subject_id": record["subject_id"],    # fixes case-unknown-unknown
            "hadm_id": record["hadm_id"],          # fixes case-unknown-unknown
            "final_diagnosis": final_dx,
            "model_diagnosis": record["model_diagnosis"],
            "correct": record["human_correct"],
            "failure_mode": record.get("human_failure_mode"),
            "doctor_confidence": record.get("human_confidence"),
            "model_confidence": record.get("model_confidence"),
            "note_text": record["note_text"],
        }