"""Per-run record book of (chart, annotation, review) tuples. Source of the
experiment JSON dump under logs/experiments/, and constructs RAG records
(validated + corrected cases) for the retrieval store.
"""
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
        Build a RAG record for a reviewed annotation.

        Both validated (correct) and corrected (incorrect-but-reviewed) cases
        enter the store as `note → correct diagnosis` pairs:
          • validated  : retrieval target = the model's diagnosis (which was right)
          • corrected  : retrieval target = the gold long_title    (the correction)
        Ambiguous cases are skipped because the gold itself is unreliable.

        Storing corrected cases removes the selection-bias loop of the
        validated-only design, where hard charts the model got wrong were
        never represented in the store.
        """
        failure_mode = record.get("human_failure_mode")
        is_correct = record.get("human_correct") is True

        if is_correct:
            final_dx = record["model_diagnosis"]
            source = "validated"
        else:
            # Skip ambiguous cases — gold may itself be wrong/vague.
            if failure_mode == "ambiguous_case":
                return None
            gold = record.get("gold_reference")
            if isinstance(gold, dict):
                final_dx = gold.get("long_title") or gold.get("short_title")
            elif isinstance(gold, str):
                final_dx = gold
            else:
                final_dx = None
            source = "corrected"

        if not final_dx or str(final_dx).lower() == "none":
            return None

        retrieval_text = self.build_retrieval_text(record["note_text"], final_dx)

        return {
            "retrieval_text": retrieval_text,
            "subject_id": record["subject_id"],
            "hadm_id": record["hadm_id"],
            "final_diagnosis": final_dx,
            "model_diagnosis": record["model_diagnosis"],
            "correct": record["human_correct"],
            "source": source,
            "failure_mode": failure_mode,
            "doctor_confidence": record.get("human_confidence"),
            "model_confidence": record.get("model_confidence"),
            "note_text": record["note_text"],
        }