from typing import Dict, Optional


class FeedbackParser:
    def parse(self, human_feedback: Dict) -> Dict:
        comment = (human_feedback.get("comment") or "").lower()
        correct = human_feedback.get("correct", None)
        confidence = human_feedback.get("confidence", None)

        failure_mode: Optional[str] = None
        error_locus: Optional[str] = None

        if correct is False:
            if any(k in comment for k in ["infer", "assume", "hallucinat", "guess"]):
                failure_mode = "unsupported_inference"
                error_locus = "reasoning"

            elif any(k in comment for k in ["symptom", "complaint", "finding"]):
                failure_mode = "symptom_diagnosis_confusion"
                error_locus = "reasoning"

            elif any(k in comment for k in ["miss", "missed", "not extracted"]):
                failure_mode = "missed_entity"
                error_locus = "extraction"

            else:
                failure_mode = "unclassified_error"
                error_locus = "unknown"

        return {
            "outcome": "correct" if correct else "incorrect",
            "failure_mode": failure_mode,
            "error_locus": error_locus,
            "error_type": "TEST",
            "human_rationale": human_feedback.get("comment", ""),
            "confidence": confidence,
            "low_confidence_flag": confidence is not None and confidence < 50,
        }
