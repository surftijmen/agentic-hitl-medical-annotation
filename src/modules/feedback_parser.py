from typing import Dict, Optional


class FeedbackParser:
    def __init__(self):
        pass

    def parse(self, human_feedback: Dict) -> Dict:
        """
        Expected input example:
        {
            "correct": false,
            "comment": "Model inferred pneumonia from prognosis discussion",
            "confidence": 80
        }
        """

        comment = (human_feedback.get("comment") or "").lower()
        correct = human_feedback.get("correct", None)
        confidence = human_feedback.get("confidence", None)

        failure_mode: Optional[str] = None


        return {
            "outcome": "correct" if correct else "incorrect",
            "failure_mode": failure_mode,
            "human_rationale": human_feedback.get("comment", ""),
            "error_type": "TEST",
            "confidence_flag": confidence,
            "low_confidence_flag": confidence is not None and confidence < 50
        }
