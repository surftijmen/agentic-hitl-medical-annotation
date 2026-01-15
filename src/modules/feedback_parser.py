from typing import Dict


class FeedbackParser:
    def parse(self, human_feedback: Dict) -> Dict:
        """
        Expected input example:
        {
            "correct": False,
            "comment": "Model hallucinated diagnosis"
        }
        """
        error_type = "none"

        comment = human_feedback.get("comment", "").lower()

        if not human_feedback.get("correct"):
            if "hallucination" in comment:
                error_type = "hallucination"
            elif "miss" in comment:
                error_type = "missed_disease"
            else:
                error_type = "false_positive"

        return {
            "error_type": error_type,
            "confidence_flag": human_feedback.get("confidence", 0) < 50,
        }
