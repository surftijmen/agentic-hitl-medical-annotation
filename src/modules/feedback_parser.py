"""Converts reviewer dicts into structured signals with routing decisions and
a low-confidence flag. The FAILURE_MODE_ROUTING map is reporting-only —
PromptAgent patches every non-ambiguous failure regardless of route.
"""
from typing import Dict, Optional


# Maps each failure mode to which technique should correct it
FAILURE_MODE_ROUTING = {
    "missed_entity":                "rag",             # show model similar correctly-extracted cases
    "terminology_gap":              "rag",             # show model correct terminology in context
    "hallucination":                "prompt_patch",    # add negative instruction
    "unsupported_inference":        "prompt_patch",    # add reasoning constraint
    "symptom_diagnosis_confusion":  "prompt_patch",    # clarify boundary instruction
    "ambiguous_case":               "flag",            # neither — flag for data quality review
}


class FeedbackParser:
    def parse(self, human_feedback: Dict) -> Dict:
        comment = (human_feedback.get("comment") or "").lower()
        correct = human_feedback.get("correct", None)
        confidence = human_feedback.get("confidence", None)

        # Use failure_mode directly from the UI if present
        failure_mode: Optional[str] = human_feedback.get("failure_mode")

        # Fall back to comment-based inference if not set by UI
        if correct is False and not failure_mode:
            if any(k in comment for k in ["infer", "assume", "hallucinat", "guess"]):
                failure_mode = "unsupported_inference"
            elif any(k in comment for k in ["symptom", "complaint", "finding"]):
                failure_mode = "symptom_diagnosis_confusion"
            elif any(k in comment for k in ["miss", "missed", "not extracted"]):
                failure_mode = "missed_entity"
            else:
                failure_mode = "unclassified_error"

        # Route to the appropriate correction technique
        correction_technique = FAILURE_MODE_ROUTING.get(failure_mode) if failure_mode else None

        return {
            "outcome": "correct" if correct is True else "incorrect",
            "failure_mode": failure_mode,
            "correction_technique": correction_technique,   # "rag", "prompt_patch", or "flag"
            "human_rationale": human_feedback.get("comment", ""),
            "confidence": confidence,
            "low_confidence_flag": confidence is not None and confidence < 50,
        }