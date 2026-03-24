from typing import List, Dict
from collections import Counter


# All failure modes the human reviewer can select
KNOWN_FAILURE_MODES = [
    "hallucination",
    "missed_entity",
    "terminology_gap",
    "unsupported_inference",
    "symptom_diagnosis_confusion",
    "ambiguous_case",
]


class PerformanceEvaluator:
    def evaluate(self, signals: List[Dict]) -> Dict:
        total = len(signals)
        if total == 0:
            return {
                "precision": 0.0,
                "accuracy": 0.0,
                "hallucinations": 0,
                "failure_mode_breakdown": {},
                "confidence_calibration": {},
                "alerts": [],
            }

        correct = sum(s.get("outcome") == "correct" for s in signals)
        incorrect = sum(s.get("outcome") == "incorrect" for s in signals)

        hallucinations = sum(
            s.get("failure_mode") == "hallucination" for s in signals
        )

        # ── Failure mode breakdown ────────────────────────────────
        # Count how many times each failure mode appeared.
        # Useful for tracking which error types are reduced across runs
        # after prompt patching or RAG addition (answers SubQ1).
        failure_modes_seen = [
            s.get("failure_mode")
            for s in signals
            if s.get("outcome") == "incorrect" and s.get("failure_mode")
        ]
        failure_mode_breakdown = dict(Counter(failure_modes_seen))
        # Always include all known modes so the table is consistent across runs
        for mode in KNOWN_FAILURE_MODES:
            failure_mode_breakdown.setdefault(mode, 0)

        # ── Confidence calibration ────────────────────────────────
        # Compare avg model confidence on correct vs incorrect cases.
        # A well-calibrated model should be more confident when it's right.
        correct_confs = [
            s.get("confidence", 0) for s in signals if s.get("outcome") == "correct"
        ]
        incorrect_confs = [
            s.get("confidence", 0) for s in signals if s.get("outcome") == "incorrect"
        ]
        confidence_calibration = {
            "avg_confidence_correct": (
                round(sum(correct_confs) / len(correct_confs), 1) if correct_confs else None
            ),
            "avg_confidence_incorrect": (
                round(sum(incorrect_confs) / len(incorrect_confs), 1) if incorrect_confs else None
            ),
        }

        # ── Routing breakdown ─────────────────────────────────────
        rag_failures = sum(s.get("correction_technique") == "rag" for s in signals)
        prompt_failures = sum(
            s.get("correction_technique") == "prompt_patch" for s in signals
        )
        flagged = sum(s.get("correction_technique") == "flag" for s in signals)

        precision = correct / total
        accuracy = correct / total  # same here since each signal = one case

        alerts = []
        if precision < 0.8:
            alerts.append("Annotation precision dropped below 0.8")
        if hallucinations > 0:
            alerts.append(f"{hallucinations} hallucination(s) detected")
        if rag_failures > 0:
            alerts.append(f"{rag_failures} case(s) suggest RAG store needs more examples")
        if prompt_failures >= 2:
            alerts.append(f"{prompt_failures} repeated reasoning failures — prompt patch recommended")

        return {
            "precision": round(precision, 3),
            "accuracy": round(accuracy, 3),
            "hallucinations": hallucinations,
            "correct": correct,
            "incorrect": incorrect,
            "total": total,
            "failure_mode_breakdown": failure_mode_breakdown,
            "confidence_calibration": confidence_calibration,
            "rag_routed_failures": rag_failures,
            "prompt_routed_failures": prompt_failures,
            "flagged_cases": flagged,
            "alerts": alerts,
        }