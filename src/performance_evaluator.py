from typing import List, Dict


class PerformanceEvaluator:
    def evaluate(self, signals: List[Dict]) -> Dict:
        total = len(signals)
        errors = sum(s["error_type"] != "none" for s in signals)

        hallucinations = sum(
            s["error_type"] == "hallucination" for s in signals
        )

        precision = 1 - (errors / total) if total else 0.0

        alerts = []
        if precision < 0.8:
            alerts.append("Annotation precision dropped")

        if hallucinations > 0:
            alerts.append("Hallucinations detected")

        return {
            "precision": precision,
            "hallucinations": hallucinations,
            "alerts": alerts,
        }
