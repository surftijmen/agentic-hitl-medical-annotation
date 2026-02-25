from typing import List, Dict


class PerformanceEvaluator:
    def evaluate(self, signals: List[Dict]) -> Dict:
        total = len(signals)
        if total == 0:
            return {"precision": 0.0, "accuracy": 0.0, "hallucinations": 0, "alerts": []}

        correct = sum(s.get("outcome") == "correct" for s in signals)
        incorrect = sum(s.get("outcome") == "incorrect" for s in signals)

        hallucinations = sum(
            s.get("failure_mode") == "hallucination" for s in signals
        )

        # Which failure modes are being routed to which technique
        rag_failures = sum(
            s.get("correction_technique") == "rag" for s in signals
        )
        prompt_failures = sum(
            s.get("correction_technique") == "prompt_patch" for s in signals
        )
        flagged = sum(
            s.get("correction_technique") == "flag" for s in signals
        )

        precision = correct / total
        accuracy = correct / total  # same here since each signal = one case

        alerts = []
        if precision < 0.8:
            alerts.append("Annotation precision dropped")
        if hallucinations > 0:
            alerts.append("Hallucinations detected")
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
            "rag_routed_failures": rag_failures,
            "prompt_routed_failures": prompt_failures,
            "flagged_cases": flagged,
            "alerts": alerts,
        }