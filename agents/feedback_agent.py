from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# -----------------------
# MODELS
# -----------------------

class ValidationResult(BaseModel):
    valid: bool
    issues: List[str] = []


class BatchMetrics(BaseModel):
    total_samples: int
    json_error_rate: float
    diagnosis_rate: float
    accuracy: Optional[float] = None


class HealthStatus(BaseModel):
    healthy: bool
    actions: List[str] = []


# -----------------------
# FEEDBACK AGENT
# -----------------------

class FeedbackAgent:
    """
    Deterministic feedback agent for model outputs.
    Responsible for:
        - validating each output
        - aggregating batch metrics
        - simple health check
    """

    def validate(self, result: Dict) -> ValidationResult:
        issues = []

        if "error" in result:
            issues.append("json_parse_error")

        if result.get("is_diagnosis_given") not in [0, 1]:
            issues.append("invalid_is_diagnosis_given")

        conf = result.get("confidence_level")
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 100):
            issues.append("invalid_confidence_level")

        # Contradiction: diagnosis should be 'none' if is_diagnosis_given == 0
        if result.get("is_diagnosis_given") == 0 and result.get("diagnosis") not in ["none", None]:
            issues.append("contradictory_output")

        return ValidationResult(valid=len(issues) == 0, issues=issues)

    def evaluate_batch(self, results: List[Dict], true_labels: Optional[List[str]] = None) -> BatchMetrics:
        total = len(results)
        json_errors = sum("error" in r for r in results)
        diagnosis_given = sum(r.get("is_diagnosis_given", 0) for r in results)

        accuracy = None
        if true_labels:
            correct = sum(r.get("diagnosis") == y for r, y in zip(results, true_labels))
            accuracy = correct / total

        return BatchMetrics(
            total_samples=total,
            json_error_rate=json_errors / total,
            diagnosis_rate=diagnosis_given / total,
            accuracy=accuracy,
        )

    def health_check(self, metrics: BatchMetrics) -> HealthStatus:
        actions = []
        if metrics.json_error_rate > 0.05:
            actions.append("freeze_prompt_updates")

        if metrics.diagnosis_rate > 0.9:
            actions.append("possible_overdiagnosis")

        if metrics.accuracy is not None and metrics.accuracy < 0.7:
            actions.append("performance_degradation")

        return HealthStatus(healthy=len(actions) == 0, actions=actions)
