from typing import Dict, Optional


class HumanReviewer:
    def review(self, annotation: Dict, gold: Optional[str]) -> Dict:
        return {
            "correct": annotation.get("diagnosis") == gold,
            "confidence": annotation.get("confidence_level", 0),
        }
