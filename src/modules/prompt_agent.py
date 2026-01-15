from typing import Dict


class PromptAgent:
    def propose_update(
        self,
        prompt_dict: Dict,
        metrics: Dict,
    ) -> Dict:
        if not metrics["alerts"]:
            return prompt_dict

        updated = prompt_dict.copy()

        if "Hallucinations detected" in metrics["alerts"]:
            updated["instructions"] = (
                prompt_dict["instructions"]
                + ["Never output a diagnosis unless explicitly stated."]
            )

        return updated
