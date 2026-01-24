import json
import os
from datetime import datetime
from typing import Dict


class PromptAgent:
    def __init__(self, prompt_dir="prompts"):
        self.prompt_dir = prompt_dir
        os.makedirs(prompt_dir, exist_ok=True)

    def propose_update(
        self,
        current_prompt: Dict,
        metrics: Dict,
        reason: str = "",
    ) -> Dict:
        """
        Proposes a new prompt but DOES NOT auto-activate it.
        Human must approve loading it next run.
        """

        new_prompt = current_prompt.copy()

        # Example: inject clarification based on errors
        if metrics.get("symptom_confusion_rate", 0) > 0.2:
            new_prompt["instructions"].append(
                "Only label diagnoses that are explicitly confirmed by a physician."
            )

        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        path = os.path.join(self.prompt_dir, f"{version}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_prompt, f, indent=2)

        return {
            "version": version,
            "path": path,
            "reason": reason,
        }
