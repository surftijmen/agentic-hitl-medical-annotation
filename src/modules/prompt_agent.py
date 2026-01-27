import json
import os
from datetime import datetime
from typing import Dict, List

PROMPT_PATCHES = {
    "unsupported_inference": [
        "Do NOT infer diagnoses from prognosis discussions, family meetings, or treatment decisions.",
        "Only extract diagnoses that are explicitly stated by a clinician."
    ],
    "symptom_diagnosis_confusion": [
        "Do NOT label symptoms, complaints, or clinical findings as diagnoses unless explicitly diagnosed."
    ],
    "hallucination": [
        "If a diagnosis is not explicitly mentioned in the text, do not guess or infer it."
    ],
    "missed_entity": [
        "Ensure all explicitly stated diagnoses are extracted, even if mentioned briefly or once."
    ]
}

class PromptAgent:
    def __init__(self, prompt_dir: str = "logs/prompts"):
        self.prompt_dir = prompt_dir
        os.makedirs(self.prompt_dir, exist_ok=True)

    def propose_update(
        self,
        base_prompt: Dict,
        parsed_feedback: List[Dict],
    ) -> Dict:
        """
        Proposes a prompt PATCH based on aggregated human feedback.
        Does NOT activate automatically — requires human approval.
        """

        # ---- Aggregate failure modes ----
        failure_counts: Dict[str, int] = {}

        for fb in parsed_feedback:
            if (
                fb.get("outcome") == "incorrect"
                and fb.get("error_locus") == "reasoning"
            ):
                mode = fb.get("failure_mode")
                if mode:
                    failure_counts[mode] = failure_counts.get(mode, 0) + 1

        # ---- Generate patches ----
        patches = []

        for mode, count in failure_counts.items():
            if mode in PROMPT_PATCHES and count > 0:
                patches.append({
                    "failure_mode": mode,
                    "count": count,
                    "instructions_added": PROMPT_PATCHES[mode],
                })

        if not patches:
            return {
                "status": "no_update_needed",
                "base_prompt": base_prompt.get("version"),
            }

        # ---- Build proposed prompt update ----
        proposed_prompt = {
            "base_prompt": base_prompt.get("version"),
            "patches": patches,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "pending_human_review",
        }

        version = datetime.now().strftime("v%Y%m%d_%H%M%S_patch")
        path = os.path.join(self.prompt_dir, f"{version}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(proposed_prompt, f, indent=2)

        return {
            "version": version,
            "path": path,
            "num_patches": len(patches),
            "status": "pending_human_review",
        }
