from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import json

# -----------------------
# PROMPT UPDATE SUGGESTION
# -----------------------

class PromptSuggestion(BaseModel):
    suggested_change: str
    rationale: str


# -----------------------
# PROMPT AGENT
# -----------------------

class PromptAgent:
    """
    Suggests minimal prompt updates based on feedback issues.
    Only proposes small, safe modifications using LLM guidance.
    """

    def __init__(self, client: genai.Client, model_name: str, temperature: float = 0.0):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def suggest_prompt_update(
        self,
        batch_results: List[Dict],
        metrics,
        prompt_dict: Dict,
        recent_issues: Optional[List[str]] = None,
    ) -> Optional[PromptSuggestion]:
        """
        Returns a single conservative prompt update if the batch has issues.
        """
        if metrics.json_error_rate < 0.05 and (metrics.accuracy is None or metrics.accuracy >= 0.7):
            # No update needed
            return None

        # Build a minimal system instruction for LLM
        issues_text = ", ".join(set(recent_issues)) if recent_issues else "general errors in outputs"

        instruction = (
            f"Suggest a single minimal, conservative update to the following prompt to "
            f"address the issues: {issues_text}. "
            "Do not change instructions unnecessarily. Only suggest one small change."
        )

        system_prompt = prompt_dict.get("system_prompt", "")
        user_prompt = f"Current prompt:\n{json.dumps(prompt_dict, indent=2)}\n\n{instruction}"

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=300,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        # Return a conservative suggestion
        return PromptSuggestion(
            suggested_change=response.text.strip(),
            rationale="Suggested by LLM based on recent batch validation issues."
        )
