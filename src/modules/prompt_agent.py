import json
import os
from datetime import datetime
from typing import Dict, List
import toml
from google import genai
from google.genai import types


class PromptAgent:
    """
    PromptAgent proposes *instruction-level patches* to an existing prompt.
    It NEVER rewrites:
      - system_prompt
      - query
      - allowed_diagnoses
      - format_instruction
    """

    def __init__(
        self,
        prompt_dir: str = "logs/prompts",
        secrets_path: str = "./secrets.toml",
        model_name: str = "gemini-2.5-flash",
        debug: bool = False,
    ):
        self.prompt_dir = prompt_dir
        self.model_name = model_name
        self.secrets_path = secrets_path
        self.debug = debug

        os.makedirs(self.prompt_dir, exist_ok=True)
        self._load_client()

    def _log(self, *args):
        if self.debug:
            print(*args)

    def _load_client(self):
        secrets = toml.load(self.secrets_path)
        self.client = genai.Client(api_key=secrets["api"]["key"])

    # ============================================================
    # FEEDBACK AGGREGATION
    # ============================================================

    def _aggregate_feedback(self, parsed_feedback: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregates repeated failure modes and example rationales.
        """
        agg: Dict[str, Dict] = {}

        for fb in parsed_feedback:
            if fb.get("outcome") != "incorrect":
                continue

            mode = fb.get("failure_mode")
            if not mode or mode == "ambiguous_case":
                # ambiguous_case = data quality issue, not a prompt bug → skip
                continue

            if mode not in agg:
                agg[mode] = {
                    "count": 0,
                    "error_locus": fb.get("error_locus"),
                    "examples": [],
                }

            agg[mode]["count"] += 1

            rationale = fb.get("human_rationale")
            if rationale:
                agg[mode]["examples"].append(rationale)

        return agg

    # ============================================================
    # LLM PROMPT CONSTRUCTION
    # ============================================================

    def _build_llm_prompt(
        self,
        base_prompt: Dict,
        aggregated_failures: Dict[str, Dict],
    ) -> str:
        failures_text = []

        for mode, info in aggregated_failures.items():
            failures_text.append(
                f"""
Failure mode: {mode}
Count: {info['count']}
Error locus: {info.get('error_locus')}
Reviewer examples:
{chr(10).join('- ' + ex for ex in info['examples'][:3])}
"""
            )

        return f"""
You are assisting with IMPROVING an existing medical information extraction prompt.

IMPORTANT CONSTRAINTS (DO NOT VIOLATE):
- You MUST NOT rewrite or rephrase the task.
- You MUST NOT change the system prompt.
- You MUST NOT change the query.
- You MUST NOT change the allowed_diagnoses list.
- You MUST NOT change the output JSON format.
- You MUST NOT introduce new task scope.

You are ONLY allowed to:
- Propose ADDITIONAL instructions
- Propose SMALL refinements to EXISTING instructions

The goal is to reduce the observed failure modes WITHOUT harming recall
of explicitly stated diagnoses.

------------------------------------
CURRENT SYSTEM PROMPT (READ ONLY):
{base_prompt.get("system_prompt", "")}

CURRENT QUERY (READ ONLY):
{base_prompt.get("query", "")}

CURRENT INSTRUCTIONS (BASELINE – TREAT AS VALIDATED):
{chr(10).join('- ' + i for i in base_prompt.get("instructions", []))}
------------------------------------

OBSERVED FAILURE MODES FROM HUMAN REVIEW:
{chr(10).join(failures_text)}

TASK:
Propose instruction-level patches only.

Return JSON in EXACTLY this format:

{{
  "instructions_to_add": [
    "New instruction text"
  ],
  "instructions_to_refine": [
    {{
      "target_instruction": "Exact text of an existing instruction",
      "refined_version": "Slightly refined version"
    }}
  ],
  "rationale": "Why these changes reduce the failures without expanding scope"
}}

If no changes are needed, return empty lists.
"""

    # ============================================================
    # LLM CALL
    # ============================================================

    def _call_llm(self, prompt: str) -> Dict:
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=5000,  # slightly higher
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        text = response.text.strip()

        # Remove markdown fences if present
        if text.startswith("```"):
            text = text.split("```", 2)[1]

        # Try parsing progressively
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract largest JSON block
        import re
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {
            "error": "Failed to parse LLM JSON",
            "raw_output": response.text,
        }

    # ============================================================
    # PUBLIC API
    # ============================================================

    def propose_update(
        self,
        base_prompt: Dict,
        parsed_feedback: List[Dict],
        min_failures: int = 2,
    ) -> Dict:
        """
        Generates an instruction-level prompt patch.
        Requires human approval before application.
        """

        aggregated = self._aggregate_feedback(parsed_feedback)

        total_failures = sum(v["count"] for v in aggregated.values())
        if total_failures < min_failures:
            return {
                "status": "insufficient_signal",
                "reason": "Not enough repeated failures to justify prompt update",
                "observed_failures": aggregated,
            }

        llm_prompt = self._build_llm_prompt(base_prompt, aggregated)
        llm_result = self._call_llm(llm_prompt)

        if "error" in llm_result:
            print("LLM failure")
            # return {
            #     "status": "llm_failure",
            #     "error": llm_result,
            # }

        proposal = {
            "base_prompt_version": base_prompt.get("version"),
            "generated_at": datetime.utcnow().isoformat(),
            "aggregated_failures": {
                k: {"count": v["count"]} for k, v in aggregated.items()
            },
            "instruction_patch": llm_result,
            "status": "pending_human_review",
        }

        version = datetime.now().strftime("v%Y%m%d_%H%M%S_instruction_patch")
        path = os.path.join(self.prompt_dir, f"{version}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)

        return {
            "status": "pending_human_review",
            "version": version,
            "path": path,
            "num_failure_modes": len(aggregated),
        }
