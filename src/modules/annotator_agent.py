import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import Counter
import toml
from google import genai
from google.genai import types
from modules.rag_memory import RAGMemory


# ================================================================
# TECHNIQUE CONFIG
# ================================================================

@dataclass
class TechniqueConfig:
    """
    Flags controlling which prompt engineering techniques are active.
    Each technique is independently togglable for ablation studies.
    """
    use_rag: bool = False
    use_self_consistency: bool = False
    n_consistency_samples: int = 5
    consistency_temperature: float = 0.7
    use_prompt_patching: bool = False      # applied between runs in main.py
    use_feedback_routing: bool = False     # route failures to rag vs prompt_patch


# ================================================================
# ANNOTATOR
# ================================================================

class Annotator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        mimic_notes_path: str = "data/mimiciii_notes.parquet",
        mimic_adm_path: str = "data/mimiciii_patients_admissions.parquet",
        prompt_path: str = "logs/prompts/v1_initial.json",
        secrets_path: str = "./secrets.toml",
        debug: bool = False,
        technique_config: TechniqueConfig = None,
        # Legacy parameter — kept for backwards compatibility
        use_rag: bool = False,
    ):
        self.model_name = model_name
        self.mimic_notes_path = mimic_notes_path
        self.mimic_adm_path = mimic_adm_path
        self.prompt_path = prompt_path
        self.secrets_path = secrets_path
        self.debug = debug

        # TechniqueConfig takes precedence; fall back to legacy use_rag flag
        self.config = technique_config or TechniqueConfig(use_rag=use_rag)

        # RAG is only instantiated if needed — caller should override with
        # shared instance via annotator.rag = rag to persist across runs
        self.rag: Optional[RAGMemory] = None

        self.prompt_version = prompt_path
        self.prompt_dict = self.load_prompt_dict()
        self._load_client()

    # ── Properties for backwards compat ─────────────────────────

    @property
    def use_rag(self) -> bool:
        return self.config.use_rag

    @use_rag.setter
    def use_rag(self, value: bool):
        self.config.use_rag = value

    # ── Internal utils ───────────────────────────────────────────

    def _log(self, *args):
        if self.debug:
            print(*args)

    def _load_client(self):
        secrets = toml.load(self.secrets_path)
        self.client = genai.Client(api_key=secrets["api"]["key"])

    # ── Prompt ───────────────────────────────────────────────────

    def load_prompt_dict(self) -> Dict:
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_user_prompt(
        self,
        input_text: Optional[str] = None,
        examples: Optional[List[str]] = None,
    ) -> str:
        parts = []

        if self.prompt_dict.get("query"):
            parts.append(self.prompt_dict["query"])

        if self.prompt_dict.get("instructions"):
            parts.append("Instructions:")
            for inst in self.prompt_dict["instructions"]:
                parts.append(f"- {inst}")

        if examples:
            parts.append(
                "Here are some examples for text that do or do not contain "
                "explicit disease diagnoses:"
            )
            parts.extend(examples)

        if self.prompt_dict.get("format_instruction"):
            parts.append(self.prompt_dict["format_instruction"])

        if input_text:
            parts.append(f"Medical text: {input_text}")

        return "\n\n".join(parts)

    # ── Text cleaning ────────────────────────────────────────────

    @staticmethod
    def clean_mimic_note(text: str) -> str:
        text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
        text = re.sub(
            r"Admission Date:.*?Discharge Date:.*?",
            "",
            text,
            flags=re.I | re.S,
        )
        text = re.sub(
            r"Dictated By:.*|JOB#:.*|MEDQUIST.*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"D:\s*\d+.*|T:\s*\d+.*", "", text)

        headers = [
            "Service:",
            "ADDENDUM:",
            "RADIOLOGIC STUDIES:",
            "HEAD CT:",
            "ABDOMINAL CT:",
        ]
        for h in headers:
            text = re.sub(re.escape(h), "", text, flags=re.I)

        text = re.sub(r",?\s*M\.D\..*$", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ── Output parsing ───────────────────────────────────────────

    def extract_results(self, llm_output: str) -> Dict:
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)

        if not match:
            return {
                "diagnosis": "none",
                "is_diagnosis_given": 0,
                "confidence_level": 0,
                "error": "No JSON found",
                "raw_output": llm_output,
            }

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            return {
                "diagnosis": "none",
                "is_diagnosis_given": 0,
                "confidence_level": 0,
                "error": str(e),
                "raw_output": llm_output,
            }

        data["raw_output"] = llm_output
        return data

    # ── Gemini inference (single call) ───────────────────────────

    def _build_config(self, temperature: float = 0.0) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.prompt_dict.get("system_prompt", ""),
            temperature=temperature,
            max_output_tokens=50000,
            tools=[self.rag.get_tool()] if self.config.use_rag and self.rag else [],
        )

    def _call_model(self, user_prompt: str, temperature: float = 0.0) -> str:
        config = self._build_config(temperature)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )
        return response

    def _log_rag_grounding(self, response):
        grounding_count = 0
        try:
            for candidate in (response.candidates or []):
                meta = getattr(candidate, "grounding_metadata", None)
                if meta:
                    chunks = getattr(meta, "grounding_chunks", None) or []
                    grounding_count += len(chunks)
        except Exception:
            pass
        print(f"    RAG: Gemini retrieved and grounded on {grounding_count} case(s)")

    def analyze_single(self, medical_text: str) -> Dict:
        """Single inference call at temperature=0."""
        user_prompt = self.build_user_prompt(input_text=medical_text)
        response = self._call_model(user_prompt, temperature=0.0)

        if self.config.use_rag:
            self._log_rag_grounding(response)

        return self.extract_results(response.text)

    # ── Self-consistency ─────────────────────────────────────────

    def analyze_with_consistency(self, medical_text: str) -> Dict:
        """
        Sample the model N times and take the majority vote.
        Consistency score replaces self-reported confidence — much more meaningful.
        """
        n = self.config.n_consistency_samples
        temp = self.config.consistency_temperature
        user_prompt = self.build_user_prompt(input_text=medical_text)

        votes = []
        raw_results = []

        for i in range(n):
            response = self._call_model(user_prompt, temperature=temp)
            result = self.extract_results(response.text)
            votes.append(result.get("diagnosis", "none").lower().strip())
            raw_results.append(result)

        counts = Counter(votes)
        majority_diagnosis, majority_count = counts.most_common(1)[0]
        consistency_score = majority_count / n

        if self.debug:
            print(f"    Consistency votes: {dict(counts)}")

        # Find the full result object matching the majority diagnosis
        majority_result = next(
            (r for r in raw_results if r.get("diagnosis", "").lower().strip() == majority_diagnosis),
            raw_results[0]
        )

        return {
            **majority_result,
            "diagnosis": majority_result.get("diagnosis"),   # preserve original casing
            "confidence_level": int(consistency_score * 100),
            "consistency_score": consistency_score,
            "vote_distribution": dict(counts),
            "n_samples": n,
        }

    # ── Public entrypoint ────────────────────────────────────────

    def analyze_medical_text(self, medical_text: str) -> Dict:
        """
        Route to the appropriate inference strategy based on TechniqueConfig.
        """
        if self.config.use_self_consistency:
            return self.analyze_with_consistency(medical_text)
        else:
            return self.analyze_single(medical_text)

    # ── Sampling / testing ───────────────────────────────────────

    def random_mimic_test(self, n: int = 1):
        import pandas as pd
        mimic_df = pd.read_parquet(self.mimic_notes_path)
        samples = mimic_df.sample(n)

        for _, row in samples.iterrows():
            clean_text = self.clean_mimic_note(row["TEXT"])
            self._log("=" * 80)
            self._log(f"SUBJECT_ID: {row['SUBJECT_ID']} | HADM_ID: {row['HADM_ID']}")
            self._log("\nCLEANED NOTE (TRUNCATED):\n")
            self._log(clean_text[:2000])

            result = self.analyze_medical_text(clean_text)
            self._log(result)

            print("\n===== FINAL INFORMATION =====")
            print(f"Agents diagnosis: {result.get('diagnosis')}")
            print("Actual diagnosis:", row["DIAGNOSIS"])

            if "error" in result:
                self._log(f"Error: {result['error']}")

        return (result.get("diagnosis"), row["DIAGNOSIS"])


if __name__ == "__main__":
    pipeline = Annotator(debug=False)
    pipeline.random_mimic_test(n=1)