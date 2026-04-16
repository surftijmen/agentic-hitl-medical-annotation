import json
import re
import time
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
    n_consistency_samples: int = 3
    consistency_temperature: float = 0.7
    use_prompt_patching: bool = False      # applied between runs in main.py
    use_feedback_routing: bool = False     # route failures to rag vs prompt_patch
    chain_of_thoughts: bool = False        # ask model to reason step-by-step before output
    use_few_shot: bool = False             # inject static few-shot examples into the prompt


# ── Static few-shot examples ─────────────────────────────────────────────────
# These are fictional but medically realistic examples.
# They show the model exactly what a correct extraction looks like across three
# representative cases: confirmed positive, confirmed none, and synonym mapping.

# Each example is (medical_text, output_without_cot, output_with_cot).
# When CoT is active the model requires a "reasoning" field — using the same
# examples without it causes format conflicts that produce hallucinations.
FEW_SHOT_EXAMPLES: list[tuple[str, str, str]] = [
    # ── Positive: simple explicit confirmation ──────────────────────────────
    (
        "Patient is a 72-year-old male admitted for acute decompensated congestive "
        "heart failure. Echocardiogram confirms EF of 30%. History of CHF.",
        '{"diagnosis": "CHF", "is_diagnosis_given": 1, "confidence_level": 97}',
        '{"reasoning": "1. Conditions: CHF. 2. CHF is explicitly confirmed (\'acute decompensated congestive heart failure\', EF 30%). 3. Maps to CHF. 4. CHF is the primary confirmed diagnosis.", "diagnosis": "CHF", "is_diagnosis_given": 1, "confidence_level": 97}',
    ),
    # ── Negative: rule-out / not yet confirmed ──────────────────────────────
    (
        "Patient presents with dyspnea and productive cough. Rule out pneumonia. "
        "Chest X-ray inconclusive. No confirmed diagnosis documented.",
        '{"diagnosis": "none", "is_diagnosis_given": 0, "confidence_level": 95}',
        '{"reasoning": "1. Conditions: pneumonia (rule out). 2. Pneumonia is NOT confirmed — explicitly stated as rule-out and CXR inconclusive. 3. No confirmed condition maps to the allowed list. 4. Return none.", "diagnosis": "none", "is_diagnosis_given": 0, "confidence_level": 95}',
    ),
    # ── Hard negative: clinical indicators present but diagnosis word absent ─
    (
        "67-year-old female brought in by EMS with fever to 104°F, HR 168, "
        "hypotension (BP 82/50), and altered mental status. WBC 22k. Blood "
        "cultures drawn. Started on broad-spectrum antibiotics empirically for "
        "presumed infection. ICU admission for hemodynamic monitoring.",
        '{"diagnosis": "none", "is_diagnosis_given": 0, "confidence_level": 90}',
        '{"reasoning": "1. Conditions mentioned: fever, tachycardia, hypotension, altered mental status, elevated WBC — signs of infection. 2. The word \'sepsis\' never appears. \'Presumed infection\' is not a confirmed diagnosis. I must not infer Sepsis from symptoms alone. 3. No allowed diagnosis is explicitly confirmed. 4. Return none.", "diagnosis": "none", "is_diagnosis_given": 0, "confidence_level": 90}',
    ),
    # ── Positive: same presentation but diagnosis explicitly named ──────────
    (
        "55-year-old male admitted to the ICU with sepsis secondary to a urinary "
        "source. Blood cultures grew E. coli. Treated for septic shock with "
        "vasopressors and IV antibiotics.",
        '{"diagnosis": "Sepsis", "is_diagnosis_given": 1, "confidence_level": 99}',
        '{"reasoning": "1. Conditions: sepsis (explicitly stated), septic shock (sub-type). 2. Sepsis is explicitly confirmed: \'admitted to the ICU with sepsis\'. 3. Maps directly to Sepsis. 4. Sepsis is the primary diagnosis.", "diagnosis": "Sepsis", "is_diagnosis_given": 1, "confidence_level": 99}',
    ),
    # ── Hard positive: multiple confirmed comorbidities — pick primary ───────
    (
        "78-year-old male with known CAD, CHF (EF 35%), and CKD stage 3 presents "
        "with crushing chest pain. EKG shows ST-elevation in leads II, III, aVF. "
        "Troponin peaked at 45. Emergent cardiac catheterization confirmed acute "
        "inferior ST-elevation myocardial infarction. PCI with stent to RCA performed.",
        '{"diagnosis": "MI", "is_diagnosis_given": 1, "confidence_level": 100}',
        '{"reasoning": "1. Conditions: CAD (history), CHF (history), CKD (history), MI (acute, confirmed). 2. All four are confirmed, but CAD/CHF/CKD are prior history. MI is the acute admission diagnosis — \'confirmed acute inferior ST-elevation myocardial infarction\'. 3. MI maps to MI. 4. MI is the most clinically primary confirmed diagnosis.", "diagnosis": "MI", "is_diagnosis_given": 1, "confidence_level": 100}',
    ),
]

# ── CoT format instruction override ──────────────────────────────────────────
# Replaces the default format_instruction when chain_of_thoughts is active.
# Reasoning is embedded as a field so the JSON parser never breaks.

COT_FORMAT_INSTRUCTION = (
    'Return ONLY valid JSON with exactly these fields: '
    '{"reasoning": "your step-by-step thinking", '
    '"diagnosis": "string or none", '
    '"is_diagnosis_given": 0 or 1, '
    '"confidence_level": integer from 0 to 100}. '
    'Do not add, remove, or rename fields. Do not use markdown outside the JSON.'
)

COT_THINKING_PROMPT = (
    "Before answering, reason step-by-step inside the 'reasoning' field:\n"
    "  1. List every medical condition mentioned in the text.\n"
    "  2. For each condition, decide if it is explicitly confirmed — not "
    "'possible', 'suspected', 'rule out', or negated ('no evidence of').\n"
    "  3. Map confirmed conditions to the allowed_diagnoses list "
    "(use synonym mapping where needed).\n"
    "  4. Select the most clinically relevant confirmed diagnosis, or 'none'.\n"
    "CRITICAL: Only reference words and phrases that literally appear in the text. "
    "Do NOT invent section names, quotes, or content that is not present."
)


# ================================================================
# ANNOTATOR
# ================================================================

class Annotator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        mimic_notes_path: str = "data/new/processed/notes_with_gold.parquet",
        mimic_adm_path: Optional[str] = None,  # legacy, unused with the new processed parquet
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

        allowed = self.prompt_dict.get("allowed_diagnoses")
        if allowed:
            parts.append(
                "Allowed diagnoses (you MUST use one of these exact labels, or 'none'):\n"
                + ", ".join(allowed)
            )

        if self.prompt_dict.get("instructions"):
            parts.append("Instructions:")
            for inst in self.prompt_dict["instructions"]:
                parts.append(f"- {inst}")

        # ── Few-shot examples ─────────────────────────────────────
        # Priority: caller-supplied dynamic examples > static FEW_SHOT_EXAMPLES
        # When CoT is active, use the CoT-formatted output (index 2) so the
        # examples match the required format — mismatched formats cause the
        # model to hallucinate or produce unparseable output.
        active_examples = examples
        if not active_examples and self.config.use_few_shot:
            output_idx = 2 if self.config.chain_of_thoughts else 1
            active_examples = [
                f"Medical text: {ex[0]}\nOutput: {ex[output_idx]}"
                for ex in FEW_SHOT_EXAMPLES
            ]

        if active_examples:
            parts.append(
                "Here are examples of correct extractions. "
                "Follow the same reasoning and format:"
            )
            for ex in active_examples:
                parts.append(ex)

        # ── Chain-of-thought ──────────────────────────────────────
        if self.config.chain_of_thoughts:
            parts.append(COT_THINKING_PROMPT)
            parts.append(COT_FORMAT_INSTRUCTION)
        elif self.prompt_dict.get("format_instruction"):
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

    @staticmethod
    def redact_discharge_section(text: str) -> str:
        """Remove only the sections that directly state the discharge diagnosis.

        Keeps the full clinical narrative (HPI, hospital course, medications,
        labs, etc.) so the model has enough context to reason about the diagnosis,
        but strips the sections that simply list the answer as a labelled output:
          - DISCHARGE DIAGNOSES / DISCHARGE DIAGNOSIS
          - FINAL DIAGNOSIS / PRINCIPAL DIAGNOSIS / PRIMARY DIAGNOSIS
          - DISCHARGE CONDITION (often restates the diagnosis)
          - DISCHARGE MEDICATIONS (lists drugs by diagnosis, gives it away)
          - DISCHARGE INSTRUCTIONS / FOLLOW-UP (post-discharge, not clinical)

        Falls back to the original text if nothing is redacted.
        """
        # Normalise multi-space headers
        text = re.sub(r"([A-Z])\s{2,}([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z])\s{2,}([A-Z])", r"\1 \2", text)

        _HEADER = re.compile(r"(?<!\w)([A-Za-z][A-Za-z0-9 /\-]{2,}):")

        # Sections to REMOVE — these directly reveal the discharge diagnosis
        _DROP_NAMES = re.compile(
            r"^(?:DISCHARGE\s+DIAGNOS(?:IS|ES)"
            r"|FINAL\s+DIAGNOS(?:IS|ES)"
            r"|PRINCIPAL\s+DIAGNOS(?:IS|ES)"
            r"|PRIMARY\s+DIAGNOS(?:IS|ES)"
            r"|DISCHARGE\s+CONDITION"
            r"|CONDITION\s+(?:AT|ON)\s+DISCHARGE"
            r"|DISCHARGE\s+MEDICATIONS?"
            r"|MEDICATIONS?\s+ON\s+DISCHARGE"
            r"|DISCHARGE\s+INSTRUCTIONS?(?:/FOLLOWUP)?"
            r"|FOLLOWUP\s+INSTRUCTIONS?"
            r"|FOLLOW[\s\-]?UP\s+(?:PLANS?|APPOINTMENTS?|INSTRUCTIONS?)"
            r"|DISCHARGE\s+(?:STATUS|DISPOSITION)"
            r")$",
            re.I,
        )

        headers = list(_HEADER.finditer(text))
        if not headers:
            return text

        kept_parts = [text[:headers[0].start()]]  # preamble

        for idx, m in enumerate(headers):
            header_name = m.group(1).strip()
            start = m.start()
            end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
            if not _DROP_NAMES.match(header_name):
                kept_parts.append(text[start:end])

        result = " ".join(kept_parts).strip()
        if len(result) < 100:
            return text  # fallback
        return result

    # ── Output parsing ───────────────────────────────────────────

    def extract_results(self, llm_output: str) -> Dict:
        if not llm_output:
            return {
                "diagnosis": "none",
                "is_diagnosis_given": 0,
                "confidence_level": 0,
                "error": "Empty response from model (possible safety filter)",
                "raw_output": "",
            }

        # Strip markdown code fences the model may wrap its output in
        cleaned = re.sub(r"```(?:json)?", "", llm_output).strip()

        data = None

        # 1. Direct parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Find first '{' and decode exactly one JSON object from that position
        if data is None:
            start = cleaned.find("{")
            if start != -1:
                decoder = json.JSONDecoder()
                try:
                    data, _ = decoder.raw_decode(cleaned, start)
                except json.JSONDecodeError:
                    pass

        if data is None:
            return {
                "diagnosis": "none",
                "is_diagnosis_given": 0,
                "confidence_level": 0,
                "error": "No JSON found",
                "raw_output": llm_output,
            }

        # Preserve the model's free-text diagnosis verbatim. The judge
        # (AutoReviewer) compares it semantically against the ICD-9 LONG_TITLE
        # — normalising to a canonical label here would destroy the detail
        # the semantic judge uses.
        raw_dx = data.get("diagnosis")
        if not raw_dx or str(raw_dx).strip() == "":
            data["diagnosis"] = "none"
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

    def _call_model(self, user_prompt: str, temperature: float = 0.0):
        config = self._build_config(temperature)
        max_retries = 4
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
            except Exception as e:
                err = str(e)
                if ("503" in err or "429" in err or "UNAVAILABLE" in err) and attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f" [retry {attempt+1}/{max_retries-1}, {wait}s]", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise

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

        result = self.extract_results(response.text)

        if self.config.chain_of_thoughts and self.debug and "reasoning" in result:
            print(f"    CoT reasoning: {result['reasoning'][:300]}...")

        return result

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

        Technique combinations handled:
          - self_consistency alone or combined with CoT/few-shot:
            build_user_prompt already embeds CoT/few-shot instructions,
            so each consistency sample automatically uses them.
          - single inference: direct call with whatever prompt is built.
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
            print(f"Actual diagnosis: {row['long_title']} (ICD-9 {row['icd9_code']})")

            if "error" in result:
                self._log(f"Error: {result['error']}")

        return (result.get("diagnosis"), row["long_title"])


if __name__ == "__main__":
    pipeline = Annotator(debug=False)
    pipeline.random_mimic_test(n=1)