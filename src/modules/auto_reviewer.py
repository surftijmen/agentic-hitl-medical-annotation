"""
AutoReviewer — LLM-as-judge replacement for HumanReviewer.

Uses a stronger Gemini model (default: gemini-2.5-pro) to simulate the
expert human reviewer. Mirrors the exact .review() / .close() interface of
HumanReviewer so run_pipeline() can swap between modes with one parameter.

Why this is valid for research:
  - The annotator runs on gemini-2.5-flash (weaker, faster model).
  - The judge runs on gemini-2.5-pro (stronger model acting as "expert").
  - This is the standard "LLM-as-judge" pattern used in NLP research
    (e.g. MT-Bench, AlpacaEval) to scale evaluation beyond manual labelling.
  - You should still run a subset with real human review for final validation
    and to verify the auto-reviewer's judgments are trustworthy.
"""

import json
import re
import time
import toml
from typing import Dict, Optional
from google import genai
from google.genai import types


# Diagnoses the annotator is allowed to return — the judge needs this to
# understand what valid synonym mappings look like.
ALLOWED_DIAGNOSES = [
    "Hypertension", "Diabetes", "COPD", "CHF", "Pneumonia", "Sepsis",
    "Atrial Fibrillation", "CAD", "CKD", "Asthma", "Cancer", "Stroke",
    "MI", "Tuberculosis", "Anemia", "Cirrhosis", "HIV", "Obesity",
    "UTI", "DVT", "PE",
]

FAILURE_MODES = [
    "hallucination",
    "missed_entity",
    "terminology_gap",
    "unsupported_inference",
    "symptom_diagnosis_confusion",
    "ambiguous_case",
]

_SYSTEM_PROMPT = """\
You are a senior clinician evaluating a medical text annotation system.
Your task: decide if the annotator's extracted diagnosis is CORRECT or INCORRECT.

━━━ WHAT THE ANNOTATOR MUST DO ━━━
1. Extract ONLY diagnoses that are explicitly confirmed in the note.
   (Skip suspected, possible, rule-out, or negated findings.)
2. Map synonyms/abbreviations to the allowed list:
   heart attack → MI | AIDS → HIV | type 2 DM / diabetes mellitus → Diabetes
   end-stage heart failure / congestive heart failure / CHF → CHF
   COPD exacerbation → COPD | acute MI / STEMI / NSTEMI → MI
   pulmonary embolism → PE | deep vein thrombosis → DVT
   urinary tract infection → UTI | atrial fibrillation / AF / AFib → Atrial Fibrillation
   end-stage renal disease / ESRD / chronic renal failure → CKD (ESRD is stage 5 CKD — same condition)
3. Return "none" if no confirmed diagnosis from the list is present.
4. Return the single most clinically primary confirmed diagnosis.

━━━ ALLOWED DIAGNOSES (the only valid outputs) ━━━
{allowed_diagnoses}

━━━ FAILURE MODE GUIDE (pick the MOST SPECIFIC one) ━━━
• hallucination              — diagnosis is not supported by anything in the note
• missed_entity              — a clearly confirmed diagnosis was ignored (annotator said "none" or picked wrong one)
• terminology_gap            — correct concept but wrong label (not mapped to allowed list, e.g. "End-Stage Heart Failure" instead of "CHF")
• unsupported_inference      — annotator inferred/assumed a diagnosis not explicitly stated
• symptom_diagnosis_confusion — annotator returned a symptom or sign, not a diagnosis
• ambiguous_case             — ONLY use this if it is genuinely impossible to judge (e.g. note is too short/corrupted)

━━━ LABEL MATCHING — BE LENIENT ━━━
When comparing the annotator's label to the allowed list, accept reasonable near-matches:
• "Metastatic renal cell carcinoma" ≈ "Cancer" → CORRECT (same concept)
• "Septic shock" ≈ "Sepsis" → CORRECT (sub-type of the allowed label)
• "End-Stage Renal Disease" ≈ "CKD" → CORRECT (ESRD is stage 5 CKD — same disease)
• "Paroxysmal atrial fibrillation" ≈ "Atrial Fibrillation" → CORRECT (sub-type)
Do NOT penalise minor wording differences if the clinical concept is clearly the same.

━━━ GOLD STANDARD NOTE ━━━
The gold label is a reference from MIMIC-III, not absolute truth.
Judge by what is EXPLICITLY written in the note text.

━━━ OUTPUT FORMAT ━━━
You MUST respond with ONLY a JSON object — no prose, no markdown, no explanation outside the JSON.
{{
  "correct": true or false,
  "failure_mode": "hallucination" | "missed_entity" | "terminology_gap" | "unsupported_inference" | "symptom_diagnosis_confusion" | "ambiguous_case" | null,
  "comment": "one sentence explaining your verdict",
  "confidence": <integer 0-100>
}}
""".format(
    allowed_diagnoses=", ".join(ALLOWED_DIAGNOSES),
)


class AutoReviewer:
    """
    Drop-in replacement for HumanReviewer. Uses a Gemini judge model
    to evaluate annotations automatically.

    Usage:
        reviewer = AutoReviewer()
        feedback = reviewer.review(annotation, medical_note, gold)
        reviewer.close()  # no-op, kept for interface compatibility
    """

    def __init__(
        self,
        judge_model: str = "gemini-3-flash-preview",
        secrets_path: str = "./secrets.toml",
        verbose: bool = True,
    ):
        self.judge_model = judge_model
        self.verbose = verbose
        secrets = toml.load(secrets_path)
        self.client = genai.Client(api_key=secrets["api"]["key"])

        if self.verbose:
            print(f"  AutoReviewer ready (judge: {judge_model})")

    def review(
        self,
        annotation: Dict,
        medical_note: str,
        gold: Optional[str] = None,
        case_num: int = None,
        total_cases: int = None,
    ) -> Dict:
        """
        Ask the judge model to evaluate the annotation.
        Returns the same dict format as HumanReviewer.review().
        """
        user_prompt = self._build_judge_prompt(annotation, medical_note, gold)

        if case_num and total_cases and self.verbose:
            print(f"    Judge [{case_num}/{total_cases}] calling {self.judge_model}...", end=" ", flush=True)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.judge_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.0,       # deterministic judgment
                        max_output_tokens=1024,
                    ),
                )
                break  # success
            except Exception as e:
                err_str = str(e)
                is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str
                if is_transient and attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    if self.verbose:
                        print(f"\n    [retry {attempt+1}/{max_retries-1}] transient error, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"AutoReviewer: API call failed for model '{self.judge_model}'.\n"
                        f"  If you used Gemini 3, verify the exact model ID at "
                        f"https://ai.google.dev/gemini-api/docs/models\n"
                        f"  Original error: {e}"
                    ) from e

        result = self._parse_judgment(response.text)

        if self.verbose:
            verdict = "CORRECT" if result["correct"] else f"INCORRECT ({result.get('failure_mode', '?')})"
            print(f"    Judge: {verdict}  (confidence: {result['confidence']}%)")
            if result.get("comment"):
                print(f"    Comment: {result['comment']}")

        # Mirror exact HumanReviewer output format
        return {
            "correct": result["correct"],
            "failure_mode": result.get("failure_mode"),
            "comment": result.get("comment"),
            "confidence": result["confidence"],
            "model_confidence": annotation.get("confidence_level", 0),
            "auto_reviewed": True,
            "judge_model": self.judge_model,
        }

    def _build_judge_prompt(
        self,
        annotation: Dict,
        medical_note: str,
        gold: Optional[str],
    ) -> str:
        parts = []

        parts.append(f"MEDICAL NOTE:\n{medical_note[:3000]}")  # cap at 3k chars

        ann_summary = {
            "diagnosis": annotation.get("diagnosis"),
            "is_diagnosis_given": annotation.get("is_diagnosis_given"),
            "confidence_level": annotation.get("confidence_level"),
        }
        # Include CoT reasoning if it was generated — the judge can use it
        if annotation.get("reasoning"):
            ann_summary["reasoning"] = annotation["reasoning"]

        parts.append(f"ANNOTATOR OUTPUT:\n{json.dumps(ann_summary, indent=2)}")

        if gold:
            parts.append(f"GOLD REFERENCE DIAGNOSIS (MIMIC-III): {gold}")
        else:
            parts.append("GOLD REFERENCE DIAGNOSIS: not available")

        return "\n\n".join(parts)

    def _parse_judgment(self, llm_output: str) -> Dict:
        """Extract the JSON judgment from the model response."""
        if not llm_output:
            if self.verbose:
                print("\n    [WARN] Judge returned empty response")
            return {
                "correct": False,
                "failure_mode": "ambiguous_case",
                "comment": "Judge model returned empty response (possible safety filter).",
                "confidence": 0,
            }

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", llm_output).strip()

        # Try parsing strategies from most to least strict.
        # Using regex to extract {…} is unreliable when the comment field itself
        # contains braces — json.decoder.scan_string handles this correctly.
        data = None

        # 1. Direct parse (model returned only JSON)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Find the first '{' and let the JSON decoder consume exactly one object
        if data is None:
            start = cleaned.find("{")
            if start != -1:
                decoder = json.JSONDecoder()
                try:
                    data, _ = decoder.raw_decode(cleaned, start)
                except json.JSONDecodeError:
                    pass

        # 3. Regex fallback — handles trailing commas, prose checklists, etc.
        #    Extracts at minimum the `correct` verdict so accuracy isn't wrongly penalised.
        if data is None:
            m_correct = re.search(r'"correct"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
            if m_correct:
                correct_val = m_correct.group(1).lower() == "true"
                m_fm = re.search(
                    r'"failure_mode"\s*:\s*"([^"]+)"', cleaned
                )
                m_comment = re.search(r'"comment"\s*:\s*"([^"]+)"', cleaned)
                m_conf = re.search(r'"confidence"\s*:\s*(\d+)', cleaned)
                data = {
                    "correct": correct_val,
                    "failure_mode": m_fm.group(1) if m_fm else ("ambiguous_case" if not correct_val else None),
                    "comment": m_comment.group(1) if m_comment else "(extracted via regex fallback)",
                    "confidence": int(m_conf.group(1)) if m_conf else 50,
                }
                if self.verbose:
                    print(f"\n    [WARN] Used regex fallback to parse judge output (correct={correct_val})")

        if data is None:
            if self.verbose:
                print(f"\n    [WARN] Judge output not parseable: {llm_output[:300]!r}")
            return {
                "correct": False,
                "failure_mode": "ambiguous_case",
                "comment": f"Judge output not parseable: {llm_output[:120]}",
                "confidence": 0,
            }

        # Validate failure_mode
        if data.get("failure_mode") not in FAILURE_MODES + [None]:
            if self.verbose:
                print(f"\n    [WARN] Unknown failure_mode '{data.get('failure_mode')}' — normalising to ambiguous_case")
            data["failure_mode"] = "ambiguous_case"

        return data

    def close(self):
        """No-op. Exists for interface compatibility with HumanReviewer."""
        pass
