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
import random
import re
import time
import toml
from typing import Any, Dict, Optional
from google import genai
from google.genai import types


FAILURE_MODES = [
    "hallucination",
    "missed_entity",
    "terminology_gap",
    "unsupported_inference",
    "symptom_diagnosis_confusion",
    "ambiguous_case",
]


# Gemini response schema — forces structured output so malformed JSON is
# impossible and the regex fallback in _parse_judgment is never needed.
# `failure_mode` is omitted from `required` because it should be null when
# the verdict is correct; the prompt still tells the model to fill it when
# the verdict is incorrect.
_JUDGMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "correct": {"type": "BOOLEAN"},
        "failure_mode": {
            "type": "STRING",
            "enum": FAILURE_MODES + ["none"],
        },
        "comment": {"type": "STRING"},
        "confidence": {"type": "INTEGER"},
    },
    "required": ["correct", "comment", "confidence", "failure_mode"],
    "propertyOrdering": ["correct", "failure_mode", "comment", "confidence"],
}

_SYSTEM_PROMPT = """\
You are a senior clinician evaluating an LLM medical-text annotation system.
Your task: decide whether the annotator's extracted diagnosis is semantically
consistent with the billed PRIMARY ICD-9 diagnosis for this admission.

━━━ INPUTS YOU WILL RECEIVE ━━━
• MEDICAL NOTE — the (possibly redacted) discharge summary the annotator saw.
• ANNOTATOR OUTPUT — a JSON with the annotator's free-text `diagnosis`,
  plus `confidence_level` and (optionally) chain-of-thought `reasoning`.
• GOLD PRIMARY ICD-9 — the single billed primary diagnosis for this
  admission, shown as an ICD-9 code together with its SHORT_TITLE and
  LONG_TITLE (the human-readable description). Use the LONG_TITLE as the
  primary signal; SHORT_TITLE is a disambiguation hint.

━━━ WHAT COUNTS AS CORRECT ━━━
The annotator's `diagnosis` is CORRECT when it refers to the SAME clinical
concept as the gold ICD-9 LONG_TITLE, given what the note says. Concretely:
• Accept standard synonyms and abbreviations
  (e.g. "MI" ≈ "Acute myocardial infarction of other anterior wall";
        "CHF" ≈ "Congestive heart failure, unspecified";
        "UTI" ≈ "Urinary tract infection, site not specified";
        "AKI" ≈ "Acute kidney failure, unspecified";
        "ESRD" ≈ "End stage renal disease";
        "ICH" / "SAH" / "SDH" ≈ any intracranial hemorrhage code).
• Accept a MORE-SPECIFIC annotator answer that is a strict subtype of the
  ICD concept (e.g. "STEMI" when gold is "Acute myocardial infarction,
  unspecified"; "Paroxysmal atrial fibrillation" when gold is "Atrial
  fibrillation"; "Septic shock" when gold is "Severe sepsis").
• Accept a MORE-GENERAL annotator answer only when the gold is a specific
  instance of the general concept AND the note does not contradict it
  (e.g. annotator: "Pneumonia", gold: "Pneumonia, organism unspecified").
• Accept the primary CAUSE when the ICD is a direct complication
  (e.g. annotator: "MI", gold: "Cardiogenic shock" following acute MI
  that the note clearly describes).

━━━ WHAT COUNTS AS INCORRECT ━━━
• A different organ system or clinically distinct condition
  (e.g. annotator: "Pneumonia", gold: "Acute kidney failure"), even if the
  note mentions both — primary means primary.
• A symptom or sign instead of a disease
  (e.g. annotator: "Chest pain", gold: "Acute MI"; annotator: "Hypoxia",
  gold: "Pneumonia").
• A diagnosis that the note does not support at all (hallucination), even
  if it happens to be close to the gold title.
• An unsupported inference from ambiguous findings
  (e.g. annotator: "Sepsis" when note only says "presumed infection,
  started empiric antibiotics" with no confirmed sepsis).

━━━ AUTHORITY OF THE GOLD LABEL ━━━
Gold is the billed primary ICD-9 diagnosis — strong evidence, but not
absolute truth. Always cross-check against what is EXPLICITLY written in
the note. If the note clearly contradicts the gold (rare but possible),
trust the note and mark the annotator correct/incorrect accordingly —
flag this in `comment`.

━━━ FAILURE MODE GUIDE (pick the MOST SPECIFIC one) ━━━
• hallucination              — diagnosis is not supported by anything in the note
• missed_entity              — a clearly confirmed diagnosis was ignored (annotator said "none" or picked something else while the correct concept is plainly stated)
• terminology_gap            — correct concept, wrong or non-clinical terminology that obscures the match
• unsupported_inference      — annotator inferred/assumed a diagnosis not explicitly stated
• symptom_diagnosis_confusion — annotator returned a symptom or sign, not a diagnosis
• ambiguous_case             — ONLY when it is genuinely impossible to judge (note too short/corrupted; gold title is too vague such as "Other specified disorder")

━━━ OUTPUT FORMAT ━━━
Respond with ONLY a JSON object matching the enforced schema — no prose, no markdown.
Fields:
  "correct"       — boolean.
  "failure_mode"  — one of: "hallucination", "missed_entity", "terminology_gap",
                    "unsupported_inference", "symptom_diagnosis_confusion",
                    "ambiguous_case", or "none" if correct=true.
  "comment"       — one sentence explaining the verdict.
  "confidence"    — integer 0-100.
"""


def _format_gold(gold: Any) -> str:
    """Render the gold label section of the judge prompt.

    Accepts either a dict {icd9_code, short_title, long_title} (the current
    pipeline shape) or a bare string (defensive fallback for old callers).
    """
    if gold is None:
        return "GOLD PRIMARY ICD-9 DIAGNOSIS: not available"
    if isinstance(gold, dict):
        icd   = gold.get("icd9_code", "?")
        short = gold.get("short_title", "")
        long_ = gold.get("long_title", "")
        return (
            "GOLD PRIMARY ICD-9 DIAGNOSIS (billed, MIMIC-III):\n"
            f"  code:        {icd}\n"
            f"  long_title:  {long_}\n"
            f"  short_title: {short}"
        )
    return f"GOLD PRIMARY ICD-9 DIAGNOSIS: {gold}"


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
        gold: Optional[Any] = None,
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

        max_retries = 6
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.judge_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.0,       # deterministic judgment
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                        response_schema=_JUDGMENT_SCHEMA,
                    ),
                )
                break  # success
            except Exception as e:
                err_str = str(e)
                is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str
                if is_transient and attempt < max_retries - 1:
                    # Parse suggested retry delay from the API response if present,
                    # otherwise use exponential backoff. Always add jitter so threads
                    # that hit the rate limit together don't all retry simultaneously.
                    suggested = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)s", err_str)
                    base_wait = float(suggested.group(1)) if suggested else 10 * (2 ** attempt)
                    wait = base_wait + random.uniform(1, 5)
                    if self.verbose:
                        print(f"\n    [retry {attempt+1}/{max_retries-1}] rate limit, waiting {wait:.1f}s...", flush=True)
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
        gold: Optional[Any],
    ) -> str:
        parts = []

        parts.append(f"MEDICAL NOTE:\n{medical_note}")

        ann_summary = {
            "diagnosis": annotation.get("diagnosis"),
            "is_diagnosis_given": annotation.get("is_diagnosis_given"),
            "confidence_level": annotation.get("confidence_level"),
        }
        # Include CoT reasoning if it was generated — the judge can use it
        if annotation.get("reasoning"):
            ann_summary["reasoning"] = annotation["reasoning"]

        parts.append(f"ANNOTATOR OUTPUT:\n{json.dumps(ann_summary, indent=2)}")

        parts.append(_format_gold(gold))

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

        # Normalise failure_mode: schema allows "none" string when correct=true;
        # downstream code expects None.
        fm = data.get("failure_mode")
        if fm == "none" or (data.get("correct") is True and fm in (None, "none")):
            data["failure_mode"] = None
        elif fm not in FAILURE_MODES + [None]:
            if self.verbose:
                print(f"\n    [WARN] Unknown failure_mode '{fm}' — normalising to ambiguous_case")
            data["failure_mode"] = "ambiguous_case"

        return data

    def close(self):
        """No-op. Exists for interface compatibility with HumanReviewer."""
        pass
