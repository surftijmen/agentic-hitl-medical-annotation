import json
import re
from typing import Optional, List, Dict
import pandas as pd
import toml
from google import genai
from google.genai import types


class Annotator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        mimic_notes_path: str = "data/mimiciii_notes.parquet",
        mimic_adm_path: str = "data/mimiciii_patients_admissions.parquet",
        prompt_path: str = "logs/prompts/v1_initial.json",
        secrets_path: str = "./secrets.toml",
        debug: bool = False,
    ):
        self.model_name = model_name
        self.mimic_notes_path = mimic_notes_path
        self.mimic_adm_path = mimic_adm_path
        self.prompt_path = prompt_path
        self.secrets_path = secrets_path
        self.debug = debug

        self.prompt_version = prompt_path
        self.prompt_dict = self.load_prompt_dict()
        self._load_client()

    # =========================
    # INTERNAL UTIL
    # =========================

    def _log(self, *args):
        if self.debug:
            print(*args)

    # =========================
    # SETUP
    # =========================

    def _load_client(self):
        secrets = toml.load(self.secrets_path)
        self.client = genai.Client(api_key=secrets["api"]["key"])

    # =========================
    # PROMPT
    # =========================

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

    # =========================
    # TEXT CLEANING
    # =========================

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

    # =========================
    # OUTPUT PARSING
    # =========================

    def extract_results(self, llm_output: str) -> Dict:
        self._log("\n===== MODEL RAW OUTPUT =====\n")
        self._log(llm_output)

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

    # =========================
    # GEMINI INFERENCE
    # =========================

    def analyze_medical_text(self, medical_text: str) -> Dict:
        user_prompt = self.build_user_prompt(input_text=medical_text)

        config = types.GenerateContentConfig(
            system_instruction=self.prompt_dict.get("system_prompt", ""),
            temperature=0.0,
            max_output_tokens=3000,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        return self.extract_results(response.text)

    # =========================
    # SAMPLING / TESTING
    # =========================

    def random_mimic_test(self, n: int = 1):
        samples = self.mimic_df.sample(n)

        for _, row in samples.iterrows():
            clean_text = self.clean_mimic_note(row["TEXT"])

            self._log("=" * 80)
            self._log(
                f"SUBJECT_ID: {row['SUBJECT_ID']} | HADM_ID: {row['HADM_ID']}"
            )
            self._log("\nCLEANED NOTE (TRUNCATED):\n")
            self._log(clean_text[:2000])

            result = self.analyze_medical_text(clean_text)

            self._log(result)

            print("\n===== FINAL INFORMATION =====")
            print(f"Agents diagnosis: {result.get('diagnosis')}")
            print("Actual diagnosis:", row["DIAGNOSIS"])


            if "error" in result:
                self._log(f"Error: {result['error']}")

        return (result.get('diagnosis'), row["DIAGNOSIS"])


if __name__ == "__main__":
    pipeline = Annotator(debug=False)
    pipeline.random_mimic_test(n=1)
