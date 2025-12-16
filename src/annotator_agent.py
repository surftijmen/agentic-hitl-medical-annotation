import json
import re
from typing import Optional, List, Dict
from os.path import join
import pandas as pd
import toml
from google import genai
from google.genai import types

# =========================
# LOAD DATA AND CONFIG
# =========================

data = toml.load("./secrets.toml")
GEMINI_API_KEY = data["api"]["key"]

MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

train_df = pd.read_csv("data/train.dat", sep="\t", header=None, names=["label", "text"])


# =========================
# PROMPT LOADING
# =========================

def load_prompt_dict(path: str = join("prompts", "medical_text_prompt.json")) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# PROMPT BUILDING
# =========================

def build_user_prompt(
    prompt_dict: Dict,
    input_text: Optional[str] = None,
    examples: Optional[List[str]] = None,
) -> str:
    parts = []

    if prompt_dict.get("query"):
        parts.append(prompt_dict["query"])

    if prompt_dict.get("instructions"):
        parts.append("Instructions:")
        for inst in prompt_dict["instructions"]:
            parts.append(f"- {inst}")

    if examples:
        parts.append(
            "Here are some examples for text that do or do not contain "
            "explicit rheumatic disease diagnosis:"
        )
        parts.extend(examples)

    if prompt_dict.get("format_instruction"):
        parts.append(prompt_dict["format_instruction"])

    if input_text:
        parts.append(f"Medical text: {input_text}")

    return "\n\n".join(parts)


# =========================
# OUTPUT PARSING
# =========================

def extract_results(llm_output: str) -> Dict:
    """
    Extract JSON from model output robustly
    """
    print("\n\n===== MODEL OUTPUT =====\n")
    print(llm_output)

    # Use regex to capture JSON object
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

def analyze_medical_text(
    medical_text: str,
    prompt_dict: Dict,
    model_name: str = MODEL_NAME,
) -> Dict:
    user_prompt = build_user_prompt(
        prompt_dict=prompt_dict,
        input_text=medical_text,
    )

    config = types.GenerateContentConfig(
        system_instruction=prompt_dict.get("system_prompt", ""),
        temperature=0.0,
        max_output_tokens=512,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=config,
    )

    return extract_results(response.text)


# =========================
# MAIN FUNCTION
# =========================

def main():
    prompt_dict = load_prompt_dict()

    sample_text = (
        "1 . oligo - artritis van beide polsen bij bekende seropositieve ra , "
        "dd activiteit ra , inflammatoire ( secundaire ) artrose"
    )

    print("=" * 80)
    print("Medical Text Extraction with Gemini (LATEST SDK)")
    print("=" * 80)
    print(f"\nAnalyzing medical text:\n{sample_text}\n")

    result = analyze_medical_text(sample_text, prompt_dict)

    print("Extraction Results:")
    print("-" * 80)
    print(f"Diagnosis: {result.get('diagnosis')}")
    print(f"Is Diagnosis Given: {result.get('is_diagnosis_given')}")
    print(f"Confidence Level: {result.get('confidence_level')}")

    if "error" in result:
        print(f"\nError: {result['error']}")


# =========================
# RANDOM TEST FUNCTION
# =========================

def random_test():
    prompt_dict = load_prompt_dict("prompts/medical_text_prompt.json")

    LABEL_NAMES = {
        1: "Digestive System Diseases",
        2: "Cardiovascular Diseases",
        3: "Neoplasms",
        4: "Nervous System Diseases",
        5: "General Pathological Conditions",
    }

    # Sample one row (text + label)
    sample = train_df.sample(1).iloc[0]
    sample_text = sample["text"]
    true_label_id = sample["label"]
    true_label_name = LABEL_NAMES.get(true_label_id, "Unknown")

    print("=" * 80)
    print("Sample text:")
    print(sample_text)

    # Run model
    result = analyze_medical_text(
        medical_text=sample_text,
        prompt_dict=prompt_dict
    )

    print("\n\n===== CORRECT ANSWER =====\n")
    print(f"{true_label_id} -> {true_label_name}")

    print("\n\n===== MODEL PREDICTION =====\n")
    print(f"Diagnosis: {result.get('diagnosis')}")
    print(f"Is Diagnosis Given: {result.get('is_diagnosis_given')}")
    print(f"Confidence Level: {result.get('confidence_level')}")


if __name__ == "__main__":
    #main()
    random_test()
