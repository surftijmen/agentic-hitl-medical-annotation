from typing import List, Dict

class HumanReviewer:
    """
    Human-in-the-loop reviewer for annotated medical text.
    Allows a human to verify or correct model outputs.
    """

    def review(self, batch_results: List[Dict]):
        """
        Iterates through model outputs and original text.
        Allows human to confirm or correct the diagnosis.
        """
        reviewed_results = []

        for idx, r in enumerate(batch_results, 1):
            print("="*80)
            print(f"Sample #{idx}")
            text_preview = r.get("raw_output", "No raw output available")
            print(f"RAW OUTPUT (truncated):\n{text_preview[:2000]}\n")
            print(f"Model diagnosis: {r.get('diagnosis')}")
            print(f"Is diagnosis given? {r.get('is_diagnosis_given')}")
            print(f"Confidence: {r.get('confidence_level')}")

            # Ask human for confirmation / correction
            human_diag = input("Confirm or correct diagnosis (type 'none' if absent): ").strip()
            if human_diag:
                r["human_reviewed_diagnosis"] = human_diag
            else:
                r["human_reviewed_diagnosis"] = r.get("diagnosis")

            # Optionally flag errors
            human_flag = input("Any issues with this annotation? (y/N): ").strip().lower()
            r["human_flagged"] = human_flag == "y"

            reviewed_results.append(r)

        print("\nAll samples reviewed.")
        return reviewed_results
