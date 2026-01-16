from modules.annotator_agent import Annotator
from modules.sampler_agent import DataSampler
from modules.annotation_store import AnnotationStore
from modules.human_reviewer import HumanReviewer

from modules.feedback_parser import FeedbackParser
from performance_evaluator import PerformanceEvaluator
from modules.prompt_agent import PromptAgent


def run_pipeline(sample_size=5, debug=False):
    sampler = DataSampler(
        notes_path="data/mimiciii_notes.parquet",
        adm_path="data/mimiciii_patients_admissions.parquet",
    )
    annotator = Annotator(debug=debug)
    store = AnnotationStore()

    reviewer = HumanReviewer()          # persistent UI
    parser = FeedbackParser()
    evaluator = PerformanceEvaluator()
    prompt_agent = PromptAgent()

    batch = sampler.sample_batch(sample_size)
    parsed_signals = []

    for _, row in batch.iterrows():
        text = Annotator.clean_mimic_note(row["TEXT"])
        annotation = annotator.analyze_medical_text(text)

        store.add(
            row["SUBJECT_ID"],
            row["HADM_ID"],
            annotation,
            gold=row["DIAGNOSIS"],
        )

        # HUMAN-IN-THE-LOOP (interactive, persistent UI)
        human_fb = reviewer.review(
            annotation=annotation,
            medical_note=text,
            gold=row["DIAGNOSIS"],   # optional reference
        )

        parsed_signals.append(parser.parse(human_fb))

    # close UI once, after all reviews
    reviewer.close()

    metrics = evaluator.evaluate(parsed_signals)

    annotator.prompt_dict = prompt_agent.propose_update(
        annotator.prompt_dict,
        metrics,
    )

    return store.to_dataframe(), metrics


if __name__ == "__main__":
    df, feedback = run_pipeline(sample_size=3, debug=False)
