from modules.annotator_agent import Annotator
from modules.sampler_agent import DataSampler
from modules.annotation_store import AnnotationStore
from modules.human_reviewer import HumanReviewer

from modules.feedback_parser import FeedbackParser
from performance_evaluator import PerformanceEvaluator
from modules.prompt_agent import PromptAgent

from modules.logger import EventLogger
import uuid
import os 

def run_pipeline(sample_size=5, debug=False, prompt_path="prompts/medical_text_prompt.json"):
    run_id = f"run_{uuid.uuid4().hex[:6]}"
    sampler = DataSampler(
        notes_path="data/mimiciii_notes.parquet",
        adm_path="data/mimiciii_patients_admissions.parquet",
    )
    annotator = Annotator(
        prompt_path=prompt_path,
        debug=debug,
    )
    store = AnnotationStore()

    reviewer = HumanReviewer()          # persistent UI
    parser = FeedbackParser()
    evaluator = PerformanceEvaluator()
    prompt_agent = PromptAgent()
    logger = EventLogger(run_id, debug=debug)

    logger.log("RUN_STARTED", {
    "prompt_version": annotator.prompt_version,
    "sample_size": sample_size})

    batch = sampler.sample_batch(sample_size)
    parsed_signals = []

    for _, row in batch.iterrows():
        text = Annotator.clean_mimic_note(row["TEXT"])

        logger.log("SAMPLE_SELECTED", {
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"]
        })
            
        annotation = annotator.analyze_medical_text(text)

        logger.log("ANNOTATION_PRODUCED", {
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
            "diagnosis": annotation["diagnosis"],
            "confidence": annotation["confidence_level"]
        })

        # HUMAN-IN-THE-LOOP (interactive, persistent UI)
        human_fb = reviewer.review(
            annotation=annotation,
            medical_note=text,
            gold=row["DIAGNOSIS"],  
        )

        logger.log("HUMAN_REVIEW", human_fb)

        store.add(
        subject_id=row["SUBJECT_ID"],
        hadm_id=row["HADM_ID"],
        note_text=text,
        annotation=annotation,
        gold=row["DIAGNOSIS"],
        human_feedback=human_fb,
        prompt_version=annotator.prompt_version,
        )

        parsed_signals.append(parser.parse(human_fb))

    # close UI once, after all reviews
    reviewer.close()

    metrics = evaluator.evaluate(parsed_signals)
    logger.log("METRICS_COMPUTED", metrics)


    annotator.prompt_dict = prompt_agent.propose_update(
        annotator.prompt_dict,
        metrics,
    )

    proposal = prompt_agent.propose_update(
        annotator.prompt_dict,
        metrics,
        reason="High symptom vs diagnosis confusion",
    )
    logger.log("PROMPT_PROPOSED", proposal)


    logger.log_run(
        prompt_version=prompt_path,
        sample_size=sample_size,
        notes="Demo run",
    )

    df = store.to_dataframe()

    os.makedirs("logs/stores", exist_ok=True)
    store_path = f"logs/stores/{run_id}.json"
    df.to_json(
        store_path,
        orient="records",
        indent=2
    )

    logger.log("RUN_COMPLETED", {
    "total_samples": len(batch),
    })

    return store.to_dataframe(), metrics, proposal    


if __name__ == "__main__":
    df, metrics, proposal = run_pipeline(
        sample_size=1,
        debug=True,
        prompt_path="logs/prompts/v1_initial.json",
    )