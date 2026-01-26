"""
Medical Text Annotation Pipeline
Orchestrates sampling, annotation, human review, and prompt improvement.
"""

import os
import uuid

# Agent and utility imports
from modules.annotator_agent import Annotator
from modules.sampler_agent import DataSampler
from modules.annotation_store import AnnotationStore
from modules.human_reviewer import HumanReviewer
from modules.feedback_parser import FeedbackParser
from modules.prompt_agent import PromptAgent
from modules.logger import EventLogger
from performance_evaluator import PerformanceEvaluator


def run_pipeline(sample_size=5, debug=False, prompt_path="prompts/medical_text_prompt.json"):
    """
    Execute the full annotation pipeline with human-in-the-loop review.
    
    Args:
        sample_size: Number of medical notes to process
        debug: Enable verbose logging
        prompt_path: Path to the prompt configuration file
        
    Returns:
        tuple: (annotations_df, performance_metrics, prompt_proposal)
    """
    # ============================================================
    # 1. INITIALIZATION
    # ============================================================
    run_id = f"run_{uuid.uuid4().hex[:6]}"
    
    # Initialize all pipeline components
    sampler = DataSampler(
        notes_path="data/mimiciii_notes.parquet",
        adm_path="data/mimiciii_patients_admissions.parquet",
    )
    annotator = Annotator(prompt_path=prompt_path, debug=debug)
    store = AnnotationStore()
    reviewer = HumanReviewer()  # Persistent UI for human feedback
    parser = FeedbackParser()
    evaluator = PerformanceEvaluator()
    prompt_agent = PromptAgent()
    logger = EventLogger(run_id, debug=debug)
    
    # Log pipeline start
    logger.log("RUN_STARTED", {
        "prompt_version": annotator.prompt_version,
        "sample_size": sample_size
    })
    
    # ============================================================
    # 2. DATA SAMPLING
    # ============================================================
    batch = sampler.sample_batch(sample_size)
    
    # ============================================================
    # 3. ANNOTATION & REVIEW LOOP
    # ============================================================
    parsed_signals = []
    
    for _, row in batch.iterrows():
        # Clean and prepare medical note text
        text = Annotator.clean_mimic_note(row["TEXT"])
        
        logger.log("SAMPLE_SELECTED", {
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"]
        })
        
        # Generate AI annotation
        annotation = annotator.analyze_medical_text(text)
        
        logger.log("ANNOTATION_PRODUCED", {
            "diagnosis": annotation["diagnosis"],
            "confidence": annotation["confidence_level"]
        })
        
        # Human-in-the-loop review (interactive UI)
        human_fb = reviewer.review(
            annotation=annotation,
            medical_note=text,
            gold=row["DIAGNOSIS"],
        )
        
        logger.log("HUMAN_REVIEW", human_fb)
        
        # Store complete annotation record
        store.add(
            subject_id=row["SUBJECT_ID"],
            hadm_id=row["HADM_ID"],
            note_text=text,
            annotation=annotation,
            gold=row["DIAGNOSIS"],
            human_feedback=human_fb,
            prompt_version=annotator.prompt_version,
        )
        
        # Parse feedback for performance evaluation
        parsed_signals.append(parser.parse(human_fb))
    
    # Close the review UI after all samples processed
    reviewer.close()
    
    # ============================================================
    # 4. PERFORMANCE EVALUATION
    # ============================================================
    metrics = evaluator.evaluate(parsed_signals)
    logger.log("METRICS_COMPUTED", metrics)
    
    # ============================================================
    # 5. PROMPT IMPROVEMENT PROPOSAL
    # ============================================================
    proposal = prompt_agent.propose_update(
        annotator.prompt_dict,
        metrics,
        reason="High symptom vs diagnosis confusion",
    )
    
    logger.log("PROMPT_PROPOSED", proposal)
    
    # ============================================================
    # 6. PERSIST RESULTS
    # ============================================================
    logger.log_run(
        prompt_version=prompt_path,
        sample_size=sample_size,
        notes="Demo run",
    )
    
    # Save annotation store to disk
    df = store.to_dataframe()
    os.makedirs("logs/stores", exist_ok=True)
    store_path = f"logs/stores/{run_id}.json"
    df.to_json(store_path, orient="records", indent=2)
    
    logger.log("RUN_COMPLETED", {
        "total_samples": len(batch),
        "store_path": store_path
    })
    
    return df, metrics, proposal


if __name__ == "__main__":
    # Example execution with single sample for testing
    df, metrics, proposal = run_pipeline(
        sample_size=1,
        debug=True,
        prompt_path="logs/prompts/v1_initial.json",
    )