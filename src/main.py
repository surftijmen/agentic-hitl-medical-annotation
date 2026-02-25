import os
from datetime import datetime
from typing import Optional

# Agent and utility imports
from modules.annotator_agent import Annotator
from modules.sampler_agent import DataSampler
from modules.annotation_store import AnnotationStore
from modules.human_reviewer import HumanReviewer
from modules.rag_memory import RAGMemory
from modules.feedback_parser import FeedbackParser
from modules.prompt_agent import PromptAgent
from modules.logger import EventLogger
from performance_evaluator import PerformanceEvaluator

STORE_NAME_PATH = "logs/rag_store_name.txt"

def load_or_create_rag(display_name="medical-annotation-rag"):
    if os.path.exists(STORE_NAME_PATH):
        with open(STORE_NAME_PATH) as f:
            store_name = f.read().strip()
        print(f"  Resuming RAG store: {store_name}")
        return RAGMemory(store_name=store_name)
    else:
        rag = RAGMemory(display_name=display_name)
        with open(STORE_NAME_PATH, "w") as f:
            f.write(rag.store_name)
        return rag


def run_pipeline(sample_size=5, debug=False, prompt_path="prompts/medical_text_prompt.json"):
    """
    Execute the full annotation pipeline with human-in-the-loop review.
    Optimized for faster initialization and reduced waiting times.
    
    Args:
        sample_size: Number of medical notes to process
        debug: Enable verbose logging
        prompt_path: Path to the prompt configuration file
        
    Returns:
        tuple: (annotations_df, performance_metrics, prompt_proposal)
    """
    # ============================================================
    # 1. INITIALIZATION (with progress feedback)
    # ============================================================
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    run_id = f"run_{version}"
    
    print("=" * 60)
    print("MEDICAL ANNOTATION PIPELINE")
    print("=" * 60)
    print()
    print("--- Initializing modules ---")

    # Initialize lightweight components first
    print("  [1/8] Initializing logger...", end=" ")
    logger = EventLogger(run_id, debug=debug)
    
    print("  [2/8] Initializing storage...", end=" ")
    store = AnnotationStore()
    parser = FeedbackParser()
    
    # Load data (can be slow for large parquet files)
    print("  [3/8] Loading medical data...", end=" ", flush=True)
    sampler = DataSampler(
        notes_path="data/mimiciii_notes.parquet",
        adm_path="data/mimiciii_patients_admissions.parquet",
    )
    
    # Initialize AI annotator (RAG + API client)
    print("  [4/8] Initializing AI annotator (this may take a moment)...", end=" ", flush=True)
    annotator = Annotator(prompt_path=prompt_path, debug=debug, use_rag=True)
    
    # Log pipeline start
    logger.log("RUN_STARTED", {
        "prompt_version": annotator.prompt_version,
        "sample_size": sample_size
    })
    
    # ============================================================
    # 2. DATA SAMPLING
    # ============================================================
    print("  [5/8] Sampling medical notes...", end=" ")
    batch = sampler.sample_batch(sample_size)
    
    # ============================================================
    # 3. BATCH ANNOTATION (ALL AT ONCE)
    # ============================================================
    print("  [6/8] Generating annotations...")
    annotation_batch = []
    
    for idx, (_, row) in enumerate(batch.iterrows(), 1):
        # Clean and prepare medical note text
        text = Annotator.clean_mimic_note(row["TEXT"])
        
        logger.log("SAMPLE_SELECTED", {
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"]
        })
        
        print(f"    [{idx}/{sample_size}] Processing subject {row['SUBJECT_ID']}...", end=" ", flush=True)
        
        # Generate AI annotation
        annotation = annotator.analyze_medical_text(text)
        
        print(f"{annotation.get('diagnosis', 'N/A')}")
        
        logger.log("ANNOTATION_PRODUCED", {
            "diagnosis": annotation["diagnosis"],
            "confidence": annotation["confidence_level"]
        })
        
        # Store annotation data for review
        annotation_batch.append({
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
            "text": text,
            "annotation": annotation,
            "gold": row["DIAGNOSIS"]
        })
    
    print(f"  All {len(annotation_batch)} annotations generated")
    
    # ============================================================
    # 4. HUMAN REVIEW (LAZY UI INITIALIZATION)
    # ============================================================
    print("  [7/8] Initializing review UI...", end=" ", flush=True)
    reviewer = HumanReviewer()  # Initialize UI only when needed
    
    print()
    print("=" * 60)
    print("HUMAN REVIEW SESSION")
    print("=" * 60)
    print(f"Please review {len(annotation_batch)} annotations in the UI window")
    print()
    
    parsed_signals = []
    
    for idx, item in enumerate(annotation_batch, 1):
        print(f"[{idx}/{len(annotation_batch)}] Waiting for review (Subject: {item['subject_id']})...", flush=True)
        
        # Human-in-the-loop review (interactive UI)
        human_fb = reviewer.review(
            annotation=item["annotation"],
            medical_note=item["text"],
            gold=item["gold"],
        )
        
        logger.log("HUMAN_REVIEW", human_fb)
        
        # Store complete annotation record
        store.add(
            subject_id=item["subject_id"],
            hadm_id=item["hadm_id"],
            note_text=item["text"],
            annotation=item["annotation"],
            gold=item["gold"],
            human_feedback=human_fb,
            prompt_version=annotator.prompt_version,
        )

        # Store validated case in RAG if correct
        rag_record = store.to_rag_record(store.records[-1])
        if rag_record:
            annotator.rag.add_case(
                rag_record["retrieval_text"],
                rag_record
            )
        
        # Parse feedback for performance evaluation
        parsed_signals.append(parser.parse(human_fb))
    
    # Close the review UI after all samples processed
    reviewer.close()
    print(" All reviews completed")
    
    # ============================================================
    # 5. POST-PROCESSING (LAZY INITIALIZATION)
    # ============================================================
    print("  [8/8] Analyzing results...")
    
    # Initialize evaluator only when needed
    evaluator = PerformanceEvaluator()
    metrics = evaluator.evaluate(parsed_signals)
    logger.log("METRICS_COMPUTED", metrics)
    print("  Performance metrics computed")
    
    # Initialize prompt agent only when needed (may load LLM)
    print("    Loading LLM for prompt refinement...", end=" ", flush=True)
    prompt_agent = PromptAgent()
    
    proposal = prompt_agent.propose_update(
        annotator.prompt_dict,
        parsed_signals
    )
    
    logger.log("PROMPT_PROPOSED", proposal)
    print(" Prompt improvement proposal generated")
    
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
    
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {store_path}")
    print(f"Total samples: {len(df)}")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print()
    
    return df, metrics, proposal


if __name__ == "__main__":
    # Example execution
    df, metrics, proposal = run_pipeline(
        sample_size=3,
        debug=True,
        prompt_path="logs/prompts/v1_initial.json",
    )