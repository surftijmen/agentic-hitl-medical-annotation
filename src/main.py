import os
import json
from datetime import datetime
from typing import Optional, List, Dict

# Agent and utility imports
from modules.annotator_agent import Annotator, TechniqueConfig
from modules.sampler_agent import DataSampler
from modules.annotation_store import AnnotationStore
from modules.human_reviewer import HumanReviewer
from modules.rag_memory import RAGMemory
from modules.feedback_parser import FeedbackParser
from modules.prompt_agent import PromptAgent
from modules.logger import EventLogger
from performance_evaluator import PerformanceEvaluator

# ================================================================
# EXPERIMENT DEFINITIONS
# Uncomment the ones you want to run, or add your own.
# Each is a separate ablation condition.
# ================================================================

EXPERIMENTS = {
    "baseline": TechniqueConfig(),
    "rag_only": TechniqueConfig(use_rag=True),
    "consistency_only": TechniqueConfig(use_self_consistency=True),
    "rag_plus_consistency": TechniqueConfig(use_rag=True, use_self_consistency=True),
    "full_pipeline": TechniqueConfig(
        use_rag=True,
        use_self_consistency=True,
        use_prompt_patching=True,
        use_feedback_routing=True,
    ),
}

# RAG store is per-experiment so results aren't confounded
RAG_STORE_PATHS = {
    "baseline":             None,   # no RAG
    "rag_only":             "logs/rag_stores/rag_only.txt",
    "consistency_only":     None,   # no RAG
    "rag_plus_consistency": "logs/rag_stores/rag_plus_consistency.txt",
    "full_pipeline":        "logs/rag_stores/full_pipeline.txt",
}

# Accumulated feedback signals persist across runs for prompt patching
SIGNALS_PATH = "logs/accumulated_signals.json"


# ================================================================
# RAG PERSISTENCE
# ================================================================

def load_or_create_rag(store_path: str, display_name: str = "medical-annotation-rag") -> RAGMemory:
    """Load existing RAG store from disk, or create a new one and save its ID."""
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    if os.path.exists(store_path):
        with open(store_path) as f:
            store_name = f.read().strip()
        print(f"  Resuming RAG store: {store_name}")
        return RAGMemory(store_name=store_name)
    else:
        rag = RAGMemory(display_name=display_name)
        with open(store_path, "w") as f:
            f.write(rag.store_name)
        print(f"  Created new RAG store: {rag.store_name}")
        return rag


# ================================================================
# PROMPT PERSISTENCE
# ================================================================

def save_proposed_prompt(proposal: Dict, run_num: int, base_prompt_path: str) -> Optional[str]:
    """
    Apply the instruction patch from a proposal to the base prompt
    and save a valid prompt file. Returns path or None if skipped.
    """
    if proposal.get("status") != "pending_human_review":
        print(f"  Skipping prompt update: {proposal.get('reason', proposal.get('status'))}")
        return None

    patch = proposal.get("instruction_patch", {})
    if not patch or "error" in patch:
        print("  Skipping prompt update: instruction patch missing or invalid")
        return None

    # Load the actual base prompt (not a proposal file)
    with open(base_prompt_path, "r") as f:
        base_prompt = json.load(f)

    if "status" in base_prompt and "instructions" not in base_prompt:
        print("  Skipping prompt update: base prompt file is not a valid prompt")
        return None

    # Apply additions
    new_instructions = list(base_prompt.get("instructions", []))
    for instr in patch.get("instructions_to_add", []):
        if instr not in new_instructions:
            new_instructions.append(instr)

    # Apply refinements
    refinements = {
        r["target_instruction"]: r["refined_version"]
        for r in patch.get("instructions_to_refine", [])
    }
    new_instructions = [refinements.get(i, i) for i in new_instructions]

    # Build new prompt preserving all other fields
    new_prompt = {**base_prompt, "instructions": new_instructions}

    os.makedirs("logs/prompts", exist_ok=True)
    path = f"logs/prompts/v{run_num + 1}_auto.json"
    with open(path, "w") as f:
        json.dump(new_prompt, f, indent=2)
    print(f"  Prompt saved → {path}")
    return path


# ================================================================
# SIGNAL ACCUMULATION (for prompt patching across runs)
# ================================================================

def load_accumulated_signals() -> List[Dict]:
    if os.path.exists(SIGNALS_PATH):
        with open(SIGNALS_PATH) as f:
            return json.load(f)
    return []


def save_accumulated_signals(signals: List[Dict]):
    os.makedirs(os.path.dirname(SIGNALS_PATH), exist_ok=True)
    with open(SIGNALS_PATH, "w") as f:
        json.dump(signals, f, indent=2)


# ================================================================
# SINGLE RUN
# ================================================================

def run_pipeline(
    sample_size: int = 5,
    debug: bool = False,
    prompt_path: str = "prompts/medical_text_prompt.json",
    technique_config: TechniqueConfig = None,
    rag: Optional[RAGMemory] = None,
    experiment_name: str = "unnamed",
):
    """
    Execute one full annotation pipeline run with human-in-the-loop review.

    Args:
        sample_size:        Number of medical notes to process
        debug:              Enable verbose logging
        prompt_path:        Path to the prompt configuration file
        technique_config:   Which techniques are active for this run
        rag:                Shared RAGMemory instance (None if not using RAG)
        experiment_name:    Label for output files

    Returns:
        tuple: (annotations_df, performance_metrics, prompt_proposal)
    """
    config = technique_config or TechniqueConfig()
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    run_id = f"{experiment_name}_{version}"

    print()
    print("=" * 60)
    print(f"PIPELINE RUN  [{experiment_name}]")
    print("=" * 60)
    print()
    print("--- Initializing modules ---")

    # ── 1. Lightweight components ────────────────────────────────
    print("  [1/8] Initializing logger...", end=" ", flush=True)
    logger = EventLogger(run_id, debug=debug)
    print("done")

    print("  [2/8] Initializing storage...", end=" ", flush=True)
    store = AnnotationStore()
    parser = FeedbackParser()
    print("done")

    # ── 2. Data ──────────────────────────────────────────────────
    print("  [3/8] Loading medical data...", end=" ", flush=True)
    sampler = DataSampler(
        notes_path="data/mimiciii_notes.parquet",
        adm_path="data/mimiciii_patients_admissions.parquet",
    )
    print("done")

    # ── 3. Annotator ─────────────────────────────────────────────
    # Instantiate WITHOUT RAG to avoid creating a spurious store,
    # then inject the shared RAG instance if provided.
    print("  [4/8] Initializing AI annotator...", end=" ", flush=True)
    annotator = Annotator(
        prompt_path=prompt_path,
        debug=debug,
        technique_config=TechniqueConfig(
            **{**config.__dict__, "use_rag": False}  # disable internal RAG init
        ),
    )
    if rag is not None:
        annotator.rag = rag
        annotator.config.use_rag = True

    logger.log("RUN_STARTED", {
        "experiment": experiment_name,
        "prompt_version": annotator.prompt_version,
        "sample_size": sample_size,
        "techniques": config.__dict__,
        "rag_store": rag.store_name if rag else None,
    })
    print("done")

    # ── 4. Sample ────────────────────────────────────────────────
    print("  [5/8] Sampling medical notes...", end=" ", flush=True)
    batch = sampler.sample_batch(sample_size)
    print("done")

    # ── 5. Annotate ──────────────────────────────────────────────
    print("  [6/8] Generating annotations...")
    annotation_batch = []

    for idx, (_, row) in enumerate(batch.iterrows(), 1):
        text = Annotator.clean_mimic_note(row["TEXT"])

        logger.log("SAMPLE_SELECTED", {
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
        })

        print(f"    [{idx}/{sample_size}] Processing subject {row['SUBJECT_ID']}...", end=" ", flush=True)
        annotation = annotator.analyze_medical_text(text)
        print(annotation.get("diagnosis", "N/A"))

        logger.log("ANNOTATION_PRODUCED", {
            "diagnosis": annotation["diagnosis"],
            "confidence": annotation["confidence_level"],
            "consistency_score": annotation.get("consistency_score"),
        })

        annotation_batch.append({
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
            "text": text,
            "annotation": annotation,
            "gold": row["DIAGNOSIS"],
        })

    print(f"  All {len(annotation_batch)} annotations generated")

    # ── 6. Human review ──────────────────────────────────────────
    print("  [7/8] Initializing review UI...", end=" ", flush=True)
    reviewer = HumanReviewer()
    print("done")

    print()
    print("=" * 60)
    print("HUMAN REVIEW SESSION")
    print("=" * 60)
    print(f"Please review {len(annotation_batch)} annotations in the UI window")
    print()

    parsed_signals = []

    for idx, item in enumerate(annotation_batch, 1):
        print(f"[{idx}/{len(annotation_batch)}] Waiting for review (Subject: {item['subject_id']})...", flush=True)

        human_fb = reviewer.review(
            annotation=item["annotation"],
            medical_note=item["text"],
            gold=item["gold"],
        )

        logger.log("HUMAN_REVIEW", human_fb)

        store.add(
            subject_id=item["subject_id"],
            hadm_id=item["hadm_id"],
            note_text=item["text"],
            annotation=item["annotation"],
            gold=item["gold"],
            human_feedback=human_fb,
            prompt_version=annotator.prompt_version,
        )

        # Add validated cases to RAG (only if RAG technique is active)
        if rag is not None and config.use_rag:
            rag_record = store.to_rag_record(store.records[-1])
            if rag_record:
                rag.add_case(rag_record["retrieval_text"], rag_record)

        signal = parser.parse(human_fb)
        parsed_signals.append(signal)

    reviewer.close()
    print("  All reviews completed")

    # ── 7. Evaluate ──────────────────────────────────────────────
    print("  [8/8] Analyzing results...")

    evaluator = PerformanceEvaluator()
    metrics = evaluator.evaluate(parsed_signals)
    logger.log("METRICS_COMPUTED", metrics)
    print("  Performance metrics computed")

    # ── Prompt patching (only if technique is active) ────────────
    proposal = None
    if config.use_prompt_patching:
        # Accumulate signals across runs for better signal quality
        all_signals = load_accumulated_signals()
        all_signals.extend(parsed_signals)
        save_accumulated_signals(all_signals)

        print("    Loading LLM for prompt refinement...", end=" ", flush=True)
        prompt_agent = PromptAgent()
        proposal = prompt_agent.propose_update(
            annotator.prompt_dict,
            all_signals,
            min_failures=2,
        )
        logger.log("PROMPT_PROPOSED", proposal)
        print("done")
    else:
        print("  Prompt patching skipped (not in technique config)")

    # ── 8. Persist ───────────────────────────────────────────────
    logger.log_run(
        prompt_version=prompt_path,
        sample_size=sample_size,
        notes=f"Experiment: {experiment_name}",
    )

    df = store.to_dataframe()
    os.makedirs(f"logs/experiments/{experiment_name}", exist_ok=True)
    store_path = f"logs/experiments/{experiment_name}/{run_id}.json"
    df.to_json(store_path, orient="records", indent=2)

    logger.log("RUN_COMPLETED", {
        "total_samples": len(batch),
        "store_path": store_path,
    })

    print()
    print("=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Experiment       : {experiment_name}")
    print(f"Results saved to : {store_path}")
    print(f"Total samples    : {len(df)}")
    print(f"Accuracy         : {metrics.get('accuracy', 'N/A')}")
    print(f"Precision        : {metrics.get('precision', 'N/A')}")
    print()

    return df, metrics, proposal


# ================================================================
# MULTI-RUN (single experiment, improving over time)
# ================================================================

def run_multi_pipeline(
    num_runs: int = 2,
    sample_size: int = 5,
    debug: bool = False,
    initial_prompt_path: str = "prompts/medical_text_prompt.json",
    experiment_name: str = "full_pipeline",
):
    """
    Run one experiment multiple times in sequence.
    Each run benefits from accumulated RAG cases and an improving prompt.
    """
    config = EXPERIMENTS[experiment_name]
    rag_store_path = RAG_STORE_PATHS.get(experiment_name)

    print("=" * 60)
    print(f"MULTI-RUN  [{experiment_name}]  ({num_runs} runs × {sample_size} samples)")
    print("=" * 60)

    rag = load_or_create_rag(rag_store_path) if rag_store_path else None

    prompt_path = initial_prompt_path
    all_metrics = []

    for run_num in range(1, num_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"  RUN {run_num} of {num_runs}")
        print(f"  Prompt : {prompt_path}")
        print(f"  RAG    : {rag.store_name if rag else 'disabled'}")
        print("=" * 60)

        df, metrics, proposal = run_pipeline(
            sample_size=sample_size,
            debug=debug,
            prompt_path=prompt_path,
            technique_config=config,
            rag=rag,
            experiment_name=experiment_name,
        )

        all_metrics.append({"run": run_num, **metrics})

        # Apply improved prompt for next run if patching is active
        if run_num < num_runs and proposal:
            new_path = save_proposed_prompt(proposal, run_num, base_prompt_path=prompt_path)
            if new_path:
                prompt_path = new_path
            else:
                print(f"  Keeping current prompt: {prompt_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"MULTI-RUN SUMMARY  [{experiment_name}]")
    print("=" * 60)
    for m in all_metrics:
        print(f"  Run {m['run']}: accuracy={m.get('accuracy', 'N/A')}  precision={m.get('precision', 'N/A')}")
    print()

    return all_metrics


# ================================================================
# EXPERIMENT RUNNER (ablation study across all techniques)
# ================================================================

def run_ablation_study(
    sample_size: int = 10,
    debug: bool = False,
    prompt_path: str = "logs/prompts/v1_initial.json",
    experiments_to_run: Optional[list] = None,
):
    """
    Run each experimental condition independently and compare results.
    Use this to produce your ablation table.

    Args:
        experiments_to_run: List of experiment names from EXPERIMENTS dict.
                            Defaults to all.
    """
    to_run = experiments_to_run or list(EXPERIMENTS.keys())

    print("=" * 60)
    print(f"ABLATION STUDY  ({len(to_run)} conditions × {sample_size} samples)")
    print("=" * 60)

    results = []

    for name in to_run:
        config = EXPERIMENTS[name]
        rag_store_path = RAG_STORE_PATHS.get(name)
        rag = load_or_create_rag(rag_store_path) if rag_store_path else None

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {name}")
        print(f"Config: {config}")
        print("=" * 60)

        df, metrics, _ = run_pipeline(
            sample_size=sample_size,
            debug=debug,
            prompt_path=prompt_path,
            technique_config=config,
            rag=rag,
            experiment_name=name,
        )

        metrics["experiment"] = name
        results.append(metrics)

    # ── Comparison table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Experiment':<30} {'Accuracy':>10} {'Precision':>10} {'Hallucinations':>16}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['experiment']:<30} "
            f"{str(r.get('accuracy', 'N/A')):>10} "
            f"{str(r.get('precision', 'N/A')):>10} "
            f"{str(r.get('hallucinations', 'N/A')):>16}"
        )
    print()

    # Save summary
    os.makedirs("logs/experiments", exist_ok=True)
    summary_path = f"logs/experiments/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    # ── Option A: single experiment, multiple improving runs ─────
    run_multi_pipeline(
        num_runs=2,
        sample_size=3,
        debug=True,
        initial_prompt_path="logs/prompts/v1_initial.json",
        experiment_name="consistency_only",
    )

    # ── Option B: ablation study across all techniques ───────────
    # run_ablation_study(
    #     sample_size=10,
    #     prompt_path="logs/prompts/v1_initial.json",
    #     experiments_to_run=["baseline", "rag_only", "consistency_only", "full_pipeline"],
    # )