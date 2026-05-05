import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

# Agent and utility imports
from modules.annotator_agent import Annotator, TechniqueConfig
from modules.sampler_agent import DataSampler
from modules.annotation_store import AnnotationStore
from modules.human_reviewer import HumanReviewer
from modules.auto_reviewer import AutoReviewer
from modules.rag_memory import RAGMemory
from modules.feedback_parser import FeedbackParser
from modules.prompt_agent import PromptAgent
from modules.logger import EventLogger
from performance_evaluator import PerformanceEvaluator


# ================================================================
# EXPERIMENT DEFINITIONS
#
# Each entry is one ablation condition — a specific combination of
# prompt engineering techniques. Running the same condition for
# multiple runs shows the learning trajectory (for the research Q).
#
# Technique flags:
#   use_rag              — RAG grounding via Google File Search
#   use_self_consistency — majority vote over N samples
#   chain_of_thoughts    — step-by-step reasoning before output
#   use_few_shot         — static few-shot examples in prompt
#   use_prompt_patching  — LLM-proposed instruction patches (between runs)
#   use_feedback_routing — route failures to rag vs prompt_patch
# ================================================================

EXPERIMENTS = {
    # ── Single techniques (SubQ2: which benefits most from HITL?) ──
    "baseline": TechniqueConfig(),
    "rag_only": TechniqueConfig(use_rag=True),
    "consistency_only": TechniqueConfig(use_self_consistency=True),
    "cot_only": TechniqueConfig(chain_of_thoughts=True),
    "few_shot_only": TechniqueConfig(use_few_shot=True),

    # ── Combinations ──────────────────────────────────────────────
    "rag_plus_consistency": TechniqueConfig(use_rag=True, use_self_consistency=True),
    "cot_plus_few_shot": TechniqueConfig(chain_of_thoughts=True, use_few_shot=True),
    "rag_plus_cot": TechniqueConfig(use_rag=True, chain_of_thoughts=True),

    # ── HITL learning curves (SubQ1: which feedback types matter?) ──
    # Same as single-technique conditions but with HITL feedback loop enabled.
    # Run multi-run to observe the learning trajectory per technique.
    "baseline_hitl": TechniqueConfig(use_prompt_patching=True, use_feedback_routing=True),
    "few_shot_hitl": TechniqueConfig(use_few_shot=True, use_prompt_patching=True, use_feedback_routing=True),
    "cot_hitl": TechniqueConfig(chain_of_thoughts=True, use_prompt_patching=True, use_feedback_routing=True),
    "cot_plus_few_shot_hitl": TechniqueConfig(chain_of_thoughts=True, use_few_shot=True, use_prompt_patching=True, use_feedback_routing=True),

    # ── Full pipeline (all techniques) ────────────────────────────
    "full_pipeline": TechniqueConfig(
        use_rag=True,
        use_self_consistency=True,
        chain_of_thoughts=True,
        use_few_shot=True,
        use_prompt_patching=True,
        use_feedback_routing=True,
    ),
}

# RAG store is per-experiment so results aren't confounded across conditions
RAG_STORE_PATHS = {
    "baseline":            None,
    "rag_only":            "logs/rag_stores/rag_only.txt",
    "consistency_only":    None,
    "cot_only":            None,
    "few_shot_only":       None,
    "rag_plus_consistency":     "logs/rag_stores/rag_plus_consistency.txt",
    "cot_plus_few_shot":        None,
    "rag_plus_cot":             "logs/rag_stores/rag_plus_cot.txt",
    "baseline_hitl":            None,
    "few_shot_hitl":            None,
    "cot_hitl":                 None,
    "cot_plus_few_shot_hitl":   None,
    "full_pipeline":            "logs/rag_stores/full_pipeline.txt",
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

    # propose_update saves the full proposal to disk and returns only the path —
    # load the full data from there to get instruction_patch.
    proposal_path = proposal.get("path")
    if proposal_path and os.path.exists(proposal_path):
        with open(proposal_path) as f:
            full_proposal = json.load(f)
        patch = full_proposal.get("instruction_patch", {})
    else:
        patch = proposal.get("instruction_patch", {})

    if not patch or "error" in patch:
        print("  Skipping prompt update: instruction patch missing or invalid")
        return None

    with open(base_prompt_path, "r") as f:
        base_prompt = json.load(f)

    if "status" in base_prompt and "instructions" not in base_prompt:
        print("  Skipping prompt update: base prompt file is not a valid prompt")
        return None

    new_instructions = list(base_prompt.get("instructions", []))
    for instr in patch.get("instructions_to_add", []):
        if instr not in new_instructions:
            new_instructions.append(instr)

    refinements = {
        r["target_instruction"]: r["refined_version"]
        for r in patch.get("instructions_to_refine", [])
    }
    new_instructions = [refinements.get(i, i) for i in new_instructions]

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
    prompt_path: str = "logs/prompts/v1_initial.json",
    technique_config: TechniqueConfig = None,
    rag: Optional[RAGMemory] = None,
    experiment_name: str = "unnamed",
    run_number: int = 1,
    reviewer_mode: str = "human",
    judge_model: str = "gemini-2.5-flash",
    max_workers: int = 5,
    sampler: Optional[DataSampler] = None,
    annotator_model: str = "gemini-2.5-flash",
):
    """
    Execute one full annotation pipeline run with human-in-the-loop review.

    Args:
        reviewer_mode:  "human" — interactive Tkinter UI (default)
                        "auto"  — LLM-as-judge using a stronger Gemini model
        judge_model:    Which model to use when reviewer_mode="auto".
                        Default is gemini-2.5-pro for better judgment quality.

    Returns:
        tuple: (annotations_df, performance_metrics, prompt_proposal)
    """
    config = technique_config or TechniqueConfig()
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    run_id = f"{experiment_name}_{version}"

    print()
    print("=" * 60)
    print(f"PIPELINE RUN  [{experiment_name}]  (run {run_number})")
    print("=" * 60)
    active = [k for k, v in config.__dict__.items() if v is True]
    print(f"Active techniques : {', '.join(active) if active else 'none (baseline)'}")
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
    if sampler is None:
        sampler = DataSampler()
    print("done")

    # ── 3. Annotator ─────────────────────────────────────────────
    print(f"  [4/8] Initializing AI annotator ({annotator_model})...", end=" ", flush=True)
    annotator = Annotator(
        model_name=annotator_model,
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
        "run_number": run_number,
        "prompt_version": annotator.prompt_version,
        "sample_size": sample_size,
        "techniques": config.__dict__,
        "rag_store": rag.store_name if rag else None,
        "annotator_model": annotator_model,
        "judge_model": judge_model,
    })
    print("done")

    # ── 4. Sample ────────────────────────────────────────────────
    print("  [5/8] Sampling medical notes...", end=" ", flush=True)
    batch = sampler.sample_batch(sample_size)
    print("done")

    # ── 5. Annotate (parallel) ───────────────────────────────────
    # 8 workers stays safely under Gemini's per-minute RPM cap at flash/pro
    # rates while still parallelising batches of 200. Higher concurrency
    # triggers thundering-herd retries that waste time net-negative.
    annotator_workers = min(sample_size, 8)
    print(f"  [6/8] Generating annotations (parallel, {annotator_workers} workers)...")
    rows = list(batch.iterrows())

    def _annotate(args):
        i, (_, row) = args
        text = Annotator.redact_discharge_section(row["TEXT"])
        text = Annotator.clean_mimic_note(text)
        annotation = annotator.analyze_medical_text(text)
        return i, row, text, annotation

    _ann_results: Dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=annotator_workers) as pool:
        futures = {pool.submit(_annotate, item): item[0] for item in enumerate(rows, 1)}
        for f in as_completed(futures):
            i, row, text, annotation = f.result()
            _ann_results[i] = (row, text, annotation)
            print(f"    [{i}/{sample_size}] Subject {row['SUBJECT_ID']}: {annotation.get('diagnosis', 'N/A')}")

    # Log and build batch in original order
    annotation_batch = []
    for i in sorted(_ann_results):
        row, text, annotation = _ann_results[i]
        logger.log("SAMPLE_SELECTED", {
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
            "icd9_code": row["icd9_code"],
            "long_title": row["long_title"],
            "icd9_chapter": row.get("icd9_chapter"),
        })
        logger.log("ANNOTATION_PRODUCED", {
            "diagnosis": annotation["diagnosis"],
            "confidence": annotation.get("confidence_level", 0),
            "consistency_score": annotation.get("consistency_score"),
            "has_cot_reasoning": "reasoning" in annotation,
        })
        annotation_batch.append({
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
            "text": text,
            "annotation": annotation,
            "gold": row["gold"],
        })

    print(f"  All {len(annotation_batch)} annotations generated")

    # ── 6. Review (human or auto) ─────────────────────────────────
    print(f"  [7/8] Initializing reviewer ({reviewer_mode})...", end=" ", flush=True)
    if reviewer_mode == "auto":
        reviewer = AutoReviewer(judge_model=judge_model, verbose=debug)
    else:
        reviewer = HumanReviewer()
    print("done")

    print()
    print("=" * 60)
    print(f"REVIEW SESSION  [{reviewer_mode.upper()}]")
    print("=" * 60)
    if reviewer_mode == "human":
        print(f"Please review {len(annotation_batch)} annotations in the UI window")
    else:
        print(f"Auto-reviewing {len(annotation_batch)} annotations with {judge_model}...")
    print()

    parsed_signals = []
    n = len(annotation_batch)

    if reviewer_mode == "auto":
        # Rate limiter: enforces a minimum interval between API call starts so we
        # stay under the judge model's quota (at 2.5-flash: 60 req/min → 1 per 1s).
        # The lock is released before the actual API call so multiple calls can be
        # in-flight simultaneously — new ones just can't START more than once per interval.
        import time as _time
        _rate_lock = threading.Lock()
        _next_allowed: List[float] = [0.0]
        _MIN_INTERVAL = 0.0 if sample_size <= 50 else 2.5  # no throttle for small batches

        def _review(args):
            i, item = args
            with _rate_lock:
                now = _time.time()
                wait = _next_allowed[0] - now
                if wait > 0:
                    _time.sleep(wait)
                _next_allowed[0] = _time.time() + _MIN_INTERVAL
            fb = reviewer.review(
                annotation=item["annotation"],
                medical_note=item["text"],
                gold=item["gold"],
                case_num=i,
                total_cases=n,
            )
            return i, item, fb

        _rev_results: Dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=min(n, 20)) as pool:
            futures = {pool.submit(_review, (i, item)): i for i, item in enumerate(annotation_batch, 1)}
            for f in as_completed(futures):
                i, item, fb = f.result()
                _rev_results[i] = (item, fb)
                verdict = "✓" if fb.get("correct") else "✗"
                print(f"  [{i}/{n}] Subject {item['subject_id']}: {verdict}", flush=True)

        # Store and log in original order; collect RAG records for batch upload
        rag_batch = []
        for i in sorted(_rev_results):
            item, human_fb = _rev_results[i]
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
            if rag is not None and config.use_rag:
                rag_record = store.to_rag_record(store.records[-1])
                if rag_record:
                    rag_batch.append((rag_record["retrieval_text"], rag_record))
            parsed_signals.append(parser.parse(human_fb))

        if rag_batch:
            rag.add_cases_parallel(rag_batch)

    else:
        # Human review — must stay sequential (interactive UI)
        for idx, item in enumerate(annotation_batch, 1):
            print(f"[{idx}/{n}] Reviewing subject {item['subject_id']}...", flush=True)
            human_fb = reviewer.review(
                annotation=item["annotation"],
                medical_note=item["text"],
                gold=item["gold"],
                case_num=idx,
                total_cases=n,
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
            if rag is not None and config.use_rag:
                rag_record = store.to_rag_record(store.records[-1])
                if rag_record:
                    rag.add_case(rag_record["retrieval_text"], rag_record)
            parsed_signals.append(parser.parse(human_fb))

    reviewer.close()
    print("  All reviews completed")

    # ── 7. Evaluate ──────────────────────────────────────────────
    print("  [8/8] Analyzing results...")

    evaluator = PerformanceEvaluator()
    metrics = evaluator.evaluate(parsed_signals)
    metrics["run_number"] = run_number
    logger.log("METRICS_COMPUTED", metrics)

    # ── Print failure mode breakdown ──────────────────────────────
    breakdown = metrics.get("failure_mode_breakdown", {})
    active_failures = {k: v for k, v in breakdown.items() if v > 0}
    if active_failures:
        print("  Failure modes this run:")
        for mode, count in active_failures.items():
            print(f"    {mode}: {count}")

    calib = metrics.get("confidence_calibration", {})
    if calib.get("avg_confidence_correct") is not None:
        print(f"  Avg confidence — correct: {calib['avg_confidence_correct']}  "
              f"incorrect: {calib.get('avg_confidence_incorrect', 'N/A')}")

    # ── Prompt patching (only if technique is active) ────────────
    proposal = None
    if config.use_prompt_patching:
        # Accumulate signals for RAG routing reference, but only pass the
        # CURRENT run's signals to the prompt patcher. Passing all historical
        # signals causes old (already-fixed) failure modes to keep driving
        # new patches, adding noise rather than signal.
        all_signals = load_accumulated_signals()
        all_signals.extend(parsed_signals)
        save_accumulated_signals(all_signals)

        print("    Loading LLM for prompt refinement...", end=" ", flush=True)
        prompt_agent = PromptAgent()
        proposal = prompt_agent.propose_update(
            annotator.prompt_dict,
            parsed_signals,   # current run only — not all_signals
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
        notes=f"Experiment: {experiment_name}  Run: {run_number}",
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
    print(f"Run number       : {run_number}")
    print(f"Results saved to : {store_path}")
    print(f"Total samples    : {len(df)}")
    print(f"Accuracy         : {metrics.get('accuracy', 'N/A')}")
    print(f"Precision        : {metrics.get('precision', 'N/A')}")
    print(f"Hallucinations   : {metrics.get('hallucinations', 'N/A')}")
    for alert in metrics.get("alerts", []):
        print(f"  ⚠  {alert}")
    print()

    return df, metrics, proposal


# ================================================================
# MULTI-RUN (single experiment, improving over time)
# ================================================================

def run_multi_pipeline(
    num_runs: int = 3,
    sample_size: int = 10,
    debug: bool = False,
    initial_prompt_path: str = "logs/prompts/v1_initial.json",
    experiment_name: str = "full_pipeline",
    reviewer_mode: str = "human",
    judge_model: str = "gemini-2.5-flash",
    annotator_model: str = "gemini-2.5-flash",
):
    """
    Run one experiment multiple times in sequence.
    Each run benefits from accumulated RAG cases and an improving prompt.
    Shows the learning trajectory for a single technique combination.
    """
    config = EXPERIMENTS[experiment_name]
    rag_store_path = RAG_STORE_PATHS.get(experiment_name)

    print("=" * 60)
    print(f"MULTI-RUN  [{experiment_name}]  ({num_runs} runs × {sample_size} samples)")
    print("=" * 60)

    rag = load_or_create_rag(rag_store_path) if rag_store_path else None

    # Single shared sampler across all runs — stratified batches ensure
    # equal class representation and each run draws a fresh random sample.
    shared_sampler = DataSampler()

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
            run_number=run_num,
            reviewer_mode=reviewer_mode,
            judge_model=judge_model,
            sampler=shared_sampler,
            annotator_model=annotator_model,
        )

        all_metrics.append({"run": run_num, **metrics})

        if run_num < num_runs and proposal:
            new_path = save_proposed_prompt(proposal, run_num, base_prompt_path=prompt_path)
            if new_path:
                # ── Human approval of patch (fix #1) ──────────────────
                print()
                print("  Proposed prompt patch:")
                try:
                    import json as _json
                    with open(new_path) as _f:
                        _new = _json.load(_f)
                    for instr in _new.get("instructions", []):
                        if instr not in (df.get("instructions", []) if isinstance(df, dict) else []):
                            print(f"    + {instr}")
                except Exception:
                    print(f"    (see {new_path})")
                approve = input("  Apply this patch to the next run? [y/n, default y]: ").strip().lower()
                if approve in ("", "y"):
                    prompt_path = new_path
                    print(f"  Patch applied: {prompt_path}")
                else:
                    print(f"  Patch rejected — keeping: {prompt_path}")
            else:
                print(f"  Keeping current prompt: {prompt_path}")

    _print_multi_run_summary(experiment_name, all_metrics)
    return all_metrics


# ================================================================
# LONGITUDINAL STUDY
# Primary research protocol: runs EVERY condition for N runs and
# tracks metrics per run, producing the learning-curve comparison
# table that directly answers the main research question.
# ================================================================

def run_longitudinal_study(
    num_runs: int = 3,
    sample_size: int = 10,
    debug: bool = False,
    prompt_path: str = "logs/prompts/v1_initial.json",
    experiments_to_run: Optional[list] = None,
    reviewer_mode: str = "human",
    judge_model: str = "gemini-2.5-flash",
    annotator_model: str = "gemini-2.5-flash",
):
    """
    Run each experimental condition for `num_runs` sequential runs.

    Within each condition:
      - RAG store accumulates validated cases across runs.
      - Prompt patching (if active) applies improved prompts each run.

    Output: a learning-curve comparison table showing accuracy, hallucinations,
    and failure mode counts per run per condition, saved to logs/experiments/.

    This function directly answers:
      Main Q  — Does HITL feedback improve accuracy over runs?
      SubQ1   — Which failure modes decrease most with feedback?
      SubQ2   — Which techniques show the most improvement per run?
    """
    to_run = experiments_to_run or list(EXPERIMENTS.keys())

    print("=" * 60)
    print(f"LONGITUDINAL STUDY  ({len(to_run)} conditions × {num_runs} runs × {sample_size} samples)")
    print("=" * 60)

    # Single shared sampler — stratified batches, no case repetition across conditions/runs
    shared_sampler = DataSampler()
    # longitudinal_results[experiment_name] = list of metrics dicts (one per run)
    longitudinal_results: Dict[str, List[Dict]] = {}

    for name in to_run:
        config = EXPERIMENTS[name]
        rag_store_path = RAG_STORE_PATHS.get(name)
        rag = load_or_create_rag(rag_store_path) if rag_store_path else None

        print(f"\n{'─' * 60}")
        print(f"CONDITION: {name}")
        active = [k for k, v in config.__dict__.items() if v is True]
        print(f"Techniques: {', '.join(active) if active else 'none'}")
        print(f"{'─' * 60}")

        run_metrics = []
        current_prompt = prompt_path

        for run_num in range(1, num_runs + 1):
            _, metrics, proposal = run_pipeline(
                sample_size=sample_size,
                debug=debug,
                prompt_path=current_prompt,
                technique_config=config,
                rag=rag,
                experiment_name=name,
                run_number=run_num,
                reviewer_mode=reviewer_mode,
                judge_model=judge_model,
                sampler=shared_sampler,
                annotator_model=annotator_model,
            )
            run_metrics.append({"run": run_num, **metrics})

            if run_num < num_runs and proposal:
                new_path = save_proposed_prompt(proposal, run_num, base_prompt_path=current_prompt)
                if new_path:
                    current_prompt = new_path

        longitudinal_results[name] = run_metrics
        _print_multi_run_summary(name, run_metrics)

    # ── Print full cross-condition learning curve table ───────────
    _print_longitudinal_table(longitudinal_results, num_runs)

    # ── Save full results ─────────────────────────────────────────
    os.makedirs("logs/experiments", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"logs/experiments/longitudinal_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(longitudinal_results, f, indent=2)
    print(f"\nFull results saved to: {summary_path}")

    return longitudinal_results


def _print_multi_run_summary(experiment_name: str, all_metrics: List[Dict]):
    """Print a compact per-run summary table for one condition."""
    print(f"\n{'=' * 75}")
    print(f"MULTI-RUN SUMMARY  [{experiment_name}]")
    print("=" * 75)
    print(
        f"  {'Run':<5} {'Acc':>6} {'Hold':>6} {'Halluc':>7} {'Conf✓':>7} {'Conf✗':>7} "
        f"{'RAG→':>6} {'Patch→':>7} {'Flag':>5}  Top failure modes"
    )
    print(f"  {'-' * 78}")
    for m in all_metrics:
        calib = m.get("confidence_calibration", {})
        conf_ok  = calib.get("avg_confidence_correct")
        conf_bad = calib.get("avg_confidence_incorrect")
        holdout  = m.get("holdout_accuracy")

        breakdown = m.get("failure_mode_breakdown", {})
        top_failures = ", ".join(
            f"{k}:{v}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1]) if v > 0
        ) or "none"

        print(
            f"  {m['run']:<5} "
            f"{m.get('accuracy', 0):>6.2f} "
            f"{str(round(holdout, 2)) if holdout is not None else 'N/A':>6} "
            f"{m.get('hallucinations', 0):>7} "
            f"{str(round(conf_ok, 0)) if conf_ok is not None else 'N/A':>7} "
            f"{str(round(conf_bad, 0)) if conf_bad is not None else 'N/A':>7} "
            f"{m.get('rag_routed_failures', 0):>6} "
            f"{m.get('prompt_routed_failures', 0):>7} "
            f"{m.get('flagged_cases', 0):>5}  "
            f"{top_failures}"
        )
    print()
    # Column legend
    print("  Acc     = training-set accuracy (in-sample)")
    print("  Hold    = held-out accuracy (out-of-sample generalisation)")
    print("  Conf✓/✗ = avg model confidence on correct/incorrect cases")
    print("  RAG→    = failures routed to RAG (need more examples)")
    print("  Patch→  = failures routed to prompt patch")
    print("  Flag    = ambiguous cases flagged for data-quality review")
    print()


def _print_longitudinal_table(results: Dict[str, List[Dict]], num_runs: int):
    """
    Print the main research comparison table:
    Condition | Run1→RunN accuracy | Trend | Halluc R1→Rn | Conf calibration R1→Rn
    """
    print("\n" + "=" * 90)
    print("LONGITUDINAL RESULTS — LEARNING CURVE")
    print("=" * 90)

    run_headers = "  ".join(f"R{r} acc" for r in range(1, num_runs + 1))
    print(f"{'Condition':<22}  {run_headers}  {'Trend':>6}  {'Halluc':>10}  {'Conf✓ R1→Rn':>14}  {'Flagged':>8}")
    print("-" * 90)

    for name, run_metrics in results.items():
        accs   = [m.get("accuracy", 0.0) for m in run_metrics]
        halluc = [m.get("hallucinations", 0) for m in run_metrics]
        conf_ok = [
            m.get("confidence_calibration", {}).get("avg_confidence_correct")
            for m in run_metrics
        ]
        flagged = sum(m.get("flagged_cases", 0) for m in run_metrics)

        acc_cells = "  ".join(f"{a:>6.2f}" for a in accs)

        if len(accs) >= 2:
            delta = accs[-1] - accs[0]
            trend = (f"↑+{delta:.2f}" if delta > 0.01
                     else f"↓{delta:.2f}" if delta < -0.01
                     else "  ~0.00")
        else:
            trend = "N/A"

        halluc_str = f"{halluc[0]}→{halluc[-1]}" if len(halluc) >= 2 else "N/A"

        c0 = f"{conf_ok[0]:.0f}" if conf_ok[0] is not None else "?"
        cn = f"{conf_ok[-1]:.0f}" if conf_ok[-1] is not None else "?"
        conf_str = f"{c0}→{cn}" if len(conf_ok) >= 2 else "N/A"

        print(f"{name:<22}  {acc_cells}  {trend:>6}  {halluc_str:>10}  {conf_str:>14}  {flagged:>8}")

    print()
    print("  Trend  : accuracy change from run 1 → last run")
    print("  Conf✓  : avg model confidence on correct cases (higher = better calibrated)")
    print("  Flagged: total ambiguous cases across all runs")
    print()


# ================================================================
# ABLATION STUDY (single run per condition, compare snapshot)
# ================================================================

def run_ablation_study(
    sample_size: int = 10,
    debug: bool = False,
    prompt_path: str = "logs/prompts/v1_initial.json",
    experiments_to_run: Optional[list] = None,
    reviewer_mode: str = "human",
    judge_model: str = "gemini-2.5-flash",
    annotator_model: str = "gemini-2.5-flash",
):
    """
    Run each experimental condition once and compare results.
    Use this for a quick snapshot; use run_longitudinal_study for
    the full research protocol that shows improvement over time.
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
        print("=" * 60)

        _, metrics, _ = run_pipeline(
            sample_size=sample_size,
            debug=debug,
            prompt_path=prompt_path,
            technique_config=config,
            rag=rag,
            experiment_name=name,
            reviewer_mode=reviewer_mode,
            judge_model=judge_model,
            annotator_model=annotator_model,
        )

        metrics["experiment"] = name
        results.append(metrics)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<26} {'Accuracy':>10} {'Precision':>10} {'Hallucinations':>16}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['experiment']:<26} "
            f"{str(r.get('accuracy', 'N/A')):>10} "
            f"{str(r.get('precision', 'N/A')):>10} "
            f"{str(r.get('hallucinations', 'N/A')):>16}"
        )
    print()

    os.makedirs("logs/experiments", exist_ok=True)
    summary_path = f"logs/experiments/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


# ================================================================
# INTERACTIVE CLI
# Run `python main.py` to launch this menu.
# ================================================================

def _list_experiments():
    print()
    print("  Available experiments:")
    for i, (name, cfg) in enumerate(EXPERIMENTS.items(), 1):
        active = [k for k, v in cfg.__dict__.items() if v is True]
        flags = ", ".join(active) if active else "none (baseline)"
        print(f"    {i:>2}. {name:<26}  [{flags}]")
    print()


def _ask_experiment(allow_all: bool = False) -> Optional[str]:
    _list_experiments()
    keys = list(EXPERIMENTS.keys())
    suffix = " (or 'all')" if allow_all else ""
    choice = input(f"  Enter experiment name or number{suffix}: ").strip()
    if allow_all and choice.lower() == "all":
        return "all"
    if choice.isdigit() and 1 <= int(choice) <= len(keys):
        return keys[int(choice) - 1]
    if choice in EXPERIMENTS:
        return choice
    print(f"  Unknown choice: {choice}")
    return None


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"  {prompt} [default {default}]: ").strip()
    return int(raw) if raw.isdigit() else default


def _ask_reviewer_mode() -> tuple:
    """Ask the user whether to use human or auto review. Returns (mode, judge_model)."""
    print()
    print("  Reviewer mode:")
    print("    [h]  Human  — interactive UI (you review each annotation)")
    print("    [a]  Auto   — LLM-as-judge (stronger Gemini reviews automatically)")
    print()
    choice = input("  Select [h/a, default h]: ").strip().lower()
    if choice == "a":
        print()
        print("  Judge model options:")
        print("    [1]  gemini-2.5-flash          (Gemini 2.5 Flash — fastest, default)")
        print("    [2]  gemini-2.5-pro            (Gemini 2.5 Pro — stronger)")
        print("    [c]  custom                    — type any model ID")
        print()
        m = input("  Select [1-2/c, default 1]: ").strip().lower()
        judge_map = {
            "1": "gemini-2.5-flash",
            "2": "gemini-2.5-pro",
        }
        if m in judge_map:
            judge = judge_map[m]
        elif m == "c":
            raw = input("  Model ID (leave blank to cancel): ").strip()
            judge = raw if raw else "gemini-2.5-flash"
            if not raw:
                print("  No model entered — defaulting to gemini-2.5-flash")
        else:
            judge = "gemini-2.5-flash"
        print(f"  Auto-review enabled with judge: {judge}")
        return "auto", judge
    return "human", "gemini-2.5-flash"


def _show_prompt_diff():
    """Print a before/after diff of prompt instructions for any patch file."""
    import glob as _glob

    patch_files = sorted(_glob.glob("logs/prompts/*_instruction_patch.json"))
    if not patch_files:
        print("  No patch files found in logs/prompts/")
        return

    print()
    print("  Available prompt patches:")
    for i, path in enumerate(patch_files, 1):
        ts = os.path.basename(path).replace("_instruction_patch.json", "")
        print(f"    [{i}]  {ts}")
    print()

    raw = input(f"  Select patch [1-{len(patch_files)}]: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(patch_files)):
        print("  Invalid selection.")
        return

    patch_path = patch_files[int(raw) - 1]
    with open(patch_path) as f:
        patch = json.load(f)

    # Load the base prompt to show original instructions
    ip = patch.get("instruction_patch", {})
    to_add = ip.get("instructions_to_add", [])
    to_refine = ip.get("instructions_to_refine", [])
    failures = patch.get("aggregated_failures", {})

    print()
    print("=" * 60)
    print(f"  PROMPT DIFF — {os.path.basename(patch_path)}")
    print("=" * 60)

    print()
    print("  Failure modes that triggered this patch:")
    for mode, info in sorted(failures.items(), key=lambda x: -x[1].get("count", 0)):
        print(f"    {mode:35s} {info.get('count', '?')}x")

    if to_add:
        print()
        print("  + ADDED instructions:")
        for instr in to_add:
            print(f"    + {instr}")

    if to_refine:
        print()
        print("  ~ REFINED instructions:")
        for item in to_refine:
            print(f"    BEFORE: {item.get('target_instruction', '?')}")
            print(f"    AFTER : {item.get('refined_version', '?')}")
            print()

    if not to_add and not to_refine:
        print("  (no instruction changes recorded)")

    print()
    print("  Rationale:")
    rationale = ip.get("rationale", "(none)")
    # Wrap at 70 chars
    import textwrap
    for line in textwrap.wrap(rationale, width=70):
        print(f"    {line}")
    print()


def interactive_cli():
    prompt_path = "logs/prompts/v1_initial.json"

    while True:
        print()
        print("=" * 60)
        print("  MEDICAL ANNOTATION EXPERIMENT RUNNER")
        print("=" * 60)
        print()
        print("  [1]  Single run         — annotate + review once")
        print("  [2]  Multi-run          — one condition, N improving runs")
        print("  [3]  Ablation study     — all conditions, single run, compare")
        print("  [4]  Longitudinal study — all conditions, N runs (full protocol)")
        print("  [5]  Prompt diff        — show what changed between prompt versions")
        print()
        print("  [q]  Quit")
        print()

        choice = input("  Select: ").strip().lower()

        if choice == "q":
            print("  Bye!")
            break

        elif choice == "1":
            exp = _ask_experiment()
            if not exp:
                continue
            n = _ask_int("Sample size", 5)
            reviewer_mode, judge_model = _ask_reviewer_mode()
            config = EXPERIMENTS[exp]
            rag_path = RAG_STORE_PATHS.get(exp)
            rag = load_or_create_rag(rag_path) if rag_path else None
            run_pipeline(
                sample_size=n,
                prompt_path=prompt_path,
                technique_config=config,
                rag=rag,
                experiment_name=exp,
                reviewer_mode=reviewer_mode,
                judge_model=judge_model,
            )

        elif choice == "2":
            exp = _ask_experiment()
            if not exp:
                continue
            n = _ask_int("Sample size per run", 10)
            r = _ask_int("Number of runs", 3)
            reviewer_mode, judge_model = _ask_reviewer_mode()
            run_multi_pipeline(
                num_runs=r,
                sample_size=n,
                initial_prompt_path=prompt_path,
                experiment_name=exp,
                reviewer_mode=reviewer_mode,
                judge_model=judge_model,
            )

        elif choice == "3":
            exp = _ask_experiment(allow_all=True)
            if not exp:
                continue
            n = _ask_int("Sample size per condition", 10)
            reviewer_mode, judge_model = _ask_reviewer_mode()
            exps = list(EXPERIMENTS.keys()) if exp == "all" else [exp]
            run_ablation_study(
                sample_size=n,
                prompt_path=prompt_path,
                experiments_to_run=exps,
                reviewer_mode=reviewer_mode,
                judge_model=judge_model,
            )

        elif choice == "4":
            exp = _ask_experiment(allow_all=True)
            if not exp:
                continue
            n = _ask_int("Sample size per run", 10)
            r = _ask_int("Number of runs per condition", 3)
            reviewer_mode, judge_model = _ask_reviewer_mode()
            exps = list(EXPERIMENTS.keys()) if exp == "all" else [exp]
            run_longitudinal_study(
                num_runs=r,
                sample_size=n,
                prompt_path=prompt_path,
                experiments_to_run=exps,
                reviewer_mode=reviewer_mode,
                judge_model=judge_model,
            )

        elif choice == "5":
            _show_prompt_diff()

        else:
            print("  Unknown option. Try 1–5 or q.")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    interactive_cli()
