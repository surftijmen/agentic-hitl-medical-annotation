import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "rag_plus_consistency":"logs/rag_stores/rag_plus_consistency.txt",
    "cot_plus_few_shot":   None,
    "rag_plus_cot":        "logs/rag_stores/rag_plus_cot.txt",
    "full_pipeline":       "logs/rag_stores/full_pipeline.txt",
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
    judge_model: str = "gemini-3-flash-preview",
    max_workers: int = 5,
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
    sampler = DataSampler(
        notes_path="data/mimiciii_notes.parquet",
        adm_path="data/mimiciii_patients_admissions.parquet",
    )
    print("done")

    # ── 3. Annotator ─────────────────────────────────────────────
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
        "run_number": run_number,
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

    # ── 5. Annotate (parallel) ───────────────────────────────────
    print(f"  [6/8] Generating annotations (parallel, {sample_size} workers)...")
    rows = list(batch.iterrows())

    def _annotate(args):
        i, (_, row) = args
        text = Annotator.clean_mimic_note(row["TEXT"])
        annotation = annotator.analyze_medical_text(text)
        return i, row, text, annotation

    _ann_results: Dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=sample_size) as pool:
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
        })
        logger.log("ANNOTATION_PRODUCED", {
            "diagnosis": annotation["diagnosis"],
            "confidence": annotation["confidence_level"],
            "consistency_score": annotation.get("consistency_score"),
            "has_cot_reasoning": "reasoning" in annotation,
        })
        annotation_batch.append({
            "subject_id": row["SUBJECT_ID"],
            "hadm_id": row["HADM_ID"],
            "text": text,
            "annotation": annotation,
            "gold": row["label"],
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
        # Parallel auto-review — all judge API calls fire at once
        def _review(args):
            i, item = args
            fb = reviewer.review(
                annotation=item["annotation"],
                medical_note=item["text"],
                gold=item["gold"],
                case_num=i,
                total_cases=n,
            )
            return i, item, fb

        _rev_results: Dict[int, tuple] = {}
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_review, (i, item)): i for i, item in enumerate(annotation_batch, 1)}
            for f in as_completed(futures):
                i, item, fb = f.result()
                _rev_results[i] = (item, fb)
                verdict = "✓" if fb.get("correct") else "✗"
                print(f"  [{i}/{n}] Subject {item['subject_id']}: {verdict}", flush=True)

        # Store and log in original order
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
                    rag.add_case(rag_record["retrieval_text"], rag_record)
            parsed_signals.append(parser.parse(human_fb))

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
    judge_model: str = "gemini-3-flash-preview",
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
        )

        all_metrics.append({"run": run_num, **metrics})

        if run_num < num_runs and proposal:
            new_path = save_proposed_prompt(proposal, run_num, base_prompt_path=prompt_path)
            if new_path:
                prompt_path = new_path
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
    judge_model: str = "gemini-3-flash-preview",
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
        f"  {'Run':<5} {'Acc':>6} {'Halluc':>7} {'Conf✓':>7} {'Conf✗':>7} "
        f"{'RAG→':>6} {'Patch→':>7} {'Flag':>5}  Top failure modes"
    )
    print(f"  {'-' * 70}")
    for m in all_metrics:
        calib = m.get("confidence_calibration", {})
        conf_ok  = calib.get("avg_confidence_correct")
        conf_bad = calib.get("avg_confidence_incorrect")

        breakdown = m.get("failure_mode_breakdown", {})
        top_failures = ", ".join(
            f"{k}:{v}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1]) if v > 0
        ) or "none"

        print(
            f"  {m['run']:<5} "
            f"{m.get('accuracy', 0):>6.2f} "
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
    judge_model: str = "gemini-3-flash-preview",
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
        print("    [1]  gemini-3.1-pro-preview   (latest Gemini 3 — strongest judge)")
        print("    [2]  gemini-3-pro-preview      (Gemini 3 Pro)")
        print("    [3]  gemini-3-flash-preview    (Gemini 3 Flash — faster)")
        print("    [4]  gemini-2.5-pro            (Gemini 2.5 Pro — stable)")
        print("    [5]  gemini-2.5-flash          (Gemini 2.5 Flash — fastest)")
        print("    [c]  custom                    — type any model ID")
        print()
        m = input("  Select [1-5/c, default 3]: ").strip().lower()
        judge_map = {
            "1": "gemini-3.1-pro-preview",
            "2": "gemini-3-pro-preview",
            "3": "gemini-3-flash-preview",
            "4": "gemini-2.5-pro",
            "5": "gemini-2.5-flash",
        }
        if m in judge_map:
            judge = judge_map[m]
        elif m == "c":
            raw = input("  Model ID (leave blank to cancel): ").strip()
            judge = raw if raw else "gemini-3-flash-preview"
            if not raw:
                print("  No model entered — defaulting to gemini-3-flash-preview")
        else:
            judge = "gemini-3-flash-preview"
        print(f"  Auto-review enabled with judge: {judge}")
        return "auto", judge
    return "human", "gemini-3-flash-preview"


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

        else:
            print("  Unknown option. Try 1–4 or q.")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    interactive_cli()
