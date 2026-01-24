# Agentic Human-in-the-Loop Medical Annotation Framework

## Overview

This project implements an **agentic, human-in-the-loop (HITL) framework** for medical text annotation.  
The goal is **not** to maximize model accuracy in isolation, but to study **how structured human feedback can iteratively improve annotation quality through prompt refinement**, while keeping evaluation and control external to the system.

The framework is inspired by adaptive HITL systems such as ARIA, but emphasizes:
- modularity
- transparency
- controlled self-improvement
- explicit separation between system behavior and evaluation

---

## Core Idea

Large Language Models can annotate medical text, but they often fail in subtle, systematic ways (e.g. confusing symptoms with diagnoses).  
Rather than retraining the model, this system **improves the instructions (prompt)** used by the model, guided by **human qualitative feedback**.

Crucially:
- the system **does not evaluate itself**
- the system **does not select its own data autonomously**
- all adaptation is **human-gated and externally observable**

---

## System Architecture

### Main Components

| Module | Responsibility |
|------|----------------|
| **DataSampler** | Loads clinical notes and samples data according to an explicit sampling policy |
| **Annotator Agent** | Applies an LLM to extract structured medical information using a fixed prompt |
| **Human Reviewer** | Interactively validates annotations and provides qualitative feedback |
| **Feedback Parser** | Converts free-text human feedback into structured signals |
| **Performance Evaluator** | Aggregates feedback into metrics (outside the core system logic) |
| **Prompt Agent** | Proposes prompt refinements based on aggregated feedback |
| **Annotation Store** | Logs all annotations, references, and feedback for offline analysis |

---

## Human-in-the-Loop Design

The **human reviewer is authoritative**.

- The reviewer sees:
  - the full medical note
  - the model’s predicted diagnosis
  - model confidence
  - an optional reference label
- The reviewer explicitly marks:
  - **Correct / Incorrect**
  - Optional qualitative feedback explaining *why*

A persistent UI is used so that multiple samples can be reviewed in sequence without restarting the interface.

The system **waits for human input** before continuing.

---

## Self-Improvement Loop (Controlled)

The system improves **between runs**, not online.

### One iteration:
1. Sample a batch of notes
2. Annotate using the current prompt
3. Collect human feedback
4. Log all actions and outcomes
5. Evaluate performance externally
6. Propose a revised prompt
7. Human approves the update
8. Next run uses the updated prompt

This ensures:
- reproducibility
- interpretability
- no hidden feedback loops

---

## About “Gold” Labels

This project uses MIMIC-III data.

**Important:**  
`ADMISSIONS.DIAGNOSIS` in MIMIC is **not a confirmed ground-truth diagnosis**.

It is:
- free-text
- often a chief complaint or administrative summary
- sometimes vague or incomplete

Therefore:
- gold labels are treated as **weak reference signals**
- they are **never used to automatically correct the system**
- disagreements between model and gold are expected

This limitation motivates the need for **human-in-the-loop validation**.

---

## Why Sampling Is Decoupled

The DataSampler:
- does **not** adapt automatically based on model performance
- follows an explicit, documented sampling policy

Sampling decisions are:
- informed by external analysis
- optionally adjusted by a human
- applied in the next run

This avoids:
- self-reinforcing bias
- distribution drift
- hidden optimization artifacts

---

## Logging and Evaluation Philosophy

The system logs:
- inputs
- annotations
- prompts
- human judgments
- feedback signals

**Evaluation happens outside the system boundary.**

This allows:
- slice-based analysis (e.g. by note type, disease category)
- bias inspection
- reproducible experiments

The system itself does not “know” if it is doing well.

---

## Running the Pipeline

```bash
python src/main.py
```

The main pipeline:

- samples notes
- runs annotation
- opens the human review UI
- collects feedback
- proposes prompt updates