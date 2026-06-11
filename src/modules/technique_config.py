"""Legacy standalone copy of the technique-flag dataclass — kept as inline
documentation of what each flag means. The authoritative TechniqueConfig
used at runtime lives in annotator_agent.py (main.py imports it from
there). The two diverge on a few field defaults; treat annotator_agent.py
as the source of truth.
"""
from dataclasses import dataclass

@dataclass
class TechniqueConfig:
    use_rag: bool = False  
    # RAG (Retrieval-Augmented Generation): 
    # Enhances model responses by retrieving relevant external documents 
    # (e.g., from a vector database or search engine) and injecting them 
    # into the prompt to improve factual accuracy and reduce hallucinations.

    use_self_consistency: bool = False  
    # Self-Consistency:
    # Generates multiple reasoning paths for the same question and selects 
    # the most consistent or frequent final answer. Improves reliability 
    # especially for reasoning-heavy tasks (e.g., math, logic).

    n_consistency_samples: int = 5  
    # Number of independent reasoning samples to generate when 
    # self-consistency is enabled. Higher values increase robustness 
    # but also increase latency and cost.

    use_prompt_patching: bool = False  
    # Prompt Patching:
    # Dynamically modifies or augments prompts at runtime based on 
    # task type, detected weaknesses, safety rules, or intermediate outputs.
    # Often used to insert clarifications, constraints, or formatting guidance.

    use_human_feedback_routing: bool = False  
    # Human Feedback Routing:
    # Routes uncertain, high-risk, or low-confidence outputs to a human 
    # reviewer. Can be triggered by confidence thresholds, policy checks, 
    # or ambiguity detection.

    chain_of_thoughts: bool = False  
    # Chain-of-Thought (CoT):
    # Encourages the model to generate intermediate reasoning steps 
    # before producing a final answer. Improves performance on complex 
    # reasoning tasks but may increase token usage.