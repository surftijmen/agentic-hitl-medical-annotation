from dataclasses import dataclass

@dataclass
class TechniqueConfig:
    use_rag: bool = False
    use_self_consistency: bool = False
    n_consistency_samples: int = 5
    use_prompt_patching: bool = False
    use_human_feedback_routing: bool = False  