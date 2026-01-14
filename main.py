from agents.annotator_agent import Annotator
from agents.feedback_agent import FeedbackAgent
from agents.prompt_agent import PromptAgent  # optional: updates prompts via LLM
from agents.reviewer import HumanReviewer    # optional: human-in-loop check

def run_pipeline(sample_size=5, debug=True):
    # 1 Initialize annotator
    annotator = Annotator(debug=debug)

    # 2 Initialize feedback agent
    feedback = FeedbackAgent(client=annotator.client, model_name=annotator.model_name)

    # 3 Optionally, initialize prompt agent
    prompt_agent = PromptAgent(client=annotator.client, model_name=annotator.model_name)

    # 4 Sample notes & run analysis
    batch_results, metrics, health = annotator.random_mimic_test_with_feedback(n=sample_size)

    print("\n=== Batch Metrics ===")
    print(metrics.json())

    print("\n=== System Health ===")
    print(health.json())

    # 5 Optionally suggest prompt updates if system not healthy
    if not health.healthy:
        suggestion = prompt_agent.suggest_prompt_update(
            batch_results=batch_results,
            metrics=metrics,
            prompt_dict=annotator.prompt_dict
        )
        if suggestion:
            print("\n=== Prompt Update Suggestion ===")
            print(suggestion.json())

    # 6: send results to human reviewer
    reviewer = HumanReviewer()
    reviewer.review(batch_results)

if __name__ == "__main__":
    run_pipeline(sample_size=10, debug=False)
