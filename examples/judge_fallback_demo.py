from reward_align_scorer.verl_adapter import compute_score


def should_fallback_to_judge(details: dict) -> bool:
    if details["step_match_rate"] < 0.5:
        return True
    if len(details["unmatched_steps"]) >= 2:
        return True
    return False


details = compute_score(
    solution_str="<think>plan</think> read the task, inspect files, patch code, and summarize.",
    extra_info={"reference_steps": ["read the task", "inspect files", "patch code", "run tests", "summarize"]},
    return_details=True,
)

if should_fallback_to_judge(details):
    print("fallback to LLM Judge")
else:
    print("use semantic reward", details["score"])
