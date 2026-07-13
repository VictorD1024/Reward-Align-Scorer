from raise_scorer.verl_adapter import compute_score


def main():
    steps = [
        "summarize the reported bug symptoms",
        "inspect relevant files",
        "patch the implementation",
        "run tests",
    ]
    details = compute_score(
        solution_str=(
            "<think>plan fix</think> "
            "The bug shows up as a signed overflow near INT64_MIN. "
            "I inspected src/module.py and patched the boundary check. "
            "pytest tests/test_module.py passed."
        ),
        extra_info={"reference_steps": steps},
        return_details=True,
    )

    if details["fallback_recommended"]:
        print("route → LLM Judge")
        print("reasons:", details["fallback_reasons"])
    else:
        print("route → semantic reward")
        print("score:", details["score"], "confidence:", round(details["confidence"], 3))

    print("step tiers:", details["step_tiers"])


if __name__ == "__main__":
    main()
