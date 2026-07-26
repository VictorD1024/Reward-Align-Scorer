"""Recommended RLVR Parquet row layouts for RAISE.

The Parquet reader is owned by veRL. Its reward manager extracts
``reward_model.ground_truth`` and ``extra_info`` from each row before calling
the custom reward function.
"""

from raise_scorer.integrations import compute_score, resolve_reference_steps


def main():
    ground_truth_row = {
        "data_source": "repo-repair",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "reference_steps": [
                    "inspect repository",
                    ["patch code", "apply workaround"],
                    "run tests",
                ]
            },
        },
        "extra_info": {"task_id": "repair-001"},
    }
    resolution = resolve_reference_steps(
        ground_truth_row["extra_info"],
        ground_truth_row["reward_model"]["ground_truth"],
    )
    print("ground_truth layout:", resolution)

    extra_info_row = {
        "data_source": "repo-repair",
        "reward_model": {
            "style": "rule",
            "ground_truth": "optional final-answer verifier target",
        },
        "extra_info": {
            "task_id": "repair-002",
            "reference_steps": [
                "inspect repository",
                "patch code",
                "run tests",
            ],
        },
    }
    details = compute_score(
        data_source=extra_info_row["data_source"],
        solution_str="inspect repository, patch code, run tests",
        ground_truth=extra_info_row["reward_model"]["ground_truth"],
        extra_info=extra_info_row["extra_info"],
        return_details=True,
    )
    print("extra_info layout source:", details["reference_steps_source"])
    print("score:", details["score"])


if __name__ == "__main__":
    main()
