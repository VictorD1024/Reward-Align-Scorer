# veRL Integration

The package exposes a veRL-compatible reward entrypoint:

```python
from reward_align_scorer.verl_adapter import compute_score
```

For a veRL source checkout, you can create:

```text
verl/utils/reward_score/semantic_align.py
```

with:

```python
from reward_align_scorer.verl_adapter import compute_score

__all__ = ["compute_score"]
```

Then set:

```bash
export REWARD_ALIGN_MODEL_PATH=/path/to/bge-small-zh-v1.5
export REWARD_ALIGN_THRESHOLD=0.65
```

Expected `extra_info`:

```python
{
    "reference_steps": [
        "read the issue",
        "inspect relevant files",
        "modify the implementation",
        "run tests",
        "summarize the fix",
    ]
}
```

The scorer also accepts the legacy key `operation`.
