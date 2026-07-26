# Contributing

Contributions are welcome. The project is intentionally small and focused:

- keep the reward entrypoint framework-friendly;
- keep the core scorer independent from business-specific domains;
- add tests for alignment behavior whenever changing DP or windowing logic;
- keep workflow cycles explicitly bounded and add graph/path tests when
  changing workflow semantics;
- preserve the TraceEvent trust boundary: model-controlled payloads must never
  become trusted runtime evidence implicitly;
- keep runtime adapters deterministic and add success, failure, trust, and
  sensitive-field redaction tests for every new built-in tool family;
- document latency and quality tradeoffs when adding new scoring modes.

## Module boundaries

Keep implementation ownership aligned with these layers:

```text
integrations
└── runtime + workflows
    └── core
        └── backends
```

- `backends`: model loading, encoding, pooling, and cache primitives;
- `core`: windows, alignment, semantic scoring, confidence, and trajectories;
- `workflows`: bounded graph schemas and hierarchical reward;
- `runtime`: trusted events, trace validation, and tool-call normalization;
- `integrations`: framework-specific composition such as veRL.

`runtime` validates events against workflow schemas, while
`WorkflowScorer.score_events()` is the narrow convenience bridge back to the
runtime validator. Avoid introducing additional cross-layer cycles.

The flat modules under `raise_scorer/*.py` are compatibility shims. Add new
implementation code to the grouped package that owns the behavior, then
re-export only stable public symbols through its `__init__.py`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

## Pull Requests

Please include:

- a short problem statement;
- implementation summary;
- test or benchmark evidence;
- any threshold or scoring behavior changes.
