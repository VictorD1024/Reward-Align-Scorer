# Contributing

Contributions are welcome. The project is intentionally small and focused:

- keep the reward entrypoint framework-friendly;
- keep the core scorer independent from business-specific domains;
- add tests for alignment behavior whenever changing DP or windowing logic;
- document latency and quality tradeoffs when adding new scoring modes.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Pull Requests

Please include:

- a short problem statement;
- implementation summary;
- test or benchmark evidence;
- any threshold or scoring behavior changes.

