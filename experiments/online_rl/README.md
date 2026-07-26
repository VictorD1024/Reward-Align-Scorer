# Three-domain online RL protocol

本目录固定论文主实验矩阵，不新增环境专用 adapter。Code、Browser、Tool 三类
rollout 系统只需输出已有的可信 `TraceEvent`；默认比较 outcome-only、线性终局
reward、图终局 reward 和图约束 turn-potential reward，共
`3 domains × 4 ablations × 3 seeds = 36 runs`。每条 episode 必须分别记录
outcome/process/total reward、成功率、非法转移、turn 数、耗时与图截断情况。

This directory fixes the paper-facing experiment matrix before expensive
training starts. It deliberately defines no new environment adapter: each
rollout system only needs to emit the existing trusted `TraceEvent` schema.

## Matrix

- Code: SWE-Gym training, SWE-bench Verified evaluation.
- Browser: BrowserGym/WebArena training, WebArena Verified evaluation.
- Tool use: executable BFCL multi-turn training tasks, BFCL V4 agentic
  evaluation.
- Rewards: outcome only, enumerated linear terminal, graph terminal, and graph
  turn-potential.
- Seeds: 1, 2, 3. The default matrix therefore contains 36 training runs.

Load and validate the checked-in protocol:

```python
import json

from raise_scorer.experiments import OnlineRLSuite

with open("experiments/online_rl/three_domain.json") as handle:
    suite = OnlineRLSuite.from_dict(json.load(handle))

assert suite.runs == 36
```

For the dense condition, compute prefix rewards with
`WorkflowPotentialScorer`, then use `assign_turn_rewards` to place each
`gamma * Phi_t - Phi_(t-1)` value on the final generated token of its agent
turn. Outcome rewards remain environment/verifier outputs and are mixed only
after both components have been logged separately.

## Required reporting

Report task success, outcome/process/total reward, invalid-transition rate,
turn count, wall time, and graph truncation rate. Keep domain, ablation and
seed in every episode row. Tune reward weights on a development split; do not
select them on the named evaluation split.

The environment choices are grounded in the public
[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym),
[BrowserGym](https://github.com/ServiceNow/BrowserGym), and
[BFCL](https://gorilla.cs.berkeley.edu/leaderboard) projects. Pin exact
revisions and record container/browser/API snapshots in actual runs.
