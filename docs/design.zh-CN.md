# 设计

<p align="right">
  <a href="./design.md">English</a> |
  <a href="./design.zh-CN.md">中文</a>
</p>

RAISE（Reward-Aligned Agent Trajectory Scorer）的核心工程选择：

> 在调用慢速 Judge 之前，把结构化 reward 判断转换为批量语义对齐。

长响应 RL reward 的瓶颈不是算力——而是每条样本一次自回归 LLM-Judge 调用，在整个 rollout batch 上串行。下面每个机制都是为了让那条调用可以被批量张量工作替代。

## 流水线

1. 从 `reference_steps`（主键）或 `actions`（Agentic-RL 别名）解析参考结构——checklist 项、轨迹节点、子任务。
2. 将长响应切分成重叠的、边界感知的滑动窗口。
3. 将 reference items 和 response windows 编码为归一化 embedding（批量编码，LRU 缓存）。
4. 在 GPU / Ascend NPU 上以单次 GEMM 计算稠密相似度矩阵。
5. 在 CPU（numpy）上运行带阈值的单调对齐 DP。
6. 返回 `score = match_rate × order_rate × max_score` 以及可解释的对齐细节。

## 为什么用滑动窗口

对长模型响应来说，句子边界很脆弱。一个必需的语义单元可能跨越标点边界，而一条 4096 token 的响应里可能围绕简短关键动作包含大片解释。滑动窗口：

- 在控制比较单元数量的同时保留局部上下文（`max_windows`）；
- 按真实标点切割（`。；，\n<space>`），而非硬性字符计数，语义单元保持完整；
- 采用粗→细两段式方案并支持提前退出：粗扫（大窗口、宽松阈值）低成本捕获全覆盖样本；仅当粗扫不完整时才跑细扫。

加窗是整个流水线可批处理的前提：它将变长响应转化为一组有界的定长块，使得单次批量编码和单次 GEMM 成为可能。

## 为什么用单次 GEMM

昂贵的语义比较只需要一次矩阵乘法：

```text
sim = reference_embeddings @ window_embeddings.T
```

这用一次稠密张量运算替代了 `num_steps × num_windows` 次独立的 Judge 风格语义判断。自回归解码（LLM Judge 真正的成本）被一次 encoder-only 前向 + 一次 GEMM 替换，单样本成本约低两个数量级。LRU 文本→embedding 缓存在 rollout batch 和训练 step 间复用编码，长运行 reward worker 的编码成本趋近于零。

## 为什么用单调 DP

贪心匹配容易产生级联错误：如果第一个 reference item 匹配到了 response 靠后的段落，剩余所有 items 都被迫在这个错误位置之后搜索。单调 DP 在相似度矩阵中搜索全局有序路径：

```text
dp[i,j] = max( dp[i,  j-1]                # 跳过 window j
               dp[i-1,j  ]                # 跳过 step i
               dp[i-1,j-1] + gain[i,j] )  # 匹配 (i,j)
gain[i,j] = sim[i,j] if sim[i,j] >= threshold else 0
```

阈值门控使低相似度单元格贡献为 0，因此 window 不能通过"最不差"选项来伪造匹配。

## 为什么 DP 在 CPU（numpy）上运行

DP 矩阵很小（`num_steps × num_windows`，加窗后通常几千个单元格）。在 GPU 上，带标量读取的逐格 Python 循环会触发每次 host↔device 同步（约 50–100µs 每个），主导运行时间并抹掉 GEMM 的优势。在 CPU numpy 上运行 DP 消除了所有 host↔device 同步。在真实的 `monotonic_align` 递推上实测，比 torch 逐格循环快约 50×（例如 50×256 矩阵从 120ms 降到 2.3ms）。因此架构根据每种设备实际擅长的工作进行拆分：GEMM 在 GPU/NPU 上，小 DP 在 CPU 上。

## 为什么需要独立的 `order_rate`

单调 DP 路径本身保证有序——因此"通过 DP 匹配的步骤占比"无法检测出顺序错乱的响应。`order_rate` 独立于 DP 路径计算：对每个匹配步骤取 argmax window（最可能实现该 step 的 response 位置），然后测量相邻 reference-step 对的 argmax window 按预期顺序出现的比例。这样，一个命中所有步骤但顺序错误的响应在 `match_rate` 很高时仍会得到较低的 `order_rate`。最终分数乘以两者：`score = match_rate × order_rate × max_score`。

## 为什么需要逐步候选 action

在 Agentic RL 中，一个 reference step 可能由多个 action 实现（不同工具、不同措辞）。扁平的 `list[str]` 会迫使用户选择一个规范描述，漏掉语义上等价的替代方案。因此 scorer 接受 `list[str | list[str]]`：值为列表的 step 是一组候选 action，**任一**候选匹配即算该步通过。机制上，每个 step 在相似度矩阵中的行是其候选行逐元素 **max**：

```text
sim_matrix[i, j] = max over candidates c of cos(step_i_candidate_c_emb, window_j_emb)
```

DP、阈值门控和 `order_rate` 保持不变——它们操作在逐步 max 相似度矩阵上。`matched_steps` 报告在匹配窗口上胜出的具体候选，因此诊断输出仍然告诉你*哪个* action 被实现了。未匹配的多候选 step 报告为候选列表用 ` | ` 连接。

## 复读防御（P0，可选）

模型可以通过**复读** reference step 名加填充词来骗过纯语义对齐（"首先我会定位根因，接着我仔细地应用修复……"）——没有真正执行，但每个 step 名都落进一个高相似度 window。可选的 trace gate 通过 `ScorerConfig(require_trace=True)` 或 veRL adapter 的 `RAISE_REQUIRE_TRACE=1` 环境变量开启，会把这类响应判为 0 分。

### 设计：响应级 gate

gate 检查**整条响应**（而非每个匹配 window）是否含执行证据，由 `TraceDetector.has_trace` 完成：

- **工具/执行痕迹**：工具标签（`<edit_file>`）、git diff 标记（`diff --git`、`@@`、`--- a/`、`+++ b/`）、代码块、函数定义/调用、仓库文件路径、测试结论（`PASSED`/`FAILED`/`passed`/`fail*`）。
- **推理/分析痕迹**：因果连接词（`because`、`due to`、`therefore`、`leads to`、`results in/from`）以及动词锚定的名词短语（`root cause is`、`the issue lies`）。名词短语信号用动词锚定，使 step *名*（如 `locate root cause`）无法自触发 gate。
- 推理检查前，先把所有候选 step 文本从响应中剥离，避免 step 名片段伪装成推理。

响应**两者皆无**即判为复读、打 0 分；否则正常评分，不做任何 per-step masking。

### 为什么是响应级而非逐步

早先的逐步变体会对缺少痕迹的 step-window 对做 masking。在 100 条真实合并 bug-fix PR（numpy、scipy、pandas、scikit-learn、matplotlib、sympy、requests、flask）上标定，它导致 **65/100 样本至少丢一步、genuine 数据 mean score 损失约 20%**，原因有二：

1. **准备型步骤天然无执行证据**——`read the issue` 只是描述 bug，既无工具调用也无因果句式。gate 无法验证它，47/100 次将其误杀。
2. **最佳相似度 window 往往不是证据 window**——DP 把 `apply the fix` 匹配到 PR body 的"I fixed it by…"，而真实代码 hunk 在另一个 window。gate 检查的是匹配 window，31/100 次误杀该步。

响应级变体同时消除两类问题：100/100 genuine PR 完整保留（mean score 不变，整零样本数 = 无 gate 基线，零额外误杀），同时 padded recitation 仍被完全杀死（demo：1.0 → 0.0）。`benchmarks/dump_scores.py --require-trace` 的 `denied` 列与 `[recall watch]` 行可量化任意新数据集上的假阴性率。

### 威胁模型与范围

gate 针对**懒散/填充式复读**——只复述 step 名而不真正执行。它**不**检测*部分伪造*（一条响应里真做一步、该步的痕迹让另外三步复读也蒙混过关）；那是更难、明确划在范围外的威胁。gate 是一个精度补丁：在 rollout 中观察到复读时开启，用真实数据切片标定 pattern，并持续关注 `denied` 召回指标。

## Reference Step 设计指南

语义对齐**不能验证内部动作**——它检查的是 response 文本是否*覆盖*了每个 step 的语义内容。step 措辞因此直接决定 reward 质量。

### 三层分级

| 层级 | 含义 | 示例 | Scorer 可靠度 |
| --- | --- | --- | --- |
| **observable** | 执行产物应出现在 response 中 | `run pytest on affected tests`、`patch src/module.py` | 高 — 工具标签、diff、测试结论 |
| **reasoning** | 分析性内容应出现 | `explain why the overflow occurs`、`locate the root cause` | 中 — 需因果/诊断性表述 |
| **proxy** | 内部准备动作；仅靠主题重叠匹配 | `read the linked issue`、`understand the bug report` | 低 — 匹配 bug 描述段落，非「真的读了」 |

训练前可用 `raise_scorer.confidence` 的 `classify_step()` / `classify_steps()` 审计 checklist。

### 经验法则

1. **把 step 写成可观测输出**，而非隐藏动作：用 `summarize the reported bug symptoms` 代替 `read the issue`。
2. **多种合法表述/工具时用候选 action**：`["open web source", "open knowledge base"]`。
3. **纯 proxy step 降权或省略**，勿让 checklist 被其主导。
4. **保留执行型 step**（改代码、跑测试、验证）——它们锚定信号、抗复读。
5. **`threshold` 在任务 hard negatives 上标定**，不要只用通用 paraphrase 集。

### 反模式

| 差的 step | 原因 | 更好的写法 |
| --- | --- | --- |
| `read the issue` | 任何 bug 摘要段都能匹配 | `summarize the reported failure mode` |
| `understand the problem` | 不可观测 | `identify the failing function from the stack trace` |
| 七个 proxy + 一个 test | reward 被不可验证覆盖主导 | 平衡 observable / reasoning / proxy |

## 置信度路由（Judge Fallback）

库内置**置信度路由器**，避免盲目信任语义 reward。

`assess_confidence(result, reference_steps)` 返回 `ConfidenceReport`：

- `confidence` — 综合置信度 `[0, 1]`
- `fallback_recommended` — 是否应 defer 到 LLM-as-Judge
- `reasons` — 可读触发原因（`low_match_rate`、`low_margin`、`high_proxy_step_fraction` 等）
- `step_tiers` — 逐步可观测性标签
- `signals` — 原始指标（`mean_matched_sim`、`mean_margin` 等）

veRL adapter 的 `compute_score(..., return_details=True)` 始终包含 `confidence`、`fallback_recommended`、`fallback_reasons`。

推荐生产循环：

```text
details = compute_score(response, extra_info=..., return_details=True)
if details["fallback_recommended"]:
    reward = llm_judge(response, rubric)
else:
    reward = details["score"]
```

在你的 rollout 切片上_tune `RoutingConfig`。运行 `benchmarks/reward_quality.py` 查看 genuine vs recitation 的 fallback 率。

`_score_once` stats 导出：

- `matched_sims` — 每个对齐 (step, window) 对的 cosine
- `mean_matched_sim` / `min_matched_sim`
- `mean_margin` — 每步最佳 window 与次佳 window 的平均差距

margin 低表示匹配模糊 —— 当 `RoutingConfig.min_margin > 0` 时是强 fallback 信号。真实 PR-fix 数据上 mean margin 常为 **~0.008**；默认 `min_margin=0` 不做硬门控，margin 只参与综合置信度。

`require_trace_when_high_match=True`（默认）时，`match_rate ≥ 0.95` 但**无执行/推理痕迹**会触发 `high_match_without_execution_trace`。这能在不开启硬 trace 打 0 分的情况下，区分 padded recitation（高 match、无 diff/代码）与 genuine PR（高 match、有 trace）。

## 复杂度

- 编码：对 `num_steps + num_windows` 个文本做一次批量 encoder 前向（重复输入下被 LRU 缓存摊平，趋近于零）。
- 相似度：一次 GEMM，`O(num_steps × num_windows × dim)`。
- DP：`O(num_steps × num_windows)`，在 CPU 上，加窗后通常在 1ms 以内。
