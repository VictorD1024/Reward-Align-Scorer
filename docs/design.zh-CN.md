# 设计

<p align="right">
  <a href="./design.md">English</a> |
  <a href="./design.zh-CN.md">中文</a>
</p>

Reward Align Scorer 的核心工程选择：

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

## 复杂度

- 编码：对 `num_steps + num_windows` 个文本做一次批量 encoder 前向（重复输入下被 LRU 缓存摊平，趋近于零）。
- 相似度：一次 GEMM，`O(num_steps × num_windows × dim)`。
- DP：`O(num_steps × num_windows)`，在 CPU 上，加窗后通常在 1ms 以内。
