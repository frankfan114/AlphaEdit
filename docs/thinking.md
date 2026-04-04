
## A. 在 edited model 里，“target_true 的知识分布”到底是什么？

对同一个 neighbor prompt（重要：是 neighbor prompt），你跑一次 causal trace：

* 设 `expect = target_true`
* 得到 `scores_true = S_true ∈ R^{T×L}`（你脚本里 result["scores"]）

它的每个元素 `S_true[t,l]` 表示：

> subject span 被破坏时，仅恢复 (t,l) 的 state，`p(target_true)` 能恢复到多少。

因此在你的框架里：

> **target_true 的知识分布 = `S_true` 在 token×layer 平面上的“高值区域结构”**
> （哪里高、是否集中、是否落在 subject span、是否与 base 同一条因果通路）

---

## B. 你应该用哪些指标去“看懂” edited model 的 `S_true`？

我建议你只用 5 个指标就够了（它们分别回答 5 个你真正关心的问题）。

> 记：同一次 trace 里你还会有 `high_true = p_true(clean)`、`low_true = p_true(corrupt,no patch)`。

### 指标 1：**True Existence（存在性/强度）**

看 `high_true`（干净输入下 `p(target_true)`）：

* `high_true_edit` 是否明显比 `high_true_base` 下降？
* 更稳健一点：看 margin
  `m_true = log p(true) - log p(competitor)`
  competitor 可以先取 `target_new`，也可以取 top-1（更严格）

解释：

* 如果 `high_true_edit` 或 `m_true_edit` 大幅下降：**不是“分布在哪里”，而是它在行为层面已经不占优势**。

---

### 指标 2：**True Recoverability（可恢复性）**

用你已有的 `low_true` 和 `high_true` 定义一个“恢复阈值”：

[
thr = low_true + 0.5\cdot(high_true-low_true)
]

然后：

* `R_true = mean( S_true >= thr )`

解释：

* `R_true` 大：target_true 仍然有很多 state 能把它“拉回来”（通路没死）
* `R_true` 小：**target_true 的因果通路塌缩**（你想找“在哪”会很难，因为几乎哪都不强）

> 这就是你要的“在 edited model 里原来 true 的知识还剩多少”的最直接数值。

---

### 指标 3：**Localization（集中在哪）**

你要知道它是“集中在少量 token×layer”还是“弥散”。

做一个很实用的集中度指标（无需复杂统计）：

* 取 Top-K 单元（例如 K = 50 或 K = top 1%）
  `TopK_true = {(t,l) | S_true 属于最大的K个}`

然后算两件事：

1. **是否集中在 subject span**
   设 subject token range 是 `[b,e)`：

* `Frac_subj = |{(t,l)∈TopK_true : t∈[b,e)}| / K`

解释：

* `Frac_subj` 高：true 的关键恢复点依赖 subject 表征（绑定更实体化）
* `Frac_subj` 低：true 依赖更全局/模板化路径（更像 pattern / relation）

2. **集中在哪些层**

* `LayerMass[l] = mean_t S_true[t,l]`
  看峰值层 `argmax_l LayerMass[l]`

解释：

* 峰值层跑到你 edit 影响的层附近：说明编辑直接重塑了 true 的通路
* 峰值层整体漂移：说明出现了 representation re-routing（重路由）

---

### 指标 4：**Base–Edited Similarity（和 base 的关系是什么）**

你问的“它和 base 上有什么关系”，最有用的不是看 heatmap 眼对眼，而是算相似度：

把 `S_true` 拉平为向量 `v`（T×L 展平），归一化后算 cosine：

* `Sim_true = cos( normalize(vec(S_true_base)), normalize(vec(S_true_edit)) )`

解释：

* `Sim_true` 高：通路位置类似，只是强度可能变了（“同一条路变弱”）
* `Sim_true` 低：**通路搬家了**（re-localization）

---

### 指标 5：**True-vs-New Competition（“true 没了”还是“new 抢了”）**

这是你最终要解释 neighbor 失败的关键：到底是 true 通路塌了，还是 new 通路在同一位置更强？

所以你需要同一个 neighbor prompt 再跑一次：

* `expect = target_new`
* 得到 `S_new`

构造“偏好图”：

[
Pref = S_{true} - S_{new}
]

然后再比较 base vs edited 的变化：

[
ShiftPref = Pref_{edit} - Pref_{base}
]

解释：

* 如果 edited 的 `Pref` 在关键区域变负：**同一批关键 states 变成更支持 new**
* 如果 `S_true` 整体下降但 `S_new` 没怎么上升：更像 true 通路受损而非被抢占

> 这一步是把“知识分布”真正连接到“为什么 neighbor 不再倾向 true”的桥梁。
> 只看 `S_true` 你无法严谨地区分“塌缩” vs “抢占”。

---

## C. 你如何从这些指标出发，提出一个能改进 AlphaEdit 的新方法？

核心思路：把你现在“诊断用”的量（`S_true`, `Pref`, `ShiftPref` 等）变成**训练时的约束/正则项**。这就是从分析走向方法的标准路径。

我给你 3 条可写进论文、也可落地实现的改进方向（从易到难）：

---

### 方向 1：**Neighbor Preservation Loss（最直接、最像论文 baseline）**

在编辑时除了让 subject prompt → target_new，还显式加入 neighbor prompts 的保持目标：

* 对 neighbor prompts：最大化 `p(target_true)`，或最大化 margin `log p(true)-log p(new)`

目标函数示意（单 case）：

[
\max_{\Delta W};; \underbrace{\log p_{\theta+\Delta W}(new \mid \text{edit prompt})}*{\text{make edit succeed}}
;+;\lambda\underbrace{\mathbb{E}*{x\in \text{neighbors}}\log p_{\theta+\Delta W}(true \mid x)}_{\text{preserve neighbors}}
;-;\beta |\Delta W|^2
]

它解决的就是你现在的问题：**编辑训练里根本没有强制 neighbor 维持 true**，所以会退化。

你用前面的指标验证它是否有效：

* `Δm` 应该回正（true 相对 new 恢复）
* `R_true` 上升
* `ShiftPref_subj/nonsubj` 变小

---

### 方向 2：**Causal-Path Preservation（把 “S_true 分布” 当作要保留的对象）**

你最想要的其实是：编辑不应该改变 neighbor prompt 下 target_true 的“因果通路”。

那你就直接对齐 `S_true`（或 Pref）：

对 neighbor prompts，加入：

[
\mathcal{L}*{path}
= \mathbb{E}*{x\in neighbors}; | \phi(S^{base}*{true}(x))-\phi(S^{edit}*{true}(x))|
]

其中 (\phi(\cdot)) 可以选：

* Top-K mask（只保留关键单元）
* 或 layer mass 向量（更便宜）
* 或 normalize 后的 flatten cosine distance

直观解释：

* 不仅要让输出仍然是 true，还要让 “支持 true 的内部路径” 不被你编辑搞乱
* 这样更不容易出现你观察到的“neighbor 的知识偏移”

为什么这很有研究味道：

* 你把 causal tracing 从**分析工具**变成**约束信号**（causal-guided editing）

---

### 方向 3：**Subspace / Gradient Orthogonalization（解释 AlphaEdit 失败的结构性原因）**

AlphaEdit 类方法常用“在某层做低秩更新，并通过某种约束减小副作用”。你现在的失败可以用一个更具体的结构性解释来推动新方法：

> neighbor prompts 的 true 通路与 edit 目标通路在参数梯度/表示子空间上高度重叠，
> 所以即使你做了局部编辑，也不可避免伤到 neighbor 的 true。

如何把它变成方法：

* 取 neighbor preservation 的梯度 (g_{nbr})
* 取 edit success 的梯度 (g_{edit})
* 做投影/正交化，让更新方向尽量不动 neighbor 方向：

[
g_{edit}^{\perp} = g_{edit} - \text{Proj}*{g*{nbr}}(g_{edit})
]

再用 (g_{edit}^{\perp}) 去更新（或约束低秩更新的左右子空间）。

你用前面指标去证明它确实减少偏移：

* `ShiftPref` 的正质量下降
* `Sim_true` 上升（通路更接近 base）
* `Frac_subj` / layer peak 不乱漂

这条路径特别适合你写“为什么 AlphaEdit 没做好 neighbor”的 mechanistic hypothesis：

* **因为编辑方向与 neighbor true 的保持方向冲突**，需要显式处理冲突子空间。

---

## D. 给你一条最实用的落地路线（不用先“想明白所有哲学”）

你就按这个做，很快就能得到可写的结论和新方法雏形：

1. 在 neighbor prompt 上收集（base vs edited）：

   * `high_true`, `margin_true_vs_new`
   * `R_true`
   * `Frac_subj`
   * `Sim_true`
2. 补跑 `S_new`，得到：

   * `Pref = S_true - S_new`
   * `ShiftPref = Pref_edit - Pref_base`
   * 分区统计：`ShiftPref_subj` vs `ShiftPref_non`
3. 用这些数值归因：

   * `R_true↓`：true 通路塌缩
   * `Pref 翻负`：new 抢占关键 states
   * `Sim_true↓`：通路重路由
4. 基于归因选方法：

   * 如果是塌缩：优先做 **Neighbor Preservation Loss**
   * 如果是抢占：做 **Pref/ShiftPref 约束** 或 **梯度正交化**
   * 如果是重路由：做 **Causal-Path Preservation**

