# Yuan Cao 论文阅读笔记

## 1. Towards Understanding Transformers in Learning Random Walks

Wei Shi and Yuan Cao. 重点讨论一篇一层的简化 Transformer 是否能从数据中学会圆环上的随机游走，以及训练后的权重究竟在做什么。

### 任务设定

- 有 `K` 个离散位置，每一步以概率 `p` 顺时针走一步，以 `1-p` 逆时针走一步。
- 输入是长度 `N` 的游走历史 `s_1,...,s_{N-1}`，目标是预测 `s_N`。
- 每个位置用 `K` 维 one-hot `x_t` 表示。
- 有一个可学习的线性 value/readout 矩阵 `V`，以及一个注意力参数矩阵 `W`。
- 输出类别分数为

```text
f_theta(X) = V X S(X_with_pos^T W e_{x_N})
```

其中 `S` 是 softmax，`e_{x_N}` 是最后一个位置的 query token。

### 为什么 value 会变成转移矩阵

Markov 随机游走的条件分布只取决于当前状态。圆环上的转移矩阵为

```text
Pi[i,j] = P(next = j | current = i).
```

如果注意力已经选中当前位置的直接父节点 `x_{N-1}`，那么理想预测应当是

```text
Pi^T x_{N-1}.
```

而 `V x_{N-1}` 正好是用同一个线性映射把状态 `x_{N-1}` 转成下一状态分数。因此训练出的 `V` 会逼近 `Pi^T`。这说明“value 成为转移矩阵”不是额外假设，而是该任务中唯一自然的线性 map。

位置编码不需要进入 `V`，因为转移概率相对于序列位置是共享的：一旦知道“当前是哪个状态”，下一步分布由这个状态决定。位置编码负责区分历史中重复出现的同一个状态，帮助 attention 选中正确的那一次出现；`V` 负责状态到下一状态的概率映射，两者分工不同。

### 训练后得到的两段机制

在 `0 < p < 1`、`K > 4`、`N` 足够大、zero initialization 和梯度下降等条件下，论文证明：

1. 训练后的模型达到 Bayes 最优准确率 `max(p, 1-p)`；
2. 模型逐渐收敛到最优 predictor；
3. `V` 的方向收敛到 `Pi^T`；
4. softmax 把接近 1 的权重放在直接父节点 `x_{N-1}` 上，其他 token 的权重指数小。

因此可以概括成：

```text
attention: 选出 current state
V:         current state -> next-state distribution
```

### 为什么位置编码需要存在

如果没有位置信息，同一个位置在序列中反复出现时，token embedding 完全相同。仅看当前 token 的内容，attention 无法区分“上一个位置”与更早出现的同一状态。对于一阶 Markov 链，预测只应由最近父节点决定，但模型必须能从上下文中知道哪一个 occurrence 才是父节点。

论文采用特殊的正弦位置码：

```text
p_i = [sin(i*pi/(M+1)), sin(2*i*pi/(M+1)), ..., sin(M*i*pi/(M+1))]
```

它们两两正交，因此不同序列位置有可区分的几何表示。这样 attention score 可以同时利用：

- query/state 匹配：找到与当前状态相关的历史；
- position 匹配：在重复状态中偏向最近父节点。

### 为什么维度是 3/2 次方

这里的 `M` 是位置嵌入维度，不是节点数 `K`，也不是序列总长度。论文取

```text
M = Omega(N^(3/2)).
```

正交性质本身只要求位置向量数量足够表示 `N` 个位置；`N^(3/2)` 来自证明中的信号-噪声分离要求。原因是位置向量自内积约为 `M`。在训练早期，softmax logit 中与“父节点 vs 其他位置”相关的优势同时依赖这个内积和序列归一化，粗略量级为

```text
logit gap ~ M^2 / N^2.
```

为了让父节点权重达到 `1 - exp(-Omega(N))`，需要 logit gap 至少是 `Omega(N)`。因此

```text
M^2 / N^2 = Omega(N)
=> M^2 = Omega(N^3)
=> M = Omega(N^(3/2)).
```

所以 3/2 次方不是为了表达 extra semantics，而是为了让正交位置编码提供足够强的梯度/注意力分离信号。它属于理论证明的充分条件；实际 LLM 不会使用随上下文长度 3/2 次方增长的绝对位置维度。

### 负结果

当 `p=0` 或 `p=1` 时，随机游走退化为确定性游走。论文证明 zero initialization 下梯度下降会失败，模型倾向于对 token 取平均，不能把注意力集中到直接父节点。这说明定理里的 `0 < p < 1` 不是技术性瑕疵，而是刻画了一个真实的 optimization phase transition。

### 论文贡献与局限

贡献：

- 把“Transformer 学会了算法”具体化成可证明的 attention-selection + value-transition 两段计算；
- 给出有限步梯度下降的训练动力学，而不只是证明某个最优解存在；
- 统一解释随机游走和确定性游走的 success/failure，并延伸到简单 QA 任务。

局限：

- 使用的是单层、无可训练 Q/K/V 分解、只有一个 attention operator 的简化线性 Transformer；
- 假设 natural ordering 已知，并用特殊的正交位置码避免复杂位置建模；
- 证明依赖 zero initialization、特定损失和足够长的序列；
- 不能直接推出真实多层 LLM 的全部机制。

## 2. On the Robustness of Transformers against Context Hijacking for Linear Classification

Tianle Li, Chenyang Zhang, Xingwu Chen, Yuan Cao, Difan Zou.

论文研究“上下文劫持”：上下文中的例子在事实上正确，但语义上接近 query 边界且标签相反，从而诱导 Transformer 输出错误类别。理论设定是 multi-layer linear Transformer 上的 in-context linear classification。

主要结论：

- 把一层 Transformer 看成在 in-context examples 上执行一步优化；层数增加相当于更多、更细粒度的优化步。
- 在给定 context length、hijacking token 数和模型深度后，可推导出 robust accuracy 的函数形式。
- 更深的 Transformer 对 hijacking 更鲁棒，因为它能进行更精细的多步优化，而不是把局部“反例”错误放大。
- 合成实验验证了 depth 与 robustness 的预测关系。

这篇工作与随机游走论文的共同风格是：不把深层网络当成黑盒，而是给出它能执行的具体 iterative algorithm，并研究该算法在数据/干扰下的行为。

## 3. Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks

Xuan Tang, Han Zhang, Yuan Cao, Difan Zou.

研究对象是 mini-batch Adam/AdamW 在两层 over-parameterized CNN 上的泛化。核心问题是：为什么 full-batch Adam 的理论结论不能直接解释实际使用的 stochastic Adam。

主要结论：

- stochastic Adam 即使学习率趋于极小，也不会简单收敛到 full-batch Adam 的对应解。
- 在 noise-dominated 的非凸设定中，large-batch Adam/AdamW 可能收敛到测试误差很差、主要记忆 noise 的解。
- 适当 batch size 的 stochastic Adam/AdamW 可以通过随机梯度隐式抑制 noise fitting，同时保留 signal feature learning；weight decay 再显式压低残留 noise，从而接近零 test error。
- Adam 的 adaptive normalization 会放大 weight decay 的有效影响，因此 Adam 对 `lambda` 更敏感；AdamW 的解耦 weight decay 允许更大的有效范围。
- 在常数 beta 等条件下，可把 stochastic Adam/AdamW 严格近似为 SignSGD/SignSGDW，从而给出新的训练动力学分析。

这篇论文关心的是 optimizer、batch noise 和 implicit regularization 如何共同决定“记住噪声”还是“学习信号”。它与此前 benign overfitting 的理论问题直接相连。

## 4. Benign Overfitting in Two-layer Convolutional Neural Networks

Yuan Cao, Zixiang Chen, Mikhail Belkin, Quanquan Gu. NeurIPS 2022.

论文研究过参数化神经网络中“训练集完全插值，但测试仍然很好”的 benign overfitting。数据由 signal patch 和 label-independent noise patch 组成。

主要结论：

- 对两层 CNN，梯度下降可以同时学到信号和拟合噪声。
- 当 signal-to-noise ratio 满足条件时，signal learning 快于 noise memorization，最终虽然训练误差接近零，population loss 也趋近零。
- 当 SNR 不满足条件时，网络主要依赖 noise patch，得到常数级 test loss。
- 两者之间存在 sharp phase transition，界线由 SNR 决定。

证明技术是 signal-noise decomposition：把卷积 filter 分解为初始化、signal 与 noise components，并将训练转成这些系数的离散动力系统。关键洞见是过拟合本身并不必然有害；是否 benign 取决于模型在训练动力学中先学到信号还是先记住噪声。

## 四篇论文的共同线索

1. **算法化解释**：Transformer/CNN 的泛化不是只看参数量，而要看训练过程实际实现了哪条算法。
2. **优化与表示同等重要**：随机游走中 value 表示正确不等于能收敛；Adam 中表达力足够不等于泛化好。
3. **初始化与数据条件会产生相变**：随机游走在 `p=0,1` 失败，benign overfitting 在 SNR 阈值两边表现不同。
4. **可解释性需要因果干预**：只看 attention map 不够，还需要 head ablation、activation patching、不同初始化下的成功/失败 checkpoint 对照。

## 对本项目实验的直接含义

原始 relation-cued two-hop 任务不能证明多头必要性，因为最终是一个单答案分类问题，且窄单头可以匹配多头结果。新的 paired-retrieval 任务把问题改成两个独立查询、两个输出槽、joint exact match，才让“一个 softmax 分布还是两个独立归一化分布”成为实际瓶颈。

固定训练 schedule 下，32-record、20-seed 结果曾显示很强的 separation：两个独立 head 17/20 成功，匹配单头和 tied-head 全部失败。但 curriculum control 推翻了这个解释：16→32 curriculum 让两层单头 4/5 成功；两层单头 checkpoint 通过 layer 0 处理一个查询、layer 1 处理另一个查询，实现了 serial retrieval；更进一步，一层单头也在 5 个 seed 中有 2 个达到约 94–95% joint accuracy。

因此目前能下的结论不是“这个任务要求多头”，而是：同一任务存在至少三种 solution family（一层单头、单头多层、单层多头），训练 schedule 会改变模型选择哪一种。多头带来的优势更可能是并行性、优化可靠性和避免 interference，而不是某个逻辑上无法由单头实现的功能。这是一个重要的 negative result，也提示下一步应研究 fixed parameter/FLOP budget 下的 parallel-head vs serial-depth tradeoff，或直接要求输出两个分别归一化的 attention distribution。

三查询扩展进一步给出了多头真正发挥作用的证据：在单层、32 records、相同 `d_model=132` 和 5 个 seed 下，1/2/3 个 head 的 joint accuracy 分别为 0.002%、76.37%、99.995%，`>=95%` success 分别为 0/5、3/5、5/5。三头模型的三个 head 分别以 98.8%、98.7%、98.5% 的注意力质量选择三个 query 对应的 record，ablation 也只破坏对应答案。双头模型中，head 0 同时给 answer 1/2 分配约 27.8%/62.7% 的 mass，head 1 则主要给 answer 3 分配 94.5%，说明单个 head 可以通过多峰分布 multiplex 两个 query。另一个更慢的 30k-step、3 层单头 seed 最终 joint accuracy 仅 4.55%，说明串行 depth 方案也不是简单增加训练步数就会自动成功。这是非常清晰的 parallel routing，但仍然不是“每增加一个 query 就必须增加一个 head”的定理，因为两个 head 已在多个 seed 上接近或达到解。

四查询将统计曲线拉得更开：单层模型 1/2/3/4 个 head 的 mean joint accuracy 为 0%、0.16%、19.80%、99.96%，success 为 0/5、0/5、1/5、5/5；参数量仅从 410,504 增至 479,912。这说明额外的独立 attention distribution 更像是在提供 parallel selection channels，成功率主要由“能否为每个 query 稳定形成独立通道”决定，而不是总参数量的平滑收益。
