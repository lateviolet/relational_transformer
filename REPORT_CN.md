# 从随机游走到并行关系检索：阶段性研究总结

## 1. 出发点

Shi 与 Cao 的 *Towards Understanding Transformers in Learning Random Walks*
证明了一个很具体的 Transformer 算法：

- attention 学会选择当前位置的直接父节点；
- value/readout 矩阵学会一步 Markov 转移；
- 在 `0 < p < 1` 时，梯度下降可以达到 Bayes 最优准确率；
- 在确定性游走 `p=0,1` 时，零初始化容易停在对 token 取平均的错误解。

这说明 Transformer 的泛化不能只用“表达能力够不够”解释，训练动力学和求解到的算法同样重要。项目从这里出发，把“单个状态选择 + 转移”扩展为“多个独立查询 + 并行/串行检索”。

## 2. 为什么原始 two-hop 任务不能说明多头必要性

最早的关系 cued two-hop 任务中，一个 head 和多个 head 的差别看起来很大。但控制有效 head dimension 后，一个 `head_dim=32` 的单头模型达到 99.47%（10 seeds，最低 98.999%），与双头模型基本相同。

因此 original task 上的优势主要来自 shared `d_model` 如何被划分为 head、参数化和优化条件，而不是“必须有两个独立 softmax”。

## 3. 单 token key-value paired retrieval

为消除边 token 需要两步绑定带来的混淆，构造了更干净的任务：

- context 是若干单 token `(key, value)` record；
- 最后给两个不同 key；
- 必须同时输出两个 value，并计算 joint exact match；
- key/value 在一题内唯一，record 顺序随机；
- 没有重复答案、频率或固定位置捷径。

在 16 records、`d_model=96`、`head_dim=48` 上，双头模型 10/10 成功，单头 8/10 成功；双头模型的两个 head 分别选择两个答案，ablation 呈答案特异性。

## 4. 32-record 固定 schedule 下的强分离

在 32 records、两个 query、10,000 steps、20 seeds 下：

| architecture | mean joint | success >=95% | min |
|---|---:|---:|---:|
| 2 layers, 2 independent heads | 85.15% | 17/20 | 0% |
| 2 layers, 1 head, `head_dim=48` | 0.62% | 0/20 | 0% |
| 2 layers, 2 tied heads | 1.20% | 0/5 | 0% |
| 2 layers, 1 head, parameter-matched | 0% | 0/5 | 0% |
| 2 layers, 2 independent heads, no absolute positions | 80.55% | 4/5 | 2.76% |

参数匹配的单头模型是 `d_model=105`, `head_dim=48`，共 269,274 参数；双头模型是 `d_model=96`, `head_dim=48`，共 269,568 参数。两者只差 0.11%，单头仍全部失败。

成功模型的机制非常明确：

- head 0 选择查询 1 对应的 record；
- head 1 选择查询 2 对应的 record；
- 删除一个 head 只破坏对应答案；
- 交换 query 后，答案随 query 正确交换；
- query-slot activation patching 也呈答案特异性。

失败 checkpoint 则有两种模式：

- 两个 head 都选择第一个 query，形成单答案解；
- 两个 head 都没有找到 query record，形成近似均匀注意力。

如果实验只做到这里，很容易得出“该任务需要多头”的结论。但进一步的控制推翻了这个解释。

## 5.  Curriculum 揭示 depth-head 替代关系

保持数据、模型宽度和有效 head dimension 不变，只把 records 从 16 线性增加到 32：

- 两层、单头模型：4/5 成功，平均 joint accuracy 78.94%；
- 64 records 使用 16→64 curriculum：两层单头 4/5 成功，平均 80.31%；双头 5/5，平均 99.98%。

保存成功单头 checkpoint 后，activation patching 显示：

- layer 0 的 attention 主要选择答案 2 对应 record；
- layer 1 则负责答案 1；
- 替换 layer-0 head output 只改变答案 2；
- 替换 layer-1 head output 主要改变答案 1。

也就是说，双头模型用两个 head **并行**做两个 lookup；单头模型用两层 **串行**做两个 lookup。

更激进的控制中，一层、一个 head 也在 5 个 seed 中有两个达到约 94–95% joint accuracy。成功 checkpoint 的单一 attention map 同时给答案 1 record 58.2% 质量、答案 2 record 24.3% 质量，两个输出分类器从同一个多峰 attended vector 中恢复两个答案。

因此该 interface 既不需要两个 head，也不需要两层；更准确的问题是：固定参数/FLOP budget 下，优化更偏好 parallel heads 还是 serial depth。

## 6. 三查询扩展

将 record model 推广到任意 query 数。三个 query、32 records、相同 `d_model=132` 和 `head_dim=32`、15,000 steps、16→32 curriculum：

| depth | heads | mean joint | success >=95% | min |
|---|---:|---:|---:|---:|
| 1 | 1 | 0.002% | 0/5 | 0% |
| 1 | 2 | 76.37% | 3/5 | 0.71% |
| 1 | 3 | 99.995% | 5/5 | 99.98% |
| 2 | 1 | 0.49% | 0/5 | 0% |
| 3 | 1 | 0.012% | 0/5 | 0% |

三头 checkpoint 几乎完全职责分离：

- head 0 在 answer-1 record 上放 98.8% 质量；
- head 1 在 answer-2 record 上放 98.7%；
- head 2 在 answer-3 record 上放 98.5%；
- ablation 每次只让对应答案降到 0.6–0.8%。

成功的双头 checkpoint 则用 head 0 同时承载 answer 1/2（注意力质量约 27.8% 与 62.7%），head 1 负责 answer 3（94.5%）。这进一步证明：head 数增加提升的是每个 query 拥有独立归一化选择分布的概率和稳定性，而不是提供单头绝无可能计算的基元。

另一个诊断 seed 使用更慢的 8→32 curriculum、30,000 steps 和 `5e-4` 学习率训练 3 层单头，但最终 joint accuracy 仍只有 4.55%，三个答案分别约 83.7%、42.1%、42.2%。这说明“纯串行深度”并不是只要训练更久就会自动成功；三种资源策略的优化难度明显不同。

这比二查询实验更清楚地展示 parallel routing。但两个 head 仍能在多数 seed 上接近解三个 query，所以它仍是 reliability/capacity curve，而不是严格的 head 数量定理。

四查询的扩展给出更陡的曲线。在单层、32 records、`d_model=180`、`head_dim=32`、20,000 steps 下：

| heads | mean joint | success >=95% | min |
|---:|---:|---:|---:|
| 1 | 0% | 0/5 | 0% |
| 2 | 0.16% | 0/5 | 0% |
| 3 | 19.80% | 1/5 | 0.73% |
| 4 | 99.96% | 5/5 | 99.93% |

三头模型通常能学会四个答案中的三个，四头模型稳定学到全部四个。由于总参数量只从 410,504 增至 479,912（增加 16.9%），这一跃迁主要不是单纯参数预算增加造成的。

## 7. 自然语言迁移

用 Qwen3-0.6B Base 做 16 条 `reports to` facts 的双查询任务，要求按顺序输出两个名字。Base 模型在 32 个样本上 0 个完全正确；LoRA 训练 300 steps 后，六个 seed 在每项 128 examples 上使用严格 normalized exact match：

| condition | mean |
|---|---:|
| held-out names | 99.87% |
| unseen prompt templates | 100% |
| 32 facts | 99.87% |
| order-swapped queries | 99.87% |
| unseen `supervises` wording | 99.87% |

这些结果证明双答案检索可以迁移到真实 tokenizer 和 language model，但仍是模板化合成语言，不等于开放域多跳推理。

对 seed 0 adapter，我们进一步做了 28 层 × 16 个 attention head 的逐头 ablation。按 answer-token loss 排序，最关键的是 layer 13/head 6、layer 2/head 6、layer 0/head 3。严格生成评估中，baseline 为 100%，单独删除这三个 head 后分别降到 19.53%、41.41%、35.94%；删除一个无关的 layer 0/head 0 仍为 100%，同时删除三个关键 head 后降到 0%。两组关键 head 的 pair ablation 也为 0% 或 9.38%，而随机删除三个无关 head 仍为 100%。

在独立的 seed 1-4 adapter 上，layer 13/head 6 ablation 只剩 10.94%–25.00%，layer 2/head 6 只剩 34.38%–48.44%；三个候选 head 一起删除时四个 seed 都是 0%。因此这两个主要 retrieval heads 的位置在独立训练间稳定，第三个 head 的重要性则有 seed 差异。当前跨 seed 实验只在 seed 0 找出的候选上做 targeted ablation，没有对 seed 1-4 全量扫描 448 个 head。

在 seed 0 的跨分布测试中，32 facts 下删除 layer 13/head 6 和 layer 2/head 6 分别降到 20.31% 和 45.31%；unseen templates 下分别为 23.44% 和 45.31%；query swap 下为 28.13% 和 43.75%；unseen `supervises` wording 下为 15.63% 和 6.25%。四个条件的共同结果都是：三个 head 一起删除后为 0%。这说明这些不是只对训练模板有效的表层 artifact。

## 8. 主要结论

1. “多头必要性”不能通过一个看似组合化的任务自动得到。
2. 当前 32-record、固定 schedule 的结果是真实的 optimization separation，但不是表征层面的下界。
3. 多个独立查询可以由三种计算方案完成：
   - 一层多头 parallel routing；
   - 多层单头 serial routing；
   - 一层单头的多峰 attention + output decomposition。
4. 增加 query 数时，增加平行 head 会显著提高成功率；三查询下 1/2/3 head 是 0/5、3/5、5/5，四查询下 1/2/3/4 head 是 0/5、0/5、1/5、5/5。
5. 在 Qwen3-0.6B LoRA 上，同一组关键 retrieval heads（特别是 L13/H6 和 L2/H6）在 5 个独立 adapter 以及 4 类分布变化中都保持因果重要性。
6. 更值得研究的是固定预算下的 head-depth allocation、优化相的边界，以及不同任务是否会把检索功能路由到同一类 attention heads。

## 9. 下一步

- 构造真正要求两个独立归一化分布的输出接口，并在明确的 restricted readout family 下证明下界。
- 做 4-query 或更多 query 的 scaling law，比较 head 数和 layer 数。
- 在 Qwen adapter 中定位负责两个查询的 attention heads，做 head ablation 与 activation patching。
- 测试同姓实体、同义关系、换序事实和长上下文，避免模板化语言造成的虚假泛化。
