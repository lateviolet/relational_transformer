# Experimental Findings and Research Log

These are controlled, preliminary results rather than paper-ready claims.

## Current headline

- The original relation-cued task does not establish multi-head necessity:
  changing effective head dimension removes the apparent advantage.
- On the cleaner paired-retrieval task, fixed-schedule training creates a large
  two-head versus one-head gap at 32 records, but curriculum training lets a
  one-head model solve the same task.
- A one-layer one-head model can sometimes encode both answers in a single
  multi-peaked softmax; a two-layer one-head model serializes the two lookups
  across layers.
- Increasing the number of queries to three gives a clearer reliability curve:
  one/two/three heads succeed on 0/5, 3/5, and 5/5 seeds respectively in one
  layer. The four-query extension succeeds on 0/5, 0/5, 1/5, and 5/5 seeds
  for one through four heads. This is a capacity/reliability effect, not a
  strict impossibility theorem.
- Six Qwen3-0.6B LoRA adapters average 99.87% strict exact match across held-out
  names/templates, 32-fact contexts, query-order swaps, and unseen relation
  wording.
- A causal head screen identifies layer 13/head 6 and layer 2/head 6 as stable
  retrieval heads across five adapters; ablating both plus layer 0/head 3
  reduces every adapter to 0% exact match.

## Hardware

Experiments ran on shared GPU hosts with RTX 3090 and A40 GPUs. Concurrent jobs
were assigned explicitly to individual devices rather than relying on a shared
default device. The reference environment used Python 3.10, PyTorch 2.7, and
Transformers 5.15.

## Tasks

- `two_hop`: two-hop pointer chasing among shuffled directed edges.
- `cued_two_hop`: two relation chains coexist; the task token selects which
  relation chain to follow.
- Natural-language version: two natural-language relation chains, held-out
  entity names, shuffled facts, three training templates, and up to eight
  distractor edges.

## Synthetic architecture sweep

Ten seeds, 5,000 steps, 12 nodes, three distractors, `d_model=128`:

| architecture | mean accuracy | std | min | max |
|---|---:|---:|---:|---:|
| 1 layer, 2 heads, MLP | 0.499 | 0.004 | 0.492 | 0.504 |
| 2 layers, 1 head, MLP | 0.896 | 0.202 | 0.484 | 1.000 |
| 2 layers, 2 heads, no MLP | 0.718 | 0.215 | 0.355 | 1.000 |
| 2 layers, 2 heads, MLP | **0.997** | **0.001** | **0.995** | **1.000** |

Main interpretation:

- Depth is the strongest bottleneck: one layer remains near its partial-solution
  plateau even with multiple heads and an MLP.
- One head can sometimes solve the task, but its optimization basin is unstable.
- Multiple heads without an MLP can also solve some seeds, but the success
  rate is much lower.
- The combined 2L-2H-MLP architecture is reliable under this training budget.

At a 95% accuracy threshold, 2L-2H-MLP succeeds on all 10 seeds, while
2L-1H-MLP succeeds on 8/10. The same comparison is 10/10 versus 8/10 at three
layers and 10/10 versus 9/10 at four layers.

A tied-head control, where both heads use identical Q/K/V parameters, reaches
99.8% over the first five seeds but 94.9% over ten seeds (9/10 above 95%). A
one-head model with its output projection scaled by `sqrt(2)` reaches 79.6%
over five seeds, so simple output scaling does not explain the tied-head
result. This control weakens any claim that two *specialized* attention heads
are necessary on the current task. The head-count effect may instead come from
parameterization or optimization conditioning.

A direct narrow-single-head control settles the issue for this benchmark:
keeping `d_model=128` but using one attention head with `head_dim=32` reaches
99.47% over ten seeds, with a 10/10 success rate and minimum 98.999%. This
matches the two-head result while using one softmax distribution and fewer
parameters. Therefore, the current task cannot prove two-head necessity: its
apparent multi-head advantage is due to the shared `d_model` being partitioned
into smaller heads, not to two independent attention distributions.

The uncued two-hop task requires depth more clearly: a single seed gives 33.7%
for 1L-1H without MLP and 97.4% for 2L-1H without MLP.

## Mechanism probes

For the fixed-length, cued task:

- After layer 0, the final-answer position linearly decodes the first-hop node
  at 56.3%; the final answer is only 39.5% decodable.
- After layer 1, the final position decodes the final answer at 99.6%.
- A failed one-head run linearly encodes the intermediate almost perfectly
  (99.5%) but decodes the answer only at 56.6%. This is an optimization/readout
  failure, not a failure to represent the first hop.
- Ablating any attention head in the successful 2L-2H model causes a large
  accuracy drop, so the heads are not simply redundant copies.

The learned computation is not always a textbook "query position produces
intermediate, answer position produces final result" pipeline. In the
fixed-length model much of the first-hop work is already written at the
final-answer position.

## Length generalization

- Training only on three distractors gives 99.7% at length three but falls to
  53.2% with four and 35.0% with five.
- Training on larger contexts can transfer downward: training with five
  distractors gives 98.9% at five, 96.0% at four, and 98.0% at three.
- Training with two or three and testing on four or five still drops to 62.0%
  and 48.5%, respectively.
- A larger 16-node/five-distractor task retains the depth effect: 2L-2H-MLP
  reaches 98.9%, while 1L-2H-MLP stays at 20.5%.
- At that larger size, 2L-1H-MLP reaches 95.2%; removing the MLP from 2L-2H
  reaches 92.3% in-distribution but only 7.6% with three distractors and 7.8%
  with four. The MLP therefore contributes strongly to cross-length
  robustness.

This suggests that context-length generalization is directional and is not
explained only by whether the model learned the two-hop algorithm.

## Position encodings

The absolute-position and RoPE runs use the same 64-position capacity, width,
head count, MLP, data, and optimization budget. Ten seeds each:

| position method | in-distribution | 4 distractors | 5 distractors |
|---|---:|---:|---:|
| learned absolute | 99.7% | 55.5% | 36.7% |
| RoPE | 89.3% | 38.2% | 28.9% |

Standard RoPE did not improve length extrapolation in this controlled task. It
was also less stable across seeds. This is a task-specific finding, not a claim
that RoPE is generally inferior.

## Depth scaling for longer paths

The following runs use 16 nodes, five total edges, two heads, MLP, learned
absolute positions, and a fixed 64-position capacity. Each depth has ten
seeds at 7,000 optimization steps:

| relation hops | 1 layer | 2 layers | 3 layers | 4 layers |
|---|---:|---:|---:|---:|
| 3 hops | 23.6% | 61.1% | 71.2% | 82.7% |
| 4 hops | 49.7% | 66.4% | 67.7% | 80.2% |

The 7,000-step results show that one layer is insufficient for three hops and
fails to solve four hops reliably. Beyond that, average accuracy generally
improves with depth but is not monotonic at every layer, and seed variance
remains large. Individual 2-4 layer runs can either solve the task or settle
near a partial solution.

Longer training separates these cases:

- A four-hop, 1-layer run remains at 50.1% after 20,000 steps.
- A four-hop, 2-layer run reaches 69.2% after 20,000 steps.
- A four-hop, 3-layer run reaches 99.85%.
- A four-hop, 4-layer run that was only 49.7% at 7,000 steps reaches 98.2% at
  20,000 steps.

Thus the fixed-budget ablations measure both representational bottlenecks and
optimization speed. They cannot be interpreted as a clean theorem that
`hops + 1` layers are necessary.

## Natural-language extension

Base Qwen3-0.6B:

- Held-out names, three distractors: 24.2% exact match over 128 examples.
- Eight distractors: 10.9% exact match over 128 examples.

LoRA, 200 steps, three seeds:

- All three seeds reach 99%+ exact match on held-out names.
- With eight distractors, accuracy ranges from 98.4% to 100%.
- Swapping the requested relation while keeping the facts fixed gives 99.2%
  to 100%.
- Two unseen prompt templates give 99.2% to 100%.
- Seed 0 over 512 examples: 99.6% in-distribution, 99.6% with eight
  distractors, 99.8% on unseen templates, and 99.8% after relation swapping.
- Seed 0 also reaches 99.4% with 16 distractor edges over 512 examples.
- Replacing both training relation words with unseen `supports/advises` still
  gives 99.6% over 512 held-out-name examples.

The natural-language result is encouraging but still on templated synthetic
language. It has not been tested on open-domain text, arbitrary relation
paraphrases, or multiple hops beyond two.

## Natural-language paired-retrieval transfer

The cleaner synthetic task was also translated into Qwen-style text:

- sixteen `reports to` facts with unique names;
- one question asks for two independently selected direct targets;
- facts are shuffled and names are held out from training;
- the answer must be exactly two comma-separated names in the requested order.

Base Qwen3-0.6B produces 0/32 exact two-name answers on this format. With LoRA
on 12,000 training examples (300 steps), six adapters were evaluated on 128
examples per condition using strict normalized exact match:

| condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| held-out names | 100% | 100% | 100% | 99.22% | 100% | 100% | 99.87% |
| unseen prompt templates | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 32 facts | 100% | 100% | 100% | 99.22% | 100% | 100% | 99.87% |
| order-swapped query set | 100% | 100% | 100% | 100% | 99.22% | 100% | 99.87% |
| unseen `supervises` wording | 100% | 99.22% | 100% | 100% | 100% | 100% | 99.87% |

The single-seed failures are one exact-match miss out of 128, not broad
breakdown. These results show that the two-answer behavior transfers to a
small instruction-tuned language model on templated synthetic language; they
do not yet establish performance on open-domain text or arbitrary relation
paraphrases.

#### Qwen attention-head ablation

For LoRA seed 0 on Qwen3-0.6B, we screened all 448 query heads (28 layers ×
16 heads, `head_dim=128`) by zeroing each head's slice of the input to
`o_proj` and measuring teacher-forced answer-token loss on 16 held-out-name
examples. The baseline loss was `1.04e-5` with 100% answer-token accuracy.
Three heads caused the largest degradation:

| layer | head | loss | delta loss | answer-token accuracy |
|---:|---:|---:|---:|---:|
| 13 | 6 | 2.117 | 2.117 | 76.7% |
| 2 | 6 | 1.242 | 1.242 | 74.4% |
| 0 | 3 | 0.747 | 0.747 | 86.7% |

Strict generation on 128 held-out-name examples confirms the effect:

- no ablation: 100%;
- layer 13 head 6: 19.53%;
- layer 2 head 6: 41.41%;
- layer 0 head 3: 35.94%;
- random control layer 0 head 0: 100%;
- all three heads ablated: 0%.

Pairwise and matched random controls show that the effect is specific:

- layer 13/head 6 + layer 2/head 6: 9.38%;
- layer 13/head 6 + layer 0/head 3: 0%;
- layer 2/head 6 + layer 0/head 3: 0%;
- three unrelated heads (0/0, 2/1, 13/9): 100%.

The dominant heads transfer across independently retrained adapters. On 64
examples:

| ablation | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---:|---:|---:|---:|
| layer 13/head 6 | 10.94% | 12.50% | 25.00% | 21.88% |
| layer 2/head 6 | 40.63% | 39.06% | 34.38% | 48.44% |
| layer 0/head 3 | 26.56% | 71.88% | 20.31% | 34.38% |
| all three | 0% | 0% | 0% | 0% |

Layer 13/head 6 and layer 2/head 6 are therefore stable, causally important
retrieval heads across five independently trained adapters. Layer 0/head 3 is
consistently harmful to ablate but has seed-dependent importance.

The same head-localized mechanism transfers to changed input distributions.
For seed 0 on 64 examples:

| condition | layer 13/head 6 | layer 2/head 6 | all three |
|---|---:|---:|---:|
| 32 facts | 20.31% | 45.31% | 0% |
| unseen templates | 23.44% | 45.31% | 0% |
| order-swapped queries | 28.13% | 43.75% | 0% |
| unseen `supervises` wording | 15.63% | 6.25% | 0% |

This is causal evidence that the adapter relies on a small set of
query-retrieval-sensitive heads rather than distributing the operation
uniformly. Seeds 1-4 were tested only at the candidates from seed 0, not over
all 448 heads, so undiscovered alternative heads may exist.

## Negative control: why a naive intersection task is insufficient

An attempted `parallel_intersection` task had two source nodes and one common
target. A 1-layer-1-head model without MLP solved it at 100%. The answer was
the only node appearing twice as an edge value, so the model could use a
frequency shortcut without retrieving both source-specific edges. This control
was stopped and is not evidence for multi-head necessity. Any follow-up must
remove duplicate-answer, token-count, position, and relation-balance shortcuts.

## Negative control: direct dual retrieval

Two new tasks were added to test whether independent attention heads can be
associated causally with two queries:

- `dual_retrieval`: two relation-typed key-value lookups;
- `parallel_copy`: two lookups over one relation type.

Both tasks have two output slots and are scored by joint exact match. With five
distractor records, 16 nodes, one seed, and 10,000 steps, neither one nor two
heads learned the task at any tested depth:

- 1-2 layers: each slot remained around 20%, joint accuracy around 3%;
- 3-4 layers: results were essentially unchanged for both one and two heads.

With only two records and eight nodes, both single- and two-head models learn a
partial solution: each answer slot is about 50% accurate and joint accuracy is
about 48%. The model usually solves at most one part of the paired retrieval,
but two heads do not reliably induce two independent lookups.

This is a negative result for the benchmark design, not evidence that two heads
are unnecessary. The task is currently entangled with key-value composition and
optimization difficulty. A cleaner head-necessity experiment needs either
single-token key-value records or auxiliary supervision, so that the only
remaining bottleneck is parallel retrieval.

## Matched-head single-token paired retrieval

The edge-token dual tasks required multiple steps just to associate a key with
the adjacent value. I therefore built a cleaner benchmark:

- each `(key, value)` record is a single token;
- the context contains 16 records drawn from 64 keys and 64 values;
- one final query contains two distinct keys;
- both retrieved values must be correct under joint exact match;
- keys and values are unique within an episode, records are shuffled, and
  neither frequency nor position identifies an answer;
- `d_model=96`, so independent two-head attention uses `head_dim=48`;
- the matched single-head control also uses `head_dim=48`.

Results over ten seeds:

| depth | architecture | mean joint accuracy | success >=95% | minimum |
|---|---|---:|---:|---:|
| 1 | 2 heads, independent | **100%** | **10/10** | **100%** |
| 1 | 1 head, full dim 96 | 78.3% | 8/10 | 1.62% |
| 1 | 1 head, `head_dim=48` | 77.3% | 8/10 | 1.26% |
| 1 | 2 heads, tied | 58.9% | 6/10 | 1.53% |
| 2 | 2 heads, independent | **100%** | **10/10** | **100%** |
| 2 | 1 head, `head_dim=48` | 50.1% | 4/10 | 5.59% |
| 2 | 2 heads, tied | 65.7% | 5/10 | 5.31% |

This is the first task that gives strong evidence of functional head
specialization rather than merely an optimization-scale effect.

For a successful one-layer two-head model over 512 examples:

- head 0 puts 97.7% attention mass on the record for answer 1 and 0.20% on the
  record for answer 2;
- head 1 puts 97.5% on the record for answer 2 and 0.23% on the record for
  answer 1;
- ablating head 0 drops answer-1 accuracy from 100% to 2.0% while answer 2
  remains 100%;
- ablating head 1 drops answer-2 accuracy from 100% to 0.39% while answer 1
  remains 100%;
- swapping the two queries produces the correctly swapped joint answer 100% of
  the time.

At the 16-record scale this supports a two-head reliability gap under matched
head dimension, width, task, and training budget. It is not a universal
theorem: matched single-head models solve some seeds, and tied heads
occasionally succeed. Later 32-record and curriculum experiments further
weaken the necessity interpretation.

### Harder 32-record setting

The same task was extended to 32 records, 128 keys, 128 values, and 10,000
steps. Five seeds per architecture:

| depth | architecture | mean joint accuracy | success >=95% | minimum |
|---|---|---:|---:|---:|
| 1 | 2 heads, independent | 80.2% | 4/5 | 0.76% |
| 1 | 1 head, full dim | 0.76% | 0/5 | 0.72% |
| 1 | 1 head, matched dim 48 | 0.49% | 0/5 | 0% |
| 1 | 2 heads, tied | 0.73% | 0/5 | 0.61% |
| 2 | 2 heads, independent | **100%** | **5/5** | **100%** |
| 2 | 1 head, full dim | 0.52% | 0/5 | 0% |
| 2 | 1 head, matched dim 48 | 1.05% | 0/5 | 0% |
| 2 | 2 heads, tied | 1.20% | 0/5 | 0% |

At 32 records, matched one-head and tied-head models consistently learn only
one of the two answers; their individual-slot accuracy can be near 100% while
joint exact-match is near chance. Two independent heads are the only tested
architecture that reaches 100% joint accuracy at two layers.

The two-layer 32-record checkpoint shows the same mechanism:

- head 0 places 99.6% attention mass on one queried record;
- head 1 places 99.8% on the other queried record;
- ablating head 0 leaves answer 1 at 100% but answer 2 at 0.39%;
- ablating head 1 leaves answer 2 at 96.1% but answer 1 at 0.98%;
- swapping the two queries preserves 100% joint correctness after swapping the
  expected answers.

#### Twenty-seed replication

Seeds 5-9 and 10-19 were added to the original five seeds. The independent
two-head model is reliable but not deterministic: 17 of 20 seeds solve the
task, while three settle near zero joint accuracy. This replaces both the
earlier five-seed `5/5` estimate and the noisier ten-seed `8/10` estimate with
a larger reliability estimate.

| depth | architecture | seeds | mean joint accuracy | success >=95% | minimum |
|---|---|---:|---:|---:|---:|
| 2 | 1 head, full dim 96 | 5 | 0.52% | 0/5 | 0% |
| 2 | 1 head, matched dim 48 | 20 | 0.62% | 0/20 | 0% |
| 2 | 2 heads, tied | 5 | 1.20% | 0/5 | 0% |
| 2 | 2 heads, independent | 20 | **85.15%** | **17/20** | **0%** |

The seed failures are not weak partial solutions: successful independent-head
runs are at 100%, while failures are near chance. This is a bimodal
optimization/algorithm-selection result.

Two failed seeds were retrained and saved for diagnosis:

- seed 6 learned only answer 1 (100% answer-1, 4.7% answer-2, 4.7% joint).
  Both first-layer heads placed roughly 99.6% attention mass on the answer-1
  record and almost none on the answer-2 record. The second answer was never
  assigned to a head.
- seed 8 learned neither answer. Both heads placed only 3-5% mass on either
  queried record, close to the 1/33 uniform baseline; activation patching had no
  effect.

This is the expected signature of a head-allocation failure, not a hidden
alternative one-head algorithm. Checkpoints are
`checkpoints/record_2l2h_r32_failedseed6.pt` and
`checkpoints/record_2l2h_r32_failedseed8.pt` on the shared filesystem.

#### Parameter-matched one-head control

The matched-head control held `head_dim=48` fixed but had fewer parameters
overall because it removed one head's attention parameters. We therefore
increased `d_model` while retaining one attention head and `head_dim=48`. At
two layers:

- independent 2H: `d_model=96`, 269,568 parameters;
- matched 1H: `d_model=105`, 269,274 parameters, only 0.11% fewer;
- both use 10,000 steps, 32 records, 128 keys, and 128 values.

The parameter-matched one-head model reaches **0% joint accuracy on 5/5
seeds** (mean 0%, minimum 0%). This rules out the simplest explanation that the
one-head result was caused by a smaller parameter budget.

#### Cross-query activation patching

`src/analyze_record_patching.py` reproduces the trained forward pass exactly
(argmax agreement 512/512) and patches activations from a run with the two
queries swapped.

Independent query-slot patching:

| patched query slot | answer 1 -> query 2 | answer 2 -> query 1 |
|---|---:|---:|
| slot 0 only | 96.7% | 0% |
| slot 1 only | 0% | 97.5% |
| both slots | 100% | 100% |

Patching one layer-0 head value with the corresponding value from the
swapped-query run gives a reciprocal effect:

- replacing head 0 changes only answer 1 to the other query (100%);
- replacing head 1 changes only answer 2 to the other query (100%);
- replacing either layer-1 head leaves both answers correct (100%).

Thus the causal two-query routing is already implemented by the first
attention layer. The second layer is useful for optimization/readout but its
heads are not individually query-selective in this checkpoint. Head ablation
and activation patching agree on the first-layer division of labor.

#### Position-encoding control

We also trained the same task without adding absolute position embeddings.
At 32 records and two layers, five seeds give:

| architecture | mean joint accuracy | success >=95% | minimum |
|---|---:|---:|---:|
| 1 head, matched dim 48 | 2.26% | 0/5 | 1.77% |
| 2 heads, independent | **80.55%** | **4/5** | **2.76%** |

The model still has learned `record_type` and `query_type` tags, so this tests
absolute positional encoding rather than all positional information. The
two-head separation survives without learned absolute positions.

#### Curriculum invalidates a task-level necessity claim

The fixed 32-record schedule makes matched one-head models fail, but changing
only the optimization schedule changes the conclusion. A linear curriculum
from 16 to 32 records over the first 4,000 of 10,000 steps gives the matched
one-head model (`d_model=96`, `head_dim=48`) **4/5 successes**, mean joint
accuracy **78.94%**, minimum 2.23%. This is essentially the same success rate
as the fixed-schedule two-head model, but with one softmax and fewer attention
parameters.

Therefore, the paired-retrieval task itself **does not force multiple heads**.
The earlier separation is an optimization-basin result for a particular fixed
schedule, not evidence that two attention distributions are necessary for the
task. The causal head-specialized mechanism remains a real learned solution,
but it is one of at least two possible solution families.

Under 16→64 curriculum training the one-head matched model also solves the
larger task on 4/5 seeds at about 100% accuracy. Multi-head models may still be
more reliable under some schedules, but the current evidence no longer supports
a general “two heads are necessary” claim. This is now recorded as a negative
result that motivates a stronger benchmark.

A curriculum-trained two-layer one-head checkpoint reveals the alternative
solution: the two lookups are serialized across depth rather than parallelized
across heads.

- layer 0 puts 94.0% mass on the answer-2 record and 0.18% on answer 1;
- layer 1 becomes responsible for answer 1;
- replacing layer-0 head output from a query-swapped run changes only answer 2
  (100% becomes the other query);
- replacing layer-1 head output changes answer 1 on 78.7% of examples;
- query-slot patching remains answer-specific.

Thus a two-layer one-head model implements two independent retrieval branches
sequentially. This is a much more informative result than a simple
one-head failure: parallel heads and serial depth are alternative resource
tradeoffs. A true “heads are necessary” benchmark must therefore control not
only parameter count, but also whether the model can time-multiplex the
queries across layers.

The minimal architecture confirms this more strongly. A one-layer one-head
model with the same 16→32 curriculum obtains:

- seed 2: 93.96% joint accuracy;
- seed 3: 95.02% joint accuracy;
- seeds 0, 1, and 4: single-answer plateaus around 0.7%;
- mean: 38.25%, success >=95%: 1/5, minimum: 0.71%.

Therefore the same task is sometimes solvable with one head in one layer.
It does **not** possess a task-level multi-head requirement. The fixed-schedule
32-record result is only a probabilistic optimization separation: larger
capacity/head count or an easier curriculum lowers the chance of falling into
a single-answer basin.

The successful one-layer one-head checkpoint makes the mechanism explicit:

- its single softmax puts 58.2% mass on the answer-1 record and 24.3% on the
  answer-2 record;
- both values are decoded correctly with 94.5% joint exact match;
- freezing the records and replacing both query slots with their counterparts
  flips both predicted answers (99.0% and 95.7% respectively);
- replacing the single head value with the corresponding value from the
  swapped-query run also flips both answers (99.0% and 95.5%).

Thus one normalized attention map can encode both selections through a
multi-peaked distribution. The two output classifiers read the resulting
superposition. No second attention distribution is logically required by this
task.

### 64-record boundary

At 64 records, 128 keys, 128 values, `d_model=96`, and 12,000 steps, the model
is below the capacity/optimization threshold for this benchmark:

- one-head full-dimension: 0% joint accuracy on 5/5 seeds;
- one-head matched `head_dim=48`: near 0% on 5/5 seeds;
- two-head independent: one seed reaches 100%, while the other four remain at
  0%, giving a 20% aggregate success rate.

The 64-record runs were stopped after this diagnostic because all configurations
were outside the useful operating regime. The appropriate next test is to scale
`d_model`, training budget, or use curriculum training before comparing head
counts at that length.

With a 16→64 curriculum over the first 8,000 of 20,000 steps, the boundary
changes substantially:

| architecture | mean joint accuracy | success >=95% | minimum |
|---|---:|---:|---:|
| 1 head, `head_dim=48` | **80.31%** | **4/5** | **1.67%** |
| 2 heads, independent | **99.98%** | **5/5** | **99.94%** |

The earlier 64-record failure was an optimization failure, not an
expressivity boundary. Independent heads are more reliable under this
curriculum, but the matched one-head model still finds the two-answer solution
on four of five seeds. This reinforces that head count modulates the
optimization basin rather than supplying an impossible-to-replace primitive.

### Three-query scaling and parallel-head routing

The record model was generalized to an arbitrary query count. The next
experiment uses 32 records, three queries, `d_model=132`, `head_dim=32`,
15,000 steps, and a 16→32 curriculum over the first 5,000 steps. Five seeds per
architecture:

| depth | heads | mean joint accuracy | success >=95% | minimum |
|---|---:|---:|---:|---:|
| 1 | 1 | 0.002% | 0/5 | 0% |
| 1 | 2 | **76.37%** | **3/5** | **0.71%** |
| 1 | 3 | **99.995%** | **5/5** | **99.98%** |
| 2 | 1 | 0.49% | 0/5 | 0% |
| 3 | 1 | 0.012% | 0/5 | 0% |

The one-layer 1H/2H/3H models have 235,044 / 252,036 / 269,028 parameters,
respectively. The two-head and three-head conditions differ by only 6.7%, so
the reliability gap is not driven by a large parameter-budget difference.

One diagnostic seed tests whether serial depth only needs more optimization:
three layers and one head with an 8→32 curriculum over 12,000 of 30,000 steps
and learning rate `5e-4` reaches only 4.55% joint accuracy. Its per-answer
accuracies are 83.67%, 42.07%, and 42.25%. This is one seed, so it is not a
success-rate estimate, but it shows that simply giving the serial solution
more steps and a gentler curriculum does not immediately recover all three
queries.

Increasing parallel heads substantially improves three-query joint retrieval.
The two-head model still has 4/5 runs at or above 94.37%, and the three-head
model is essentially perfect. The three-head checkpoint uses one head per
query:

- head 0 puts 98.8% mass on the answer-1 record;
- head 1 puts 98.7% mass on the answer-2 record;
- head 2 puts 98.5% mass on the answer-3 record;
- ablating a head destroys only its corresponding answer (0.6-0.8% for that
  answer while the other two remain at 100%).

The successful two-head checkpoint shows how one fewer head can still almost
solve the task:

- head 0 puts 62.7% mass on answer 2 and 27.8% on answer 1;
- head 1 puts 94.5% mass on answer 3;
- ablating head 0 destroys answers 1 and 2 while answer 3 remains at 100%;
- ablating head 1 destroys answer 3 while answers 1 and 2 remain at 94.9% and
  98.8%.

Thus a head can multiplex two queries through a multi-peaked distribution.
The three-head model is more reliable because each query gets its own selection
distribution, while the two-head model relies on value superposition in one
head.

This is a clear parallel-routing solution. It still does not prove that three
heads are architecturally necessary: two heads can partially solve three
queries, and a sufficiently different sequential computation might exist.
The result instead shows that head count controls how many independent
selection channels can be learned reliably in one layer.

### Four-query scaling

The same one-layer scaling test was extended to four queries at 32 records,
`d_model=180`, `head_dim=32`, 20,000 steps, and a 16→32 curriculum over the
first 6,000 steps. Five seeds per head count:

| heads | mean joint accuracy | success >=95% | minimum |
|---:|---:|---:|---:|
| 1 | 0% | 0/5 | 0% |
| 2 | 0.16% | 0/5 | 0% |
| 3 | **19.80%** | **1/5** | **0.73%** |
| 4 | **99.96%** | **5/5** | **99.93%** |

The one-layer one-to-four-head models have 410,504, 433,640, 456,776, and
479,912 parameters. The reliability transition is therefore much steeper than
the parameter-count change. Three heads often learn three of four answers, but
one answer remains missing; four heads consistently solve all four. This is
the strongest scaling evidence so far that extra heads supply additional
independent selection channels, while still not proving a universal
architectural lower bound.

## Reproduction commands

The main 32-record sweep:

```bash
python scripts/run_record_sweep.py \
  --results-dir results/record_headnail32_seeds10_19 \
  --gpu-uuids GPU_ID_0 GPU_ID_1 --workers 2 \
  --seeds 10 11 12 13 14 15 16 17 18 19 \
  --configs 1l2h_tied 2l1h_half 2l2h \
  --num-records 32 --num-keys 128 --num-values 128 \
  --d-model 96 --steps 10000
```

Parameter-matched one-head and no-position controls:

```bash
python scripts/run_record_sweep.py \
  --results-dir results/record_param_matched_1h_d105_5seed \
  --gpu-uuids GPU_ID_0 GPU_ID_1 --workers 2 \
  --seeds 0 1 2 3 4 --configs 2l1h_half \
  --num-records 32 --num-keys 128 --num-values 128 \
  --d-model 105 --steps 10000

python scripts/run_record_sweep.py \
  --results-dir results/record_headnail32_nopos_5seed \
  --gpu-uuids GPU_ID_0 GPU_ID_1 --workers 2 \
  --seeds 0 1 2 3 4 --configs 2l1h_half 2l2h \
  --num-records 32 --num-keys 128 --num-values 128 \
  --d-model 96 --steps 10000 --position-mode none
```

Curriculum and three-query sweeps:

```bash
python scripts/run_record_sweep.py \
  --results-dir results/record_headnail32_curriculum_1h_5seed \
  --gpu-uuids GPU_ID_0 GPU_ID_1 --workers 2 \
  --seeds 0 1 2 3 4 --configs 2l1h_half \
  --num-records 32 --num-keys 128 --num-values 128 \
  --d-model 96 --steps 10000 \
  --curriculum-start-records 16 --curriculum-steps 4000

python scripts/run_custom_record_sweep.py \
  --results-dir results/multiquery_q3_r32_1l3h \
  --gpu-uuids GPU_ID_0 GPU_ID_1 GPU_ID_2 --workers 3 \
  --seeds 0 1 2 3 4 --num-queries 3 --num-records 32 \
  --num-keys 128 --num-values 128 --d-model 132 \
  --layers 1 --heads 3 --head-dim 32 --steps 15000 \
  --curriculum-start-records 16 --curriculum-steps 5000

python scripts/run_custom_record_sweep.py \
  --results-dir results/multiquery_q4_r32_1l4h \
  --gpu-uuids GPU_ID_0 GPU_ID_1 GPU_ID_2 GPU_ID_3 --workers 4 \
  --seeds 0 1 2 3 4 --num-queries 4 --num-records 32 \
  --num-keys 128 --num-values 128 --d-model 180 \
  --layers 1 --heads 4 --head-dim 32 --steps 20000 \
  --curriculum-start-records 16 --curriculum-steps 6000
```

Mechanism analysis and plots:

```bash
python src/analyze_record.py CHECKPOINT --device cpu --examples 512
python src/analyze_record_patching.py CHECKPOINT --device cpu --examples 512
python scripts/plot_r32_head_controls.py
python scripts/plot_q3_scaling.py
```

Qwen per-head screening and causal generation checks:

```bash
python src/qwen_head_screen.py \
  --adapter outputs/lora_pair_seed0 \
  --examples 16 --batch-size 8 \
  --layer-start 0 --layer-end 14 \
  --result-json results/qwen_head_screen_layers0_13.json

python scripts/merge_qwen_head_screen.py \
  results/qwen_head_screen_layers0_13.json \
  results/qwen_head_screen_layers14_27.json \
  --output results/qwen_head_screen_all.json

python src/eval_lm.py \
  --adapter outputs/lora_pair_seed0 \
  --data data/nlp_pair_data/valid_id.jsonl \
  --examples 128 --max-new-tokens 24 \
  --ablate-heads 13:6 2:6 0:3
```

## Data locations

Tracked artifacts are under:

- `checkpoints/`: tiny synthetic-model checkpoints
- `results/`: JSON/CSV metrics and logs
- `data/nlp_data/`: generated natural-language data (created locally)
- `data/nlp_pair_data/`: generated paired-retrieval data (created locally)

LoRA adapters and the base Qwen checkpoint are not committed. Set
`--model` and `--output-dir` to local paths when running those experiments.
