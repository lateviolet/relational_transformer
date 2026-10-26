# Research Direction

## Working title

Parallel Heads or Serial Depth? Resource Allocation in Transformer Relational
Learning

## Motivation

The random-walk analysis by Shi and Cao provides a controlled example in which
attention selects the immediate predecessor and the value matrix implements a
one-step transition. The natural next question is which architectural resources
are required when context selection, relation choice, and multi-step
composition are separated into different operations.

## Questions

1. Which compositional tasks produce a clean separation between one- and
   multi-layer transformers?
2. When do Transformers implement independent lookups in parallel across
   heads, and when do they serialize them across successive layers?
3. What does the MLP contribute beyond attention: relation matching, binding,
   or a more stable readout?
4. Which of these mechanisms transfer to lexicalized natural-language prompts
   with held-out names and paraphrases?
5. Under which explicitly restricted model/readout families can head necessity
   be proved rather than only inferred from optimization outcomes?

## Task family

- One-hop relation lookup.
- Two-hop pointer chasing with shuffled facts.
- Rule-cued two-hop lookup where multiple relation chains coexist.
- Length-shifted evaluation with more distractors than training.
- Lexicalized two-chain question answering with held-out names and templates.

Every episode uses a fresh random graph, so memorizing individual examples
cannot produce high held-out accuracy.

## Architecture controls

- depth: 1 vs 2
- heads: 1 vs 2
- MLP: on vs off
- equal model width and optimization budget
- multiple seeds and repeated runs
- attention-head ablation and linear state probes

## Current evidence

The fixed-length cued task shows a strong optimization separation over ten
seeds: 1L-2H-MLP stays at 49.9%, while 2L-2H-MLP reaches 99.7%. 2L-1H-MLP and
2L-2H-noMLP can solve individual runs but have much higher seed variance.

A tied-head control also reaches 99.8% over five seeds. This is important:
the observed benefit of two heads cannot yet be attributed to head diversity.
The task therefore supports a reliability claim, not strict multi-head
necessity.

A one-head model with `head_dim=32` reaches 99.47% over ten seeds
(minimum 98.999%), matching the two-head model. This identifies head dimension
and parameter partitioning, rather than head count, as the immediate cause of
the observed optimization difference. Any future necessity experiment must
match effective head dimension across the one-head and two-head conditions.

The matched-head single-token paired-retrieval task now supplies the first
clear multi-head separation. With 16 records at `d_model=96` and
`head_dim=48`, independent two-head models reach 100% joint accuracy on
10/10 one-layer and 10/10 two-layer seeds. The matched one-head models reach
77% mean joint accuracy (8/10 successful), while tied heads reach 59%
(6/10) at one layer and 66% (5/10) at two layers. Attention and ablation
analysis show that one head selects each queried record and removing either
head destroys the corresponding answer. A successful one-layer model trained
on 16 records retains 98.4% joint accuracy on 32 records.

At 32 records, the separation is sharper. Across twenty seeds, the two-layer
independent two-head model succeeds on 17/20 seeds with mean joint accuracy
85.15% (minimum 0%), while the matched one-head control fails on all 20 seeds
at 0.62% mean accuracy. The successful runs are at 100% and the failed runs are
near zero, so the headline is a bimodal optimization/algorithm-selection
result rather than deterministic learnability. Failed models usually answer
one query correctly and ignore the other. A 32-record checkpoint shows
reciprocal head ablation, and cross-query activation patching traces both
independent lookups to the first attention layer.

A parameter-matched one-head control (`d_model=105`, `head_dim=48`, 269,274
parameters) is within 0.11% of the independent two-head model's parameter count
(`d_model=96`, `head_dim=48`, 269,568 parameters). It still has 0% joint
accuracy on 5/5 seeds. Removing learned absolute position embeddings also
preserves the separation: two heads reach 4/5 successes at 80.55% mean, while
matched one head has 0/5 successes at 2.26% mean.

This fixed-schedule separation does not survive a change to curriculum
training. A 16→32 curriculum over the first 4,000 of 10,000 steps lets the
matched one-head model succeed on 4/5 seeds with 78.94% mean accuracy. A
16→64 curriculum similarly lets the matched one-head model reach 80.31% mean
accuracy and 4/5 successes, versus 99.98% and 5/5 for independent heads.
Therefore, the paired-retrieval task itself does not require two attention
distributions. Independent heads are one reliable learned solution, not a
representational necessity.

The current conclusion is narrower: depth, head dimension, MLP width, and the
optimization curriculum jointly determine whether gradient descent finds a
one-head, tied-head, or independent-head solution. Claims of multi-head
necessity require a task/interface where a single normalized attention
distribution is provably insufficient, not merely harder to learn.

A successful curriculum-trained one-head checkpoint explains the tradeoff:
layer 0 attends primarily to the answer-2 record and layer 1 takes over the
answer-1 lookup. Cross-query activation patching confirms this serial division.
The model performs the two branches sequentially across depth instead of in
parallel across heads. This suggests a broader research question that is more
tractable than strict head necessity: when does a Transformer choose
parallelism across heads versus serial computation across layers under a fixed
parameter and FLOP budget?

The minimal control is even more decisive: with the 16→32 curriculum, a
one-layer one-head model reaches 93.96% and 95.02% joint accuracy on two of
five seeds, while three seeds remain at a one-answer plateau. The current task
therefore does not establish either head or depth necessity; it exposes a
multi-basin optimization landscape.

Scaling to three queries gives a clearer positive role for heads. In a
one-layer model at 32 records, one head reaches 0/5 successes, two heads reach
3/5 at 95% (mean 76.37%), and three heads reach 5/5 with 99.995% mean joint
accuracy. The three-head checkpoint maps head 0/1/2 to queries 1/2/3 with
98.5-98.8% attention mass, and ablation is answer-specific. However, two heads
solve three queries on multiple seeds, so this is a reliability/scaling effect,
not a strict one-head impossibility result.

At four queries, the reliability curve becomes sharper. One/two/three/four
heads produce 0/5, 0/5, 1/5, and 5/5 successes, with mean joint accuracy
0%, 0.16%, 19.80%, and 99.96%. The number of parameters changes only from
410,504 to 479,912, so the transition is not explained by a large parameter
increase alone.

Natural-language Qwen3-0.6B experiments transfer the paired-retrieval task
with held-out names, unseen prompt templates, 32-fact contexts, swapped query
order, and unseen `supervises` relation wording. Base Qwen3-0.6B answers 0/32
paired questions, while six LoRA adapters average 99.87% strict exact match
across 128 examples per condition (minimum 99.22%). This is promising as a
controlled transfer result but is not yet evidence about open-domain language.

A causal head screen on adapter seed 0 identifies layer 13/head 6, layer
2/head 6, and layer 0/head 3 as the dominant retrieval heads. On 128 examples,
their individual ablation reduces exact match from 100% to 19.53%, 41.41%, and
35.94%; an unrelated head leaves accuracy at 100%; ablating all three reaches
0%. Two of the three pairwise ablations also reach 0-9.38%, while a matched
random three-head control remains at 100%. On independently trained adapters
seed 1-4, the two dominant heads again reduce accuracy to 10.94-25.00% and
34.38-48.44%, and the three-head set reduces every seed to 0%.
Layer 13/head 6 and layer 2/head 6 are therefore stable retrieval heads in
these adapters. Seeds 1-2 were tested only at the seed-0 candidates, not fully
screened. Across 32 facts, unseen templates, swapped query order, and unseen
`supervises` wording, those two heads reduce accuracy to 15.63-28.13% and
6.25-45.31%; ablating the three-head set reaches 0% in every condition.

## Next experiments

1. Inspect the successful curriculum-trained one-head checkpoints to determine
   whether one softmax learns a two-peak superposition or exploits a second
   layer/MLP.
2. Design a task whose output requirements explicitly demand two independently
   normalized state distributions. A generic two-label readout can always be
   solved by a single head in principle.
3. Formalize a lower bound only for the restricted readout family where it is
   actually true; do not treat the current empirical runs as such a theorem.
4. Test more semantic shifts in the Qwen paired-retrieval transfer, including
   distractors with the same first names or similar relation paraphrases.

## Risks

- A task can be solved by token-frequency or position shortcuts even when its
  intended algorithm looks compositional.
- A failed run at a fixed optimization budget is not an expressivity lower
  bound. Report success rate as well as final accuracy.
- LoRA accuracy on templated language does not imply broad language-model
  behavior; test format, lexical, and relation shifts separately.
