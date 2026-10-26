# Restricted-Model Lower Bound

The experiments show an empirical separation, but not a universal impossibility
result for arbitrary one-head Transformers. This note isolates a restricted
model where a clean lower bound can be proved.

## Two independently normalized selections

Split positions into two disjoint groups A and B. Let a one-head softmax produce
attention weights `a_i`. Define

```text
m_A = sum_{i in A} a_i
m_B = 1 - m_A.
```

Assume values are encoded by group indicators and the readout is linear. Any
single-head output is therefore

```text
y = c0 + c1 m_A + c2 m_B
  = (c0 + c2) + (c1 - c2) m_A.
```

Even if `y` has two output coordinates, all attainable targets lie on a
one-dimensional affine manifold parameterized by `m_A`.

Now consider a task whose target contains two independently specified group
masses `u` and `v`, with `u,v` ranging over an open two-dimensional set. No
single affine readout of `m_A` can be injective on that set, so there must be
distinct inputs with different targets but identical available statistics.

Two heads can produce separate normalizations `a^(1)` and `a^(2)`, yielding
independent masses

```text
u = sum_{i in A} a_i^(1)
v = sum_{i in B} a_i^(2).
```

The two head outputs therefore provide two degrees of freedom, enough for a
linear readout to recover the target pair.

### Proposition

Let the value map be linear and let context values identify only membership in
disjoint groups A and B. In a one-head model, write its scalar attention mass
on A as `m in (0,1)`. If the two outputs are linear functions of the attended
vector, then

```text
(y_1, y_2) = (alpha_1 + beta_1 m, alpha_2 + beta_2 m).
```

Its Jacobian with respect to `(u,v)` has rank at most one. Therefore no
one-head model in this family can realize every target pair `(u,v)` on an open
subset of `(0,1)^2`.

For two heads, let `m_1` and `m_2` be their independently normalized masses.
The readout can be

```text
(y_1, y_2) = (alpha_1 + beta_1 m_1, alpha_2 + beta_2 m_2),
```

whose Jacobian has rank two when both slopes are nonzero. This establishes
head necessity only for the stated restricted family. The restriction is
important: a learned nonlinear map or a richer value encoding can remove the
rank obstruction.

## Explicit restricted benchmark

A valid necessity benchmark therefore needs an output interface like:

```text
context: four records in two disjoint groups A={a0,a1}, B={b0,b1}
query:   two independent scalar coefficients u and v
target:  distribution 1 = [u, 1-u] over A
         distribution 2 = [v, 1-v] over B
```

In the restricted model, every attention head exposes one normalized
distribution over the four records, and the two outputs are linear readouts
without an intervening MLP. One head gives only one shared normalization and
cannot realize the open two-dimensional target set. Two heads give the two
independent normalized masses and can realize it.

This benchmark would prove a theorem about a **head-exposed linear-readout
Transformer**. It would not imply that a normal decoder-only Transformer with
an MLP requires two heads, because the MLP can rescale and rearrange the shared
attention mixture.

## Scope

This lemma proves a lower bound for the restricted linear-readout/group-mass
family. A single head with an unrestricted MLP may renormalize or transform a
mixed vector and solve some versions of paired retrieval, as observed on the
16-record task. Consequently:

- the 16-record experiments establish optimization reliability;
- the 32-record exact-match experiments establish strong algorithmic evidence
  at matched width, head dimension, data, and training budget;
- a universal theorem would require adding a readout, width, or optimization
  restriction beyond the current unrestricted Transformer.

The next theoretical target is to make the two-slot output interface explicit
and prove that one normalized attention map has insufficient degrees of
freedom for all independent query pairs.

## Why the 32-record result is not a universal theorem

A generic one-head Transformer is not covered by the group-mass lower bound.
A single attention distribution can in principle put non-negligible mass on
both queried records. If their value embeddings occupy different coordinates,
the attended vector contains both values, and an MLP or two output projections
could in principle decode both coordinates. This is why a failed optimization
run is not a proof of inexpressivity.

The 32-record experiment instead supports a narrower claim:

> Under matched depth, data, optimization budget, and effective head dimension,
> all tested one-head and tied-head models fell into a one-answer solution,
> whereas independent heads reliably found and retained the two-answer solution.

This is an algorithmic and optimization separation for this parameterization,
not the statement that no one-head Transformer of any width or depth can solve
paired retrieval.

## Evidence ladder for a head-necessity claim

A convincing empirical claim needs all of the following controls:

1. **Valid task.** Unique keys and values, shuffled records, two non-identical
   queried keys, joint exact-match scoring, and no positional or frequency cue.
2. **Matched capacities.** Compare one head and two heads at the same depth,
   effective head dimension, and approximately the same parameter count. Also
   include a full-width one-head control and a two-head tied control.
3. **Seed reliability.** Report success rate, mean, and minimum over at least
   five seeds; one favorable run is insufficient.
4. **Counterfactual behavior.** Swap the two queries and require the answers to
   swap correspondingly. This rules out fixed output-slot labels.
5. **Causal interventions.** Ablate each head, then patch individual query
   slots and per-head values from a swapped-query run. The expected causal
   signature is one-head/one-answer rather than merely different attention maps.
6. **Difficulty scaling.** Show that the separation persists as the number of
   records grows, while scaling width and optimization budget enough to avoid
   measuring only failure to optimize.
7. **Negative controls.** A tied-head model must fail, and any single-head
   matched model that succeeds must be reported as evidence against necessity.

The parameter-matched one-head control can be built by increasing width while
holding `head_dim=48`. At two layers and 32 records, the independent two-head
model has `d_model=96` and 269,568 parameters. A one-head model with
`d_model=105` and `head_dim=48` has 269,274 parameters, a difference of 0.11%.
This is a stronger budget match than the `d_model=96`, `head_dim=48` control.

## The current task is not a necessity benchmark

Changing only the training curriculum overturns the fixed-schedule separation.
With the same two-layer, `d_model=96`, `head_dim=48` matched one-head model, a
linear 16→32 record curriculum over the first 4,000 of 10,000 steps succeeds on
4/5 seeds (78.94% mean accuracy). A 16→64 curriculum over the first 8,000 of
20,000 steps lets the one-head model reach about 100% on 4/5 seeds.

Thus the paired-retrieval interface admits at least one one-head solution. The
earlier 0/20 one-head failures without curriculum measure an optimization
basin, not expressivity. This is exactly the negative control that a rigorous
necessity experiment needs to include, and it prevents us from claiming
multi-head necessity for the task.

A successful two-layer one-head checkpoint reveals how the two queries are
processed without parallel heads. Layer 0 attends primarily to the second
queried record; layer 1 specializes on the first query. Cross-query activation
patching changes only answer 2 when layer-0 head output is replaced and changes
answer 1 when layer-1 head output is replaced. The model therefore serializes
the two branches across depth.

This exposes the key resource tradeoff:

```text
2 independent heads in one layer  ~ parallel query processing
1 head over 2 layers              ~ serial query processing
```

The relevant architectural question is not “does this task prove heads are
necessary?” but “under a fixed parameter/FLOP budget, when does optimization
prefer parallel head computation versus serial depth?” That motivates
head-count versus depth phase diagrams, rather than a universal lower bound.

The minimal one-layer one-head curriculum run rules out even a two-stage
necessity claim. It reaches 93.96% and 95.02% joint accuracy on two of five
seeds. Thus the current two-query task can be solved by one attention
operation, one head, and one layer. Its multiple configurations differ in the
probability and stability of finding the solution, not in whether a solution
exists in the architecture family.

With three queries, parallel head count produces a clearer reliability curve:
one/two/three heads give 0/5, 3/5, and 5/5 successes in one layer. The
three-head checkpoint routes one head to each query, but the two-head model can
also solve three queries on several seeds by sharing a multi-peaked attention
map. This again supports head count as a reliability/capacity dimension, not a
strict combinatorial lower bound.

Four-query scaling sharpens that curve: one/two/three/four heads give
0/5, 0/5, 1/5, and 5/5 successes. The model parameter count increases by only
16.9% from one to four heads, while the success probability jumps from 0 to 1
across five seeds. This is consistent with independent attention
distributions acting as parallel selection channels, but remains empirical.

To obtain an architectural necessity result, the output must expose the two
independently normalized distributions themselves, or the model family must
restrict the readout so that one attention map provably cannot encode both.
Merely asking two classification questions does not do that: one softmax can
attend to two records, and a second layer or MLP can separate their values.
