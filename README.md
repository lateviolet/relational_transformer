# Parallel Heads or Serial Depth?

Controlled experiments on how depth, attention heads, and MLPs contribute to
multi-item retrieval in Transformers.

## Key finding

Two independent answers do not automatically require two attention heads.
Depending on training, the model can use:

- two heads in parallel;
- one head over two layers, processing the queries sequentially;
- one head in one layer, with two peaks in a single attention distribution.

More heads become increasingly important as the number of queries grows.
The same paired-retrieval task also transfers to Qwen3-0.6B, where a small set
of attention heads is causally responsible for the behavior.

![Attention routing](results/record_attention_examples.png)

## Benchmark

Each example contains shuffled single-token key-value records:

```text
k17 -> v42
k03 -> v09
k28 -> v71
...

query  = (k17, k03)
answer = (v42, v09)
```

The model must produce both values under joint exact match. Every episode uses
fresh random records, unique keys and values, randomized order, and no
frequency or position shortcut.

The main toy model contains:

- learned key and value embeddings;
- one token per record;
- a final query token with two key slots;
- one or two pre-LN Transformer blocks;
- two answer classifiers.

See [SUMMARY_EN.md](SUMMARY_EN.md) for an accessible overview and
[RESULTS.md](RESULTS.md) for the complete experiment log.

## Representative results

Two queries at 32 records:

| setting | two heads | one matched head |
|---|---:|---:|
| fixed schedule | 85.15% (17/20 seeds) | 0.62% (0/20 seeds) |
| curriculum | - | 78.94% (4/5 seeds) |

One-layer head scaling:

| queries | 1 head | 2 heads | 3 heads | 4 heads |
|---|---:|---:|---:|---:|
| 3 | 0/5 | 3/5 | 5/5 | - |
| 4 | 0/5 | 0/5 | 1/5 | 5/5 |

Qwen3-0.6B paired retrieval:

- base model: 0/32 strict exact-match examples;
- six LoRA adapters: 99.87% mean strict exact match over 128 examples;
- stable causal retrieval heads across five adapters: layer 13/head 6 and
  layer 2/head 6;
- ablating these heads and layer 0/head 3 reduces every tested adapter to 0%.

## Setup

The reference environment used Python 3.10, PyTorch 2.7, Transformers 5.15,
Datasets 5.0, PEFT 0.20, Accelerate 1.14, NumPy 2.2, and Matplotlib 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching the local CUDA driver first. For
example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Quick check

The smoke test validates task generation and model shapes for all synthetic
benchmarks and for two-, three-, and four-query retrieval:

```bash
python scripts/check_tasks.py
```

Train a small paired-retrieval model on CPU:

```bash
python src/train_record.py \
  --device cpu \
  --num-records 16 \
  --num-keys 64 \
  --num-values 64 \
  --layers 2 \
  --heads 2 \
  --head-dim 48 \
  --steps 500 \
  --checkpoint checkpoints/smoke.pt
```

## Reproduce the main experiments

Two queries at 32 records:

```bash
python scripts/run_record_sweep.py \
  --results-dir results/record_32_reproduction \
  --gpu-uuids 0 1 \
  --workers 2 \
  --seeds 0 1 2 3 4 \
  --configs 2l1h_half 2l2h \
  --num-records 32 \
  --num-keys 128 \
  --num-values 128 \
  --d-model 96 \
  --steps 10000
```

The same task with a 16-to-32 record curriculum:

```bash
python scripts/run_record_sweep.py \
  --results-dir results/record_32_curriculum \
  --gpu-uuids 0 \
  --seeds 0 1 2 3 4 \
  --configs 2l1h_half \
  --num-records 32 \
  --num-keys 128 \
  --num-values 128 \
  --d-model 96 \
  --steps 10000 \
  --curriculum-start-records 16 \
  --curriculum-steps 4000
```

For three or four queries, use
[`scripts/run_custom_record_sweep.py`](scripts/run_custom_record_sweep.py).
Exact commands are listed in [RESULTS.md](RESULTS.md).

## Inspect a checkpoint

```bash
python src/analyze_record.py checkpoints/record_2l2h_r32.pt \
  --device cpu --examples 512

python src/analyze_record_patching.py checkpoints/record_2l2h_r32.pt \
  --device cpu --examples 512
```

Generate the attention and scaling figures:

```bash
python scripts/plot_record_attention.py \
  --model "Parallel 2L-2H=checkpoints/record_2l2h_r32.pt" \
  --model "Serial 2L-1H=checkpoints/record_2l1h_r32_curriculum_seed2.pt" \
  --model "One head 1L-1H=checkpoints/record_1l1h_r32_curriculum_seed3.pt" \
  --output results/record_attention_heatmaps.png

python scripts/plot_r32_head_controls.py
python scripts/plot_q3_scaling.py
python scripts/plot_q4_scaling.py
```

## Natural-language experiment

Generate the paired-retrieval language data:

```bash
python scripts/export_pair_data.py --output-dir data/nlp_pair_data
```

Train and evaluate a Qwen LoRA adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_lora.py \
  --model Qwen/Qwen3-0.6B-Base \
  --data-dir data/nlp_pair_data \
  --max-steps 300 \
  --output-dir outputs/lora_pair_seed0

python src/eval_lm.py \
  --model Qwen/Qwen3-0.6B-Base \
  --adapter outputs/lora_pair_seed0 \
  --data data/nlp_pair_data/test_r32.jsonl \
  --examples 128 \
  --match-mode exact
```

Per-head screening and causal ablation:

```bash
python src/qwen_head_screen.py \
  --model Qwen/Qwen3-0.6B-Base \
  --adapter outputs/lora_pair_seed0 \
  --examples 16 \
  --result-json results/qwen_head_screen.json

python src/eval_lm.py \
  --model Qwen/Qwen3-0.6B-Base \
  --adapter outputs/lora_pair_seed0 \
  --data data/nlp_pair_data/valid_id.jsonl \
  --examples 128 \
  --ablate-heads 13:6 2:6 0:3
```

## Repository layout

```text
src/          model, data, training, evaluation, and analysis code
scripts/      sweep, plotting, and data-generation utilities
results/      tracked JSON metrics, logs, and figures
checkpoints/ small checkpoints used for mechanism analysis
data/         generated datasets (created locally)
```

Additional documents:

- [SUMMARY_EN.md](SUMMARY_EN.md): concise project overview;
- [RESULTS.md](RESULTS.md): full experimental log, negative results, and commands;
- [PROPOSAL.md](PROPOSAL.md): research questions and next steps;
- [THEORY.md](THEORY.md): restricted-model lower-bound argument;
- [REPORT_CN.md](REPORT_CN.md): detailed Chinese report;
- [PAPER_NOTES_CN.md](PAPER_NOTES_CN.md): notes on related papers.

## Scope and limitations

The synthetic experiments show an optimization and reliability separation under
specific depth, width, head-dimension, curriculum, and training budgets. They do
not prove a universal lower bound for all one-head Transformers.

The Qwen experiments use synthetic templated prompts. They demonstrate causal
head localization, not open-domain multi-hop reasoning.

