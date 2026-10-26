# Result artifacts

This directory contains the JSON metrics and figures produced by the
experiments in the paper-style summary. Raw run logs are intentionally ignored
because they contain machine-specific paths and produce little additional
information beyond the structured JSON records.

The main figures are:

- `record_attention_examples.png`: single-example attention routing;
- `record_attention_heatmaps.png`: average attention routing;
- `r32_head_controls.png`: architecture and curriculum controls;
- `q3_scaling.png`, `q4_scaling.png`: multi-query head scaling;
- `qwen_head_screen_seed0.png`: causal Qwen head-ablation heatmap.

Result JSON files store the command-line parameters, final accuracy, training
history, and GPU model for each run. Machine-specific paths in older records
are normalized to project-relative locations where possible.
