# Model checkpoints

This directory contains the small checkpoints used for the mechanism figures
and activation-patching analyses:

- `record_2l2h_r32.pt`: successful parallel two-head model;
- `record_2l1h_r32_curriculum_seed2.pt`: successful serial one-head model;
- `record_1l1h_r32_curriculum_seed3.pt`: successful one-head multi-peak model;
- failed two-head controls for comparison;
- checkpoints for the three-query experiments.

Each checkpoint is at most a few megabytes and can be loaded with the
`RecordTransformer` defined in `src/record_model.py`.

