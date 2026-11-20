# Controlled data-strategy study — recorded results

This directory contains the **actual saved** three-seed results for a train-only,
scale-adaptive Gaussian feature noise study. Values are not simulations.

- Model: BiGRU; seeds `42`, `43`, `44`; five epochs each.
- Train: 10,000 LLP clips; validation: 649 clips; no test-set selection.
- Training only: independently add Gaussian noise with standard deviation
  `0.05 * max(np.std(feature_matrix), 1e-6)` to each of the aligned audio,
  RGB and motion feature matrices. Validation inputs remain unchanged.
- Original control: archived clean-feature BiGRU runs with the same three seeds.
- Evaluation: custom `audio_visual_union` annotation protocol, threshold 0.5,
  event IoU >= 0.5; **not the official AVVP scoreboard**.
- Results: clip F1 0.7669 → 0.7642; segment F1 0.6531 → 0.6559;
  event F1 0.4442 → 0.4410 (event delta **−0.32 percentage points**).

The original BiGRU remains the final selected model. Changes this small over
three seeds do not establish a statistically significant effect.

`bigru_noise05_seed42.json` was **recovered from its checkpoint after a
CPU-only metadata logging failure**. The checkpoint existed and the five
epochs finished, but precise training duration and epoch history were not
captured and are therefore `null`/empty. Seeds 43 and 44 have recorded CPU
training durations; do not average these with missing seed42 as a full
three-run training-time estimate.

The augmented model checkpoints are intentionally **not committed** to Git:
they remain in the team's Drive artifact folder at
`experiments/data_strategy/bigru_noise05_seed{42,43,44}.pt`.

Revalidate and regenerate compact CSVs without training:

```bash
python scripts/verify_data_strategy_results.py
```

Prerequisite for rerunning the training code: LLP data, pre-extracted
features, and the original saved BiGRU seed44 checkpoint in the artifact root.
