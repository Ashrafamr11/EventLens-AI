# Experiment boundaries and completion criteria

This is a research-reproducibility status record, not a claim that missing experiments were run.

## Hyperparameter budget

The final four architectures used identical **final-training settings** (LR 0.001, five epochs, batch 64, AdamW), but the *search budget* was not equal: an additional BiGRU learning-rate comparison 0.001 versus 0.0003 was recorded for one seed. Therefore, final-setting parity is established, **equal tuning effort is not**. To resolve: run `configs/fair_lr_sweep_plan.json` on each architecture with identical seeds, using the same validation criterion, and retain individual logs. Do not report the proposed results until executed.

## Pretrained-versus-scratch

The frozen VGGish/ResNet152/R(2+1)D-18 feature pipeline relies on pretrained weights. Training the temporal heads from random initialization is **not** a scratch feature-extractor comparison. A valid study would hold the architecture, data, optimizer budget, splits and feature shape fixed, replace upstream feature-extractor pretrained weights with random initialization **or train the same extractor from scratch**, and retrain each corresponding localization head under a controlled protocol. It requires raw videos/audio, compute and suitable licensing. A random feature shuffle or mismatched baseline does not establish a fair pretrained-vs-scratch result. **Status: not executed**.

## Cost / memory

Original audited model-head parameter counts and checkpoint sizes are available in `experiments/analysis/computational_cost.csv`. Three-seed training time was recorded on a Tesla T4. The inference latency is **CPU head-only**, not video decoding and feature extraction. There was one ~99.8-second CPU real-video end-to-end smoke test. GPU training peak allocated/reserved memory was not saved in the original four-model logs. New `src/train.py` runs print and save CUDA peak allocated/reserved MiB for **new runs**; never retrofit those measurements onto older runs or silently treat a one-batch stress test as an original full-training benchmark.

## Cross-modal architecture

Recorded validation event micro-F1 was ~0.2511 for cross-modal attention, lower than baseline ~0.3032. `python scripts/audit_cross_modal.py` checks outputs and gradient propagation on synthetic tensors. A passed structural audit does not establish the absence of data, optimization or evaluation bugs. For stronger diagnosis inspect training curves, modality gate distributions and shared evaluator on saved checkpoints across all seeds.

## Reproduction source and provenance

Original notebook is retained in team Drive. It is not republished without secret- and output-clearing review. Small JSON run metadata are in `experiments/`. Added command-line scripts in `src/train.py` and `src/evaluate.py` can produce *new* run artifacts; they are not represented as having produced the original recorded results.
