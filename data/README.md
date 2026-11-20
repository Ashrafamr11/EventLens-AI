# LLP dataset card — EventLens AI

## Source and version

**Name:** Look, Listen and Parse (LLP), introduced with *Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing* (ECCV 2020).

**Official source:** https://github.com/YapengTian/AVVP-ECCV20

**Dataset snapshot/version:** The team worked from the ECCV-2020 LLP layout. An exact upstream commit hash or dataset checksum was **not recorded** in the supplied audit, so exact snapshot identity remains unverified. The original feature archive and metadata are retained in the team's external Google Drive.

**License/rights:** Do **not** infer that all YouTube-sourced videos, downloadable features, or annotations are freely redistributable. The upstream repository is authoritative for any licensing, permissions or usage restrictions; no explicit blanket media-redistribution license has been verified here. We publish scripts/links, not raw videos or external binary feature archives. Downstream users must check underlying footage and upstream code license independently before redistribution.

## Task and splits

- `AVVP_train.csv` — 10,000 weak clip-level training examples.
- `AVVP_val_pd.csv` — 649 validation examples, with temporal ground truth from separate annotations.
- `AVVP_test_pd.csv` — 1,200 final test examples; reserve for final evaluation.
- `AVVP_eval_audio.csv` and `AVVP_eval_visual.csv` — modality-specific, second-wise onset/offset label annotations.
- 25 event classes, 10-second clips, 10 temporal bins. Split IDs are checked against leakage in the preparation script.

## Feature extraction

The project uses original saved LLP feature conventions: VGGish audio `(10,128)`, pretrained ResNet152 appearance `(80,2048)` averaged to `(10,2048)`, and pretrained R(2+1)D-18 motion `(10,512)`.

Stored features use these paths relative to the project root:

```
data/processed/feats/vggish/<video-id>.npy
data/processed/feats/res152/<video-id>.npy
data/processed/feats/r2plus1d_18/<video-id>.npy
```

## Prepare the existing team dataset from external artifacts

The team Drive: https://drive.google.com/drive/folders/1R-5UfK9COEcOGmUn0L6_M5-UfsnDdAqO?usp=sharing

Mount or copy that external folder such that `$env:EVENTLENS_ARTIFACT_ROOT` resolves to its root. **Prepare after copying the original `data/feats.zip` and annotation manifests locally**. This is not an automated download from the original public dataset; it is a reproducible extraction of the team's previously saved asset snapshot.

```powershell
$env:EVENTLENS_ARTIFACT_ROOT = 'C:\EventLens-AI-Artifacts'
python scripts/prepare_training_assets.py --artifact-root "$env:EVENTLENS_ARTIFACT_ROOT" --project-root .
```

For evaluation on held-out test data, use `scripts/prepare_error_assets.py` (see `--help` / script arguments); do not move test observations into training.

## Metrics and caveats

Event F1 is a **custom pooled audio+visual union-of-annotations** protocol with 0.50 sigmoid threshold and temporal IoU at least 0.50. This is **not** equivalent to official AVVP modality-separated audio, visual and audio-visual scores. There were 1 invalid temporal annotation row in validation and 4 in test in the saved team's evaluation audit; check your snapshot matches rather than quietly relaxing the protocol.

## Credits

- Dataset and original parsing code: Tian et al., ECCV 2020 (official repository above).
- Pretrained encoders: VGGish (Google AudioSet research code), ResNet152, R(2+1)D-18 (Torchvision weights). See root README and technical report for citations.
