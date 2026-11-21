# EventLens AI — Audio–Visual Event Localization

**Deep Learning AI403 · Fall 2026 · Project idea #14**  
**Team W — Expired**
Pharos University in Alexandria, Faculty of Computer Science & Artificial Intelligence.

## Registered team

| Name | ID |
|---|---|
| Ashraf Amr Ebrahim Abdelfattah Elsheikh | 202300962 |
| Ezzeldin Yasser Nabil Shamaa | 202303039 |
| Kareem Abdelnaser Mohamed Alazazy | 202303078 |
| Mohamed Ehab Abdelnaby Elsayed Fouda | 202303042 |
| Youssef Ali Abdallah Shalaby Elfar | 202302115 |
| mahmoud Gomaa Elsayed Eid Mohamed | 202303293 |

> **Team approval:** All six students appear as accepted Team W entries in the course team sheet. The team confirms that the TA approved its six-person roster.

## Project summary

EventLens accepts a video of approximately 10 seconds with audio and predicts class names and time spans. It uses **FastAPI + custom HTML/CSS/JS**, local PyTorch heads, and pretrained VGGish/ResNet152/R(2+1)D-18 feature extraction. No external AI prediction API is used. Input feature shape is audio (10,128), RGB (10,2048), motion (10,512). The vocabulary contains 25 LLP event classes.

### Selected model and measured results

Selected checkpoint: **BiGRU seed44**, chosen on validation; held-out LLP test event micro-F1 **0.4367**. This is a **custom audio+visual union event metric**, not the original modality-separated AVVP leaderboard score. Temporal event matching uses IoU >= 0.50 and sigmoid threshold 0.50.

| Model | Validation clip F1 (mean ± SD) | Segment F1 | Event F1 |
|---|---:|---:|---:|
| Temporal MLP | 0.7033 ± 0.0253 | 0.5097 ± 0.0274 | 0.3032 ± 0.0210 |
| Bidirectional GRU | 0.7669 ± 0.0056 | 0.6531 ± 0.0144 | 0.4442 ± 0.0151 |
| Transformer | 0.7632 ± 0.0259 | 0.6502 ± 0.0409 | 0.4108 ± 0.0391 |
| Cross-modal attention | 0.5613 ± 0.0298 | 0.3874 ± 0.0305 | 0.2511 ± 0.0270 |


Shared training protocol: LLP 10,000 train / 649 validation / 1,200 test; 3 seeds 42/43/44, 5 epochs, AdamW, LR 0.001, weight decay 0.0001, batch 64. The main runs were logged on Tesla T4. No-audio BiGRU ablation decreased event F1 by 7.60 percentage points. A three-seed 5% train-only Gaussian feature-noise study changed event F1 −0.32 points, so the clean BiGRU remains selected.

Held-out test event error audit: TP=1,040, FP=1,085, FN=1,598, plus 485 boundary mismatches (these are not additive). All 25 per-class scores and concrete dataset-evaluator mismatch examples are available in `experiments/error_analysis/`.

### Runtime and size (localization heads only)

| Model | Train mean T4 minutes (3 seeds) | CPU head ms (10 repetitions) | Checkpoint MiB |
|---|---:|---:|---:|
| Temporal MLP | 2.72 | 1.79 | 7.96 |
| Bidirectional GRU | 2.71 | 6.07 | 11.37 |
| Transformer | 2.73 | 3.30 | 5.53 |
| Cross-modal attention | 2.74 | 9.80 | 7.08 |


CPU head latency does **not** include VGGish/ResNet/motion extraction. One full real-video CPU request took approximately 99.8 seconds, a single smoke-test observation. Peak memory was not recorded.

## Reproducibility artifacts

**Shared models/feature extractor weights/original experiments:** https://drive.google.com/drive/folders/1R-5UfK9COEcOGmUn0L6_M5-UfsnDdAqO?usp=sharing

Folders include `experiments/`, `weights/`, `tools/`, `checkpoints/`, `data/`. The share link points to external artifacts; GitHub intentionally excludes large weights, media and raw data. Folder permissions depend on the Drive sharing settings.

### Environment

- Python 3.10/3.11 recommended (some TensorFlow/VGGish environments may need specific compatible versions).
- FFmpeg and FFprobe available on PATH: https://ffmpeg.org/download.html
- Install core package and dev dependencies:

```powershell
git clone https://github.com/Ashrafamr11/EventLens-AI.git
cd EventLens-AI
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m pytest -q
```

For real-video extraction, additionally install `requirements-video.txt`; the feature-only NPZ route does not need the original TensorFlow VGGish backend.

### Launch web demo

```powershell
$env:EVENTLENS_ARTIFACT_ROOT = 'C:\EventLens-AI-Artifacts'  # replace with local copied Drive folder
$env:EVENTLENS_VGGISH_ARCHIVE = 'C:\EventLens-AI-Artifacts\tools\feature_extractor_original.zip'
python -m uvicorn web.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Check `/api/status` before uploading a new video. Use video upload for full extraction or `.npz` features with keys `audio`, `visual`, `motion` for feature-only model inference. The input video should be 9.5–10.5 seconds and contain audio.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | service status |
| GET | `/api/status` | artifact and runtime health |
| GET | `/api/models` | model registry / checkpoint availability |
| POST | `/api/analyze-video` | upload original video and run full pipeline |
| POST | `/api/analyze-features` | upload `.npz` and infer |
| POST | `/api/compare-features` | compare multiple model heads on features |

### Experiments and tests

- `scripts/build_research_tables.py`: generate research tables from experiment audit.
- `scripts/evaluate_event_errors.py`: held-out error audit.
- `scripts/prepare_error_assets.py`: restore test features from external archive.
- `scripts/prepare_training_assets.py`: restore training features from external archive.
- `scripts/run_data_strategy.py`: train-only Gaussian feature-noise study.
- `scripts/verify_data_strategy_results.py`: assert consistency with recorded data-strategy outcomes.
- `experiments/analysis/`: tables, model comparison, per-class F1 and computation audit.
- `experiments/error_analysis/`: aggregate and logged event errors; these are algorithmic differences from pooled labels, not human-reviewed mistakes.
- `experiments/data_strategy/`: three seeds plus the comparison. Seed 42 was recovered from a valid checkpoint after a logging-only CPU device error; its original runtime is unknown.
- `src/eventlens/experiment.py` and `src/eventlens/training.py`: training loop functions.

Observed tests on Windows: **33 passed**, 4 nonblocking dependency warnings (20 November 2026). A real 10-second video was processed successfully in Colab with all three pretrained extractors; API and browser timeline returned results. Predictions Dog and Speech on that single video were unverified smoke-test examples, not hand-checked labels.


## Dataset, training and evaluation (reproduction commands)

The LLP dataset documentation, source/version caveats, usage-rights considerations, split protocol and preparation instructions are in [`data/README.md`](data/README.md). The original media/feature binaries are **not** committed.

After mounting or copying the artifact directory from the linked Google Drive, ensure that `data/feats.zip` and the official annotation CSV/TSV manifests are present within the artifact root. Then unpack the saved train/validation features:

```powershell
$env:EVENTLENS_ARTIFACT_ROOT = 'C:\EventLens-AI-Artifacts' # change this to your actual copied/mounted Drive folder
python scripts/prepare_training_assets.py --artifact-root "$env:EVENTLENS_ARTIFACT_ROOT" --project-root .
```

The external dataset files must be obtained from the official LLP repository or the team Drive before this command. There is **no fully automated license-agnostic download script** for raw YouTube videos. The repository intentionally does not redistribute original footage.

To reproduce a **new** five-epoch run from scratch on the recorded *pretrained features* (not scratch feature extractors), use one of these commands. Each run writes a new checkpoint and JSON log under the **external** artifact root, never overwriting original logs:

```powershell
python src/train.py --artifact-root "$env:EVENTLENS_ARTIFACT_ROOT" --model baseline --seed 42
python src/train.py --artifact-root "$env:EVENTLENS_ARTIFACT_ROOT" --model bigru --seed 42
python src/train.py --artifact-root "$env:EVENTLENS_ARTIFACT_ROOT" --model transformer --seed 42
python src/train.py --artifact-root "$env:EVENTLENS_ARTIFACT_ROOT" --model cross_modal --seed 42
```

Change `--seed` to 43 and 44 to reproduce the recorded three-seed protocol. `configs/shared_protocol.json` records the shared final settings. The *proposed but not yet executed* equally budgeted two-learning-rate sweep appears in `configs/fair_lr_sweep_plan.json` and `scripts/fair_lr_sweep.ps1`. Do not treat it as a completed result.

The same custom pooled-annotation evaluator handles all four model families. Example validation evaluation of an **existing** trained checkpoint:

```powershell
python src/evaluate.py --model bigru --checkpoint "$env:EVENTLENS_ARTIFACT_ROOT\experiments\multi_seed\bigru_seed44.pt" --split validation
```

For a **single authorized final** test evaluation, use `--split test --allow-test` only after restoring the held-out feature arrays. The original final test was performed on BiGRU seed 44; repeated tuning on test data is disallowed.

The new `src/train.py` includes `gpu_peak_allocated_mib` and `gpu_peak_reserved_mib` in its **new-run logs** when CUDA is available. No unmeasured historical GPU memory number has been retroactively filled in.

For a genuine five-minute live demo, select `Features (.npz)` and use previously saved audio/visual/motion features from a real test video. This is cached preprocessing and **live head inference**, not fresh full-video extraction; see [`demo/README.md`](demo/README.md) and `scripts/prepare_demo_cache.py`.

The current code layout retains the canonical `src/eventlens/` implementations and FastAPI `web/` app instead of duplicating models. `src/train.py`, `src/evaluate.py`, `src/models/`, `src/datasets/`, `configs/`, `data/prepare.py` and `demo/app.py` expose the requested functional entrypoints via thin compatibility imports. This structure satisfies the purpose of the requested layout without maintaining duplicate code paths.

## Credits and research citations

- Tian, Li & Xu, *Unified Multisensory Perception* (ECCV 2020), LLP dataset: https://github.com/YapengTian/AVVP-ECCV20
- He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016): https://arxiv.org/abs/1512.03385
- Tran et al., *A Closer Look at Spatiotemporal Convolutions for Action Recognition* (CVPR 2018): https://arxiv.org/abs/1711.11248
- Google AudioSet VGGish feature encoder: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (ICLR 2019): https://arxiv.org/abs/1711.05101
- Cho et al., GRU (EMNLP 2014): https://aclanthology.org/D14-1179/
- Vaswani et al., Transformer (NeurIPS 2017): https://arxiv.org/abs/1706.03762
- FastAPI framework: https://fastapi.tiangolo.com/

Please consult the original repositories for pretrained weight terms; no blanket video redistribution permission is asserted.

## Scientific limitations

- **Pretrained vs scratch:** no verified controlled feature-extractor pretraining ablation has been run. The head models were trained from scratch on frozen *pretrained* features; this is not an equivalent substitute.
- **Memory:** peak GPU/CPU memory for every architecture was not recorded in the available experiment logs.
- **Metric:** union of temporal audio and visual GT is a custom event metric; not official AVVP modality-specific evaluation.
- **Scope:** 10-second clips, one-second boundaries, 25 classes. No long-video search, streaming or production service.
- **Reproduction:** Original notebook and original run logs are preserved on team Drive; the repo provides new training/evaluation CLI entrypoints. The published CLI was added after the original experiments and therefore does not imply that it ran the recorded historical trials.
- **Data rights:** check upstream LLP licensing and redistribute no restricted raw video.

## Deliverables and individual contribution evidence

The repository includes the 23-page technical report, slides, and source link in `report/`. The official email ZIP contains `Expired_Report.pdf`, `Expired_Slides.pptx`, and `GitHub_URL.txt`; large external assets remain linked above. The `contributions/TEAM_ASSIGNMENT_DRAFT.md` file allocates verifiable final QA responsibilities; the six separate editable draft PDFs are provided outside the official submission ZIP. Each student must verify actual weekly work, hours, PR/run evidence and obtain a real peer signature.

## References

1. Tian et al. (ECCV 2020), Unified Multisensory Perception: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/513_ECCV_2020_paper.php
2. Official LLP/AVVP repo: https://github.com/YapengTian/AVVP-ECCV20
3. Cho et al. (EMNLP 2014) GRU: https://aclanthology.org/D14-1179/
4. Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762
5. VGGish: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
6. Torchvision ResNet / video models: https://docs.pytorch.org/vision/stable/models.html
7. FastAPI: https://fastapi.tiangolo.com/
