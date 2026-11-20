# EventLens AI — Audio–Visual Event Localization

**Deep Learning AI403 · Fall 2026 · Project idea #14**  
**Team W — Expired (pending final six-member roster confirmation)**  
Pharos University in Alexandria, Faculty of Computer Science & Artificial Intelligence.

## Registered team

| Name | ID |
|---|---|
| Ashraf Amr Ebrahim Abdelfattah Elsheikh | 202300962 |
| Ezzeldin Yasser Nabil Shamaa | 202303039 |
| Kareem Abdelnaser Mohamed Alazazy | 202303078 |
| Mohamed Ehab Abdelnaby Elsayed Fouda | 202303042 |
| Youssef Ali Abdallah Shalaby Elfar | 202302115 |
| Osama Khalid Goda Fadil Aboelhassan | 202303877 |

> **TA verification:** the course team sheet marks all six as accepted in Team W but has a blank team-name cell for the sixth student; the assignment text says teams of 4–5. Confirm team-size approval and the final name before emailing the submission.

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

Folders include `experiments/`, `weights/`, `tools/`, `checkpoints/`, `data/`. Access must be verified for teaching assistants; GitHub intentionally excludes large weights, media and raw data.

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

## Scientific limitations / known rubric gaps

- **Pretrained vs scratch:** no verified controlled feature-extractor pretraining ablation has been run. The head models were trained from scratch on frozen *pretrained* features; this is not an equivalent substitute.
- **Memory:** peak GPU/CPU memory for every architecture is not recorded; do not invent it.
- **Metric:** union of temporal audio and visual GT is a custom event metric; not official AVVP modality-specific evaluation.
- **Scope:** 10-second clips, one-second boundaries, 25 classes. No long-video search, streaming or production service.
- **Reproduction:** original complete training notebooks and large pretrained assets are in Google Drive, not tracked in Git; a clean training/evaluation command for every original experiment may need adaptation.
- **Data rights:** check upstream LLP licensing and redistribute no restricted raw video.

## Submission and permissions

- Report and slides are in `report/`; email attachment `DL_Project_Expired.zip` must include `Expired_Report.pdf`, `Expired_Slides.pptx`, and `GitHub_URL.txt`.
- Confirm every team member's name and the approved six-person roster, and verify teaching assistant access to GitHub and the shared Drive folder.
- Weekly individual contribution logs must contain each student's true tasks, hours, and evidence, signed by one teammate. Empty templates must be completed; **never fabricate contributions**.
- Check repository contents before the deadline; the handout marks the last commit before deadline.

## References

1. Tian et al. (ECCV 2020), Unified Multisensory Perception: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/513_ECCV_2020_paper.php
2. Official LLP/AVVP repo: https://github.com/YapengTian/AVVP-ECCV20
3. Cho et al. (EMNLP 2014) GRU: https://aclanthology.org/D14-1179/
4. Vaswani et al., Attention Is All You Need: https://arxiv.org/abs/1706.03762
5. VGGish: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
6. Torchvision ResNet / video models: https://docs.pytorch.org/vision/stable/models.html
7. FastAPI: https://fastapi.tiangolo.com/
