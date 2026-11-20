# EventLens AI — Audio–Visual Event Localization

**AI403 Deep Learning Final Project · Fall 2026**  
Faculty of Computer Science & Artificial Intelligence, Pharos University in Alexandria  
**Selected topic:** #14, Audio-visual event localization  
**Team:** **[ADD VERIFIED TEAM MEMBER NAMES AND STUDENT IDs BEFORE SUBMISSION]**

EventLens AI accepts a **10-second video containing audio**, extracts visual, motion and audio features, and predicts event labels with time intervals. Its interactive web demo uses **FastAPI + HTML/CSS/JavaScript (no Gradio)**. Event predictions are produced by **local PyTorch models**, with no external AI prediction API.

## Demonstrated result

The selected **BiGRU (seed 44)** achieved **event micro-F1 = 0.4367** on the **held-out 1,200-video LLP test split**, using a **custom union-of-audio-and-visual-ground-truth event-matching protocol** (event IoU ≥ 0.50; probability threshold 0.50). This is **not** the official modality-specific AVVP leaderboard metric.

| Model | Validation clip micro-F1 | Validation segment micro-F1 | Validation event micro-F1 |
|---|---:|---:|---:|
| Temporal MLP baseline | 0.7033 ± 0.0253 | 0.5097 ± 0.0274 | 0.3032 ± 0.0210 |
| **Bidirectional GRU** | **0.7669 ± 0.0056** | **0.6531 ± 0.0144** | **0.4442 ± 0.0151** |
| Temporal Transformer | 0.7632 ± 0.0259 | 0.6502 ± 0.0409 | 0.4108 ± 0.0391 |
| Cross-Modal Attention | 0.5613 ± 0.0298 | 0.3874 ± 0.0305 | 0.2511 ± 0.0270 |

Validation results are mean ± sample SD across **three seeds (42, 43, 44)**, with five epochs each. The LLP split comprises **10,000 train / 649 validation / 1,200 test** clips. The BiGRU no-audio ablation lowered validation event F1 by approximately **7.60 percentage points**. Source logs/checkpoints live in the separate Drive artifacts, **not** this source repository.

## Features and architecture

1. Upload a **9.5–10.5-second video with an audio track** (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`), or a pre-extracted `.npz` file.
2. Extract **VGGish audio** `(10, 128)`, **ResNet152 RGB** `(10, 2048)`, and **R(2+1)D-18 motion** `(10, 512)` features.
3. Choose any subset of the four trained models: **Temporal MLP**, **BiGRU**, **Transformer**, and **Cross-Modal Attention**.
4. View predicted event names, confidence scores, 10-second timeline and video-seeking controls. The app supports comparing models on the same extracted features.

**Important limits:** trained on 10-second LLP clips and 25 event classes; output predictions can include false positives. Maximum video upload size: **100 MB**. New-video extraction is compute-intensive (about **100 seconds on the tested Colab CPU** for one example, not a formal latency benchmark). Arbitrary-length videos, full-video semantic search, live surveillance, and production cloud hosting are **not implemented**.

## Environment setup (Windows PowerShell)

**Requirements:** Python **3.10+** (3.10/3.11 recommended for a reproducible local setup), Git, FFmpeg and FFprobe on `PATH`. Check `ffmpeg -version` and `ffprobe -version`. See https://ffmpeg.org/download.html .

```powershell
git clone https://github.com/Ashrafamr11/EventLens-AI.git
cd EventLens-AI
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For **new-video extraction** (rather than `.npz` only), also install:

```powershell
python -m pip install -r requirements-video.txt
```

If the video packages have compatibility issues on a particular Windows/Python setup, the previously tested **Colab** runtime can run the same FastAPI application with the artifact root mounted from Google Drive. The included `.npz` API path requires no VGGish/TF runtime.

## Required pretrained and trained files (not committed to Git)

Create a **separate artifact directory** with this layout, using files already saved by the team:

```text
ARTIFACT_ROOT/
├── experiments/
│   └── multi_seed/
│       ├── baseline_seed42.pt
│       ├── bigru_seed44.pt
│       ├── transformer_seed44.pt
│       └── cross_modal_seed42.pt
├── weights/
│   ├── resnet152-b121ed2d.pth
│   └── r2plus1d18-kinetics400-v1.pth
└── tools/
    └── feature_extractor_original.zip
```

Model checkpoint filenames are resolved by `src/eventlens/model_registry.py` (additional older fallback paths are supported). The VGGish ZIP must contain the original `feature_extractor/vggish_model.ckpt` and VGGish Python source files; substituting a different audio encoder invalidates the recorded evaluation. **Do not commit these large/binary assets or raw LLP videos to the Git repository.**

**Shared team artifacts / checkpoints / original logs:** **[INSERT ACTUAL ACCESSIBLE GOOGLE DRIVE FOLDER LINK BEFORE SUBMISSION]**.

## Run the web demo

```powershell
# Replace this example with the real local artifact path
$env:EVENTLENS_ARTIFACT_ROOT = 'C:\EventLens-AI-Artifacts'
$env:EVENTLENS_VGGISH_ARCHIVE = 'C:\EventLens-AI-Artifacts\tools\feature_extractor_original.zip'
python -m uvicorn web.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The app reports missing assets via **http://127.0.0.1:8000/api/status** and available model checkpoints via `/api/models`. For a genuine 10-second video, use **Video upload**; for saved embeddings, use **Features (.npz)**. With Colab, set `EVENTLENS_ARTIFACT_ROOT=/content/drive/MyDrive/EventLens-AI` and open FastAPI via Colab's forwarded server port. Colab is a development runtime, **not a prerequisite** for the frontend.

### Reproduced real-video smoke test (20 November 2026)

The `data/demo_real_10s.mp4` example was processed with **all three extractors**, returning the exact feature dimensions above. The BiGRU produced two predicted events: **Dog 0–10 s (92.5%)** and **Speech 0–10 s (84.9%)**. Both the FastAPI **`/api/analyze-video`** endpoint and browser upload/timeline were exercised successfully on Colab (CPU). All four model families returned HTTP 200 on the extracted feature file. This is a *functionality check*, not a claim that both labels are ground truth.

## HTTP API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Service heartbeat |
| GET | `/api/status` | Check assets and runtime readiness |
| GET | `/api/models` | Model catalog and checkpoint availability |
| POST | `/api/analyze-video` | Upload real 10-second video for full extraction and inference |
| POST | `/api/analyze-features` | Upload `.npz` with `audio`, `visual`, `motion` arrays |
| POST | `/api/compare-features` | Send JSON feature tensors for comparison |

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

**Observed local result:** **19 passed**, with warnings only (20 November 2026). Unit/API tests validate routing and inference using small test checkpoints. The actual full extractor integration was verified separately in Colab using the original pretrained weights.

## Dataset and reproducibility

The dataset is **Look, Listen and Parse (LLP)** from the [official AVVP-ECCV20 repository](https://github.com/YapengTian/AVVP-ECCV20). Split annotation files include `AVVP_train.csv`, `AVVP_val_pd.csv`, `AVVP_test_pd.csv`; temporal audio/visual files include `AVVP_eval_audio.csv` and `AVVP_eval_visual.csv`. The original published feature package is external and was stored on team Drive, not redistributed here.

- Expected feature directories: `data/processed/feats/vggish/<id>.npy`, `res152/<id>.npy`, `r2plus1d_18/<id>.npy`.
- Core training functions: `src/eventlens/training.py` (`train_one_epoch`, `validate_one_epoch`) and `src/eventlens/experiment.py` (`fit_model`).
- Architecture files: `baseline_model.py`, `temporal_gru_model.py`, `temporal_transformer_model.py`, `cross_modal_model.py`.
- **Known reproduction gap:** The source-only repository currently does **not** include a ready-to-run training/evaluation CLI for each model nor all original experiment configurations and run logs. Link the original Colab notebooks and logs in the shared team folder for a complete research audit trail; do **not** invent experimental numbers or rerun training unnecessarily.

## Submission checklist (AI403 handout)

- [x] Code, four trained model families, functional FastAPI demo, baseline and three-seed comparison.
- [x] Real-video API/browser test and local unit tests.
- [ ] Add actual team member names/IDs and **real accessible Drive link**.
- [ ] Add final trained checkpoints + run logs to shared Drive and grant intended evaluators access.
- [ ] Supply **15–25-page scientific report**, slides, experiment graphs, per-class/error analysis, computational cost table, and individual contribution logs.
- [ ] Add teaching assistant(s) as GitHub collaborators and verify access.
- [ ] Email TA a **single `DL_Project_TeamName.zip`** containing `TeamName_Report.pdf`, `TeamName_Slides.pdf` (or `.pptx`) and a `.txt` file with the GitHub URL. Keep raw datasets out of the ZIP/Git history.

**Unverified rubric requirements must not be presented as completed:** data-strategy study, pretrained-vs-scratch comparison where applicable, full per-class/error counts, and computational-cost metrics for every architecture. See original experiment records to establish those claims.

## Sources and credits

- Tian et al., *Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing* (ECCV 2020): https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/513_ECCV_2020_paper.php
- LLP code, annotations, official pretrained feature package: https://github.com/YapengTian/AVVP-ECCV20
- Cho et al., GRU / RNN encoder-decoder (EMNLP 2014): https://aclanthology.org/D14-1179/
- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017): https://arxiv.org/abs/1706.03762
- PyTorch: https://pytorch.org/ ; Torchvision: https://docs.pytorch.org/vision/stable/models.html
- FastAPI: https://fastapi.tiangolo.com/
