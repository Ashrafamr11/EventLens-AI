# Demo entrypoint (FastAPI)

The production demo code remains in `web/api.py` and `web/static/`. `demo/app.py` is a thin, runnable compatibility entrypoint: it imports the original FastAPI app without duplicating code.

From repository root, after installing requirements and copying pretrained assets:

```powershell
$env:EVENTLENS_ARTIFACT_ROOT = 'C:\EventLens-AI-Artifacts'
$env:EVENTLENS_VGGISH_ARCHIVE = 'C:\EventLens-AI-Artifacts\tools\feature_extractor_original.zip'
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Check `/api/status` and `/api/models` to verify that expected checkpoints are accessible.

## Five-minute presentation: reliably show real inference

The already-observed end-to-end *CPU* extraction time for one test video was ~99.8 seconds. Therefore, do not depend on fresh CPU extraction completing during a five-minute live demo. Use a **real pre-extracted** feature file `.npz` obtained from your own previously tested video and load it in the **Features (.npz)** tab. This still performs model inference live and shows authentic timeline/events, while avoiding repeating pretrained feature extraction.

```powershell
python scripts/prepare_demo_cache.py 'C:\path\real_video_preextracted.npz' 'C:\path\demo_cached.npz'
```

Clearly state **"cached features; live model inference"** rather than implying full live video preprocessing. Separately demonstrate that a real MP4 can be uploaded and point to the previously recorded ~99.8s end-to-end smoke test. Only claim GPU extraction speed after measuring it on the actual demo runtime.

Bring the feature NPZ locally, keep the backend running before the presentation, test `/api/status`, and prepare an offline screenshot/video of prior real-MP4 feature extraction as a fallback, labeled as recorded rather than live.

## Presenter schedule (suggested, not evidence of completed work)

- Slides 1–3: Ashraf (problem / product scope)
- Slides 4–6: Ezzeldin (dataset / preprocessing)
- Slides 7–9: Kareem (models / evaluation)
- Slides 10–12: Mohamed (results / ablation)
- Slides 13–15: Youssef (errors / computational cost)
- Slides 16–18: Osama (demo / limitations / conclusion)

Rehearse to keep the slide portion at 15 minutes; live demo has a separate 5-minute allocation. Every student should be ready to answer questions about the whole pipeline, not just their slides.
