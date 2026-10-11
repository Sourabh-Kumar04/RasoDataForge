# RasoDataForge 🔥

**Video → COCO Dataset in minutes.**

RasoDataForge is a smart video annotation pipeline designed for the
**ASUS UGen AI League Stage I** demo. Upload a video, describe what you care about,
and get back a ready-to-train COCO-format annotation zip.

---

## Pipeline Overview

```
Video file
   │
   ▼
[Stage 1] Frame Filter  ←─ YOLO-nano relevance scan against text description
   │         (skips irrelevant frames, huge speed-up on long videos)
   ▼
[Stage 2] Detector      ←─ YOLOv8 detection + ByteTrack multi-object tracking
   │
   ▼
[Stage 3] Exporter      ←─ COCO JSON + annotated frame thumbnails → .zip
```

---

## Quickstart

### 1. Clone & set up

```bash
git clone https://github.com/you/rasodataforge.git
cd rasodataforge
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Gradio UI

```bash
python app.py
```

Open <http://localhost:7860> in your browser.

### 3. Use it

1. **Upload a video** (MP4, AVI, MOV…).
2. **Describe** the objects you want detected (e.g. `"person with helmet"`, `"car"`).
3. Click **🚀 Run Pipeline**.
4. Download the **COCO zip** — it contains `annotations.json` + annotated frame JPEGs.

---

## Advanced Settings

| Setting | Default | Description |
|---|---|---|
| YOLO model size | `yolov8n.pt` | `n` (fastest) → `s` → `m` (best accuracy) |
| Confidence threshold | 0.30 | Min detection confidence |
| Frame filter stride | 5 | Scan every Nth frame for relevance |
| Relevance threshold | 0.25 | Min YOLO score to consider a frame relevant |

---

## CLI / Headless Usage

You can also import the pipeline directly:

```python
from pipeline.frame_filter import filter_relevant_frames
from pipeline.detector import run_detection_pipeline
from pipeline.exporter import export_to_zip

frames = filter_relevant_frames("myvideo.mp4", description="person", sample_every=5)
result, annotated = run_detection_pipeline("myvideo.mp4", frames)
export_to_zip(result, annotated, frames, "myvideo.mp4", "person", "output.zip")
```

---

## Running Tests

```bash
source venv/bin/activate
pip install pytest
pytest test_pipeline.py -v
```

All tests use a synthetic video — no real footage needed.

---

## Project Structure

```
rasodataforge/
├── app.py                  # Gradio UI entry point
├── requirements.txt
├── test_pipeline.py        # Pytest smoke tests
├── pipeline/
│   ├── __init__.py
│   ├── frame_filter.py     # Smart frame relevance scan
│   ├── detector.py         # YOLOv8 + ByteTrack
│   └── exporter.py         # COCO JSON + zip export
├── README.md
└── PROTOTYPE.md            # Honest capability/limitation summary
```

---

## Tech Stack

| Component | Library |
|---|---|
| Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Tracking | ByteTrack via [Supervision](https://github.com/roboflow/supervision) |
| UI | [Gradio](https://gradio.app) |
| Deep learning | PyTorch |
| Export format | COCO JSON |

---

## License

MIT
