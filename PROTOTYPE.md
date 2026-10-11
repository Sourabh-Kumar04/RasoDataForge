# PROTOTYPE.md — RasoDataForge Honest Limitations

> This document is a candid summary of what the prototype does, what it does not do,
> and what the known limitations are. Required reading before the demo.

---

## ✅ What Works

| Feature | Status | Notes |
|---|---|---|
| Video ingestion | ✅ | Any format OpenCV can decode (MP4, AVI, MOV, MKV) |
| Smart frame filtering | ✅ | YOLO-nano scans every Nth frame; keeps only frames where described objects are detected above threshold |
| YOLOv8 detection | ✅ | Nano (fastest) to Medium models available; detects 80 COCO classes out of the box |
| ByteTrack tracking | ✅ | Per-object track IDs across frames via `supervision` |
| COCO JSON export | ✅ | Standard `annotations.json` with `images`, `categories`, `annotations` arrays |
| Annotated preview mosaic | ✅ | 3×3 grid of annotated frames shown in UI |
| Zip download | ✅ | Contains `annotations.json` + up to 20 annotated frame JPEGs |
| Gradio UI | ✅ | Single-page, runs locally on port 7860 |
| Headless / CLI use | ✅ | All three stages importable as Python modules |
| Pytest smoke tests | ✅ | Synthetic video; no real footage required |

---

## ⚠️ Known Limitations

### Segmentation (SAM 3 / YOLOv8-seg)
- **SAM 3** is too large to run reliably on CPU in a 3-minute demo.
- **YOLOv8-seg** is available as a drop-in upgrade (change `model_size` to `yolov8n-seg.pt`)
  but **not enabled by default** because segmentation masks increase zip size 5–10×.
- This prototype exports **bounding boxes only**. Pixel-level masks are a v2 feature.

### Text-guided relevance (CLIP vs YOLO-nano)
- The "smart frame filter" uses **YOLO-nano class name matching** (substring heuristic),
  not a true CLIP embedding similarity. This means:
  - It works well for single-word COCO class names (`person`, `car`, `dog`).
  - Multi-word or abstract descriptions (`"person carrying heavy box"`) only match if the
    noun (`person`) is a COCO class.
- True CLIP integration is wired to a stub; it will activate once `clip` package is installed.

### Performance
- CPU-only: ~2–5 fps (YOLOv8n). GPU reduces this to real-time.
- Large videos (>10 min, >1 GB) may cause OOM in the current implementation because
  annotated frames are held in RAM. Fix: stream frames to disk before zipping.

### COCO export completeness
- `iscrowd` is always `0` (no crowd region support).
- `segmentation` field is omitted (boxes only).
- `image.file_name` refers to synthetic names; actual frame extraction to disk would
  be needed for a lossless round-trip.

### UI
- No real-time streaming preview — the UI blocks until the full pipeline finishes.
- Single-user; Gradio's default queue is disabled for simplicity.
- No authentication or file-size guard; not suitable for public deployment as-is.

---

## 🔭 Roadmap (Post-Stage I)

1. **CLIP-guided relevance** — real embedding similarity, not substring matching.
2. **YOLOv8-seg / SAM 3** — pixel-level masks exported as COCO `segmentation` polygons.
3. **Streaming frame export** — avoid holding all annotated frames in RAM.
4. **Confidence calibration** — per-class threshold tuning.
5. **Dataset quality metrics** — class balance, annotation density heatmaps.
6. **Cloud export** — push COCO zip directly to Roboflow / HuggingFace Datasets.

---

## 🧪 Test Status

```
pytest test_pipeline.py -v
```

| Test | Status |
|---|---|
| frame_filter returns list of ints | ✅ |
| max_frames respected | ✅ |
| empty description fallback | ✅ |
| detection returns tuple | ✅ |
| annotated frames are valid images | ✅ |
| category_map populated | ✅ |
| zip created | ✅ |
| zip contains annotations.json | ✅ |
| COCO JSON structure valid | ✅ |

---

## Hardware Tested

| Environment | Result |
|---|---|
| Ubuntu 22.04, CPU-only, Python 3.12 | ✅ All tests pass |
| Windows 11 WSL2, no GPU | ✅ All tests pass |
| GPU (expected) | ⬜ Not yet tested |
