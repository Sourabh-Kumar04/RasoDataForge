"""
app.py
------
RasoDataForge Gradio UI

Upload a video + describe objects of interest → get annotated preview + COCO zip.
"""

from __future__ import annotations

import os
import tempfile
import time

import gradio as gr
import cv2
import numpy as np

from pipeline.frame_filter import filter_relevant_frames
from pipeline.detector import run_detection_pipeline
from pipeline.exporter import export_to_zip

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preview_gif_or_mosaic(
    annotated_frames: list[np.ndarray],
    max_tiles: int = 9,
) -> np.ndarray:
    """Return a mosaic image (3×3 grid) of annotated frames for the Gradio preview."""
    if not annotated_frames:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "No relevant frames detected",
            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )
        return placeholder

    # Sub-sample to max_tiles
    if len(annotated_frames) > max_tiles:
        idxs = np.linspace(0, len(annotated_frames) - 1, max_tiles, dtype=int)
        tiles = [annotated_frames[i] for i in idxs]
    else:
        tiles = list(annotated_frames)
        # Pad to a square
        while len(tiles) < max_tiles:
            tiles.append(np.zeros_like(tiles[0]))

    # Resize all tiles to the same size
    tw, th = 320, 180
    resized = [cv2.resize(t, (tw, th)) for t in tiles]

    cols = 3
    rows = (len(resized) + cols - 1) // cols
    # Pad last row
    while len(resized) < rows * cols:
        resized.append(np.zeros((th, tw, 3), dtype=np.uint8))

    mosaic_rows = [
        np.hstack(resized[r * cols: (r + 1) * cols]) for r in range(rows)
    ]
    mosaic = np.vstack(mosaic_rows)
    # Convert BGR → RGB for Gradio
    return cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_video(
    video_file,
    description: str,
    model_size: str,
    conf_threshold: float,
    sample_every: int,
    relevance_threshold: float,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Main processing function wired to the Gradio interface.

    Returns (preview_image, zip_path, status_message)
    """
    if video_file is None:
        return None, None, "⚠️ Please upload a video first."

    if not description.strip():
        description = "all objects"

    video_path = video_file  # Gradio passes the temp path directly

    status_parts: list[str] = []
    t0 = time.time()

    try:
        # --- Stage 1: Smart frame filtering ---
        progress(0, desc="🔍 Scanning frames for relevance…")
        status_parts.append("Stage 1: Frame filtering…")

        relevant_frames = filter_relevant_frames(
            video_path=video_path,
            description=description,
            sample_every=sample_every,
            relevance_threshold=relevance_threshold,
        )
        status_parts.append(f"  → {len(relevant_frames)} relevant frames selected.")

        # --- Stage 2: YOLOv8 + ByteTrack ---
        progress(0.3, desc="🎯 Running YOLOv8 + ByteTrack…")
        status_parts.append("Stage 2: Detection + tracking…")

        detection_result, annotated_frames = run_detection_pipeline(
            video_path=video_path,
            frame_indices=relevant_frames,
            model_size=model_size,
            conf_threshold=conf_threshold,
        )

        total_annotations = sum(
            len(fd["annotations"]) for fd in detection_result.frame_results
        )
        status_parts.append(
            f"  → {total_annotations} detections across {len(annotated_frames)} frames."
        )

        # --- Stage 3: Export ---
        progress(0.8, desc="📦 Building COCO zip…")
        status_parts.append("Stage 3: Exporting COCO JSON + zip…")

        tmp_dir = tempfile.mkdtemp(prefix="rasodataforge_")
        zip_path = os.path.join(tmp_dir, "rasodataforge_export.zip")

        export_to_zip(
            result=detection_result,
            annotated_frames=annotated_frames,
            frame_indices=relevant_frames,
            video_path=video_path,
            description=description,
            output_path=zip_path,
        )

        elapsed = time.time() - t0
        status_parts.append(f"\n✅ Done in {elapsed:.1f}s")
        status_msg = "\n".join(status_parts)

        # Build mosaic preview
        progress(0.95, desc="🖼️ Building preview…")
        mosaic = _make_preview_gif_or_mosaic(annotated_frames)

        progress(1.0, desc="Done")
        return mosaic, zip_path, status_msg

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        return None, None, f"❌ Error:\n{err}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

MODEL_CHOICES = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
]

with gr.Blocks(title="RasoDataForge", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔥 RasoDataForge
        **Video → YOLOv8 + ByteTrack → COCO Dataset**

        Upload a video, describe what you want to detect, and download a ready-to-train
        COCO-format annotation zip.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            description_input = gr.Textbox(
                label="Describe objects of interest",
                placeholder="e.g. 'person carrying box', 'car on highway', 'dog playing'",
                lines=2,
            )
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="yolov8n.pt",
                    label="YOLO model size",
                )
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.30, step=0.05,
                    label="Detection confidence threshold",
                )
                sample_slider = gr.Slider(
                    minimum=1, maximum=30, value=5, step=1,
                    label="Frame filter stride (every N frames scanned)",
                )
                relevance_slider = gr.Slider(
                    minimum=0.05, maximum=0.9, value=0.25, step=0.05,
                    label="Relevance threshold",
                )

            run_btn = gr.Button("🚀 Run Pipeline", variant="primary")

        with gr.Column(scale=2):
            preview_image = gr.Image(label="Annotated Preview (mosaic)", type="numpy")
            status_box = gr.Textbox(label="Pipeline Log", lines=10, interactive=False)
            download_btn = gr.File(label="📥 Download COCO Zip")

    run_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            description_input,
            model_dropdown,
            conf_slider,
            sample_slider,
            relevance_slider,
        ],
        outputs=[preview_image, download_btn, status_box],
    )

    gr.Markdown(
        """
        ---
        **RasoDataForge** · ASUS UGen AI League Stage I Demo ·
        Pipeline: YOLO-nano relevance scan → YOLOv8 detection → ByteTrack → COCO JSON
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
