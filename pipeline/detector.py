"""
detector.py
-----------
YOLOv8 object detection + ByteTrack tracking wrapper.

Returns supervision Detections objects per frame, annotated frames (numpy),
and accumulates data for COCO JSON export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    import supervision as sv
    _DEPS_OK = True
except ImportError as e:
    _DEPS_OK = False
    _IMPORT_ERROR = str(e)


@dataclass
class DetectionResult:
    """Accumulated detection results ready for export."""
    frame_results: list[dict] = field(default_factory=list)
    category_map: dict[int, str] = field(default_factory=dict)  # id -> name


def run_detection_pipeline(
    video_path: str,
    frame_indices: list[int],
    model_size: str = "yolov8n.pt",
    conf_threshold: float = 0.30,
    iou_threshold: float = 0.45,
    progress_cb=None,
) -> tuple[DetectionResult, list[np.ndarray]]:
    """
    Run YOLOv8 + ByteTrack on the specified frame indices.

    Parameters
    ----------
    video_path : str
        Path to input video.
    frame_indices : list[int]
        Sorted list of 0-based frame indices to process.
    model_size : str
        YOLO model weights file or name (e.g. "yolov8n.pt").
    conf_threshold : float
        Detection confidence threshold.
    iou_threshold : float
        NMS IoU threshold.
    progress_cb : callable | None
        Optional callback(current, total) for UI progress.

    Returns
    -------
    (DetectionResult, annotated_frames)
        DetectionResult contains per-frame data; annotated_frames is a list
        of BGR numpy arrays (one per processed frame) with bounding-box overlays.
    """
    if not _DEPS_OK:
        raise RuntimeError(f"Missing dependencies: {_IMPORT_ERROR}")

    model = YOLO(model_size)

    # ByteTrack tracker via supervision
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    result = DetectionResult()
    annotated_frames: list[np.ndarray] = []

    # Build category map from model names
    for cid, cname in model.names.items():
        result.category_map[int(cid) + 1] = cname  # COCO uses 1-based IDs

    cap = cv2.VideoCapture(video_path)
    frame_set = set(frame_indices)
    total = len(frame_indices)
    processed = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_set:
            processed += 1
            if progress_cb:
                progress_cb(processed, total)

            # YOLO inference
            yolo_results = model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )[0]

            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(yolo_results)

            # ByteTrack update
            detections = tracker.update_with_detections(detections)

            # Annotate frame
            labels = [
                f"#{tid} {model.names[int(cid)]}"
                for tid, cid in zip(
                    detections.tracker_id if detections.tracker_id is not None
                    else [None] * len(detections),
                    detections.class_id if detections.class_id is not None
                    else [],
                )
            ]
            annotated = frame.copy()
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            annotated_frames.append(annotated)

            # Store per-frame COCO-style data
            frame_data: dict = {
                "frame_id": frame_idx,
                "annotations": [],
            }
            if detections.xyxy is not None:
                h, w = frame.shape[:2]
                for i, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = box
                    bw, bh = float(x2 - x1), float(y2 - y1)
                    ann = {
                        "bbox": [float(x1), float(y1), bw, bh],  # COCO xywh
                        "category_id": int(detections.class_id[i]) + 1
                        if detections.class_id is not None
                        else 0,
                        "score": float(detections.confidence[i])
                        if detections.confidence is not None
                        else 1.0,
                        "track_id": int(detections.tracker_id[i])
                        if detections.tracker_id is not None
                        else -1,
                        "frame_width": w,
                        "frame_height": h,
                    }
                    frame_data["annotations"].append(ann)

            result.frame_results.append(frame_data)

        frame_idx += 1

    cap.release()
    return result, annotated_frames
