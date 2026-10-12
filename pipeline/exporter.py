"""
exporter.py
-----------
Convert DetectionResult → COCO JSON + zip archive.

COCO JSON format (detection/tracking variant):
  {
    "info": {...},
    "licenses": [],
    "categories": [...],
    "images": [...],
    "annotations": [...]
  }
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime

import cv2
import numpy as np

from .detector import DetectionResult


def build_coco_json(
    result: DetectionResult,
    video_path: str,
    description: str,
) -> dict:
    """Build a COCO-format dict from a DetectionResult."""
    now = datetime.utcnow().isoformat()

    info = {
        "description": f"RasoDataForge export — {description}",
        "url": "https://github.com/rasodataforge",
        "version": "1.0",
        "year": datetime.utcnow().year,
        "contributor": "RasoDataForge",
        "date_created": now,
    }

    categories = [
        {"id": cid, "name": cname, "supercategory": "object"}
        for cid, cname in result.category_map.items()
    ]

    images = []
    annotations = []
    ann_id = 1

    for frame_data in result.frame_results:
        fid = frame_data["frame_id"]
        img_entry: dict = {
            "id": fid,
            "file_name": f"frame_{fid:06d}.jpg",
            "date_captured": now,
        }
        # Add width/height from first annotation if available
        if frame_data["annotations"]:
            ann0 = frame_data["annotations"][0]
            img_entry["width"] = ann0["frame_width"]
            img_entry["height"] = ann0["frame_height"]
        else:
            img_entry["width"] = 0
            img_entry["height"] = 0
        images.append(img_entry)

        for ann in frame_data["annotations"]:
            x, y, w, h = ann["bbox"]
            annotation = {
                "id": ann_id,
                "image_id": fid,
                "category_id": ann["category_id"],
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "score": round(ann["score"], 4),
                "track_id": ann["track_id"],
            }
            annotations.append(annotation)
            ann_id += 1

    return {
        "info": info,
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def export_to_zip(
    result: DetectionResult,
    annotated_frames: list[np.ndarray],
    frame_indices: list[int],
    video_path: str,
    description: str,
    output_path: str,
    max_preview_frames: int = 20,
) -> str:
    """
    Write a zip file containing:
      - annotations.json  (COCO format)
      - frames/frame_XXXXXX.jpg  (annotated preview frames, up to max_preview_frames)

    Returns the path to the zip file.
    """
    coco = build_coco_json(result, video_path, description)

    # Subsample preview frames to avoid huge zips
    if getattr(result, "preview_frame_indices", None) and len(result.preview_frame_indices) == len(annotated_frames):
        if len(annotated_frames) > max_preview_frames:
            indices = np.linspace(0, len(annotated_frames) - 1, max_preview_frames, dtype=int)
            preview_frames = [annotated_frames[i] for i in indices]
            preview_fids = [result.preview_frame_indices[i] for i in indices]
        else:
            preview_frames = annotated_frames
            preview_fids = result.preview_frame_indices
    elif len(annotated_frames) > max_preview_frames:
        indices = np.linspace(0, len(annotated_frames) - 1, max_preview_frames, dtype=int)
        preview_frames = [annotated_frames[i] for i in indices]
        preview_fids = [frame_indices[i] for i in indices]
    else:
        preview_frames = annotated_frames
        preview_fids = frame_indices[: len(annotated_frames)]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write COCO JSON
        zf.writestr("annotations.json", json.dumps(coco, indent=2))

        # Write annotated frames
        for fid, frame in zip(preview_fids, preview_frames):
            success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                zf.writestr(f"frames/frame_{fid:06d}.jpg", buf.tobytes())

    return output_path
