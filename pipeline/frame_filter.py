"""
frame_filter.py
---------------
Smart frame filtering using YOLO-nano or CLIP-style text-guided relevance scan.

Strategy:
  1. Run YOLO nano on every Nth frame (fast relevance pass).
  2. Score each frame: max confidence among detections whose class name
     overlaps with the user's text description.
  3. Keep only frames above a relevance threshold.

If torch / CLIP is not available we fall back to a simple uniform sampling.
"""

from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


def _tokenize_description(description: str) -> list[str]:
    """Lower-case words from the user's text description."""
    return [w.lower() for w in re.split(r"\W+", description) if w]


def _frame_score_yolo(
    frame: np.ndarray,
    model,
    desc_tokens: list[str],
    device: str | int = "cpu",
) -> float:
    """Return relevance score [0, 1] for a single frame using YOLO."""
    results = model(frame, device=device, verbose=False)[0]
    best = 0.0
    if results.boxes is None:
        return best
    for cls_id, conf in zip(
        results.boxes.cls.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
    ):
        cls_name = model.names[int(cls_id)].lower()
        # Match if any token from the description is a substring of the class name
        # or the class name is a substring of any token.
        for tok in desc_tokens:
            if tok in cls_name or cls_name in tok:
                best = max(best, float(conf))
                break
    return best


def filter_relevant_frames(
    video_path: str,
    description: str,
    sample_every: int = 5,
    relevance_threshold: float = 0.25,
    max_frames: Optional[int] = None,
    progress_cb=None,
) -> list[int]:
    """
    Return a sorted list of frame indices that are relevant to *description*.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    description : str
        User's natural-language description of objects of interest.
    sample_every : int
        Stride for the fast relevance pass (default: every 5th frame).
    relevance_threshold : float
        Minimum YOLO confidence to consider a frame relevant.
    max_frames : int | None
        Cap the total number of returned frame indices (None = no cap).
    progress_cb : callable | None
        Optional callback(current, total) for progress reporting.

    Returns
    -------
    list[int]
        Sorted list of 0-based frame indices to process.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        raise ValueError(f"Cannot read video or video is empty: {video_path}")

    desc_tokens = _tokenize_description(description)

    # Try YOLO nano for relevance scoring
    model = None
    device = "cpu"
    if _YOLO_AVAILABLE and desc_tokens:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO("yolov8n.pt")  # nano — fast
        except Exception:
            model = None

    relevant: list[int] = []

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    sampled = 0
    total_sampled = (total_frames + sample_every - 1) // sample_every

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            sampled += 1
            if progress_cb:
                progress_cb(sampled, total_sampled)

            if model is not None:
                score = _frame_score_yolo(frame, model, desc_tokens, device=device)
                if score >= relevance_threshold:
                    # Mark surrounding frames as relevant (±sample_every window)
                    start = max(0, frame_idx - sample_every + 1)
                    end = min(total_frames - 1, frame_idx + sample_every - 1)
                    for fi in range(start, end + 1):
                        relevant.append(fi)
            else:
                # Fallback: keep every sampled frame
                relevant.append(frame_idx)

        frame_idx += 1

    cap.release()

    # De-duplicate and sort
    relevant = sorted(set(relevant))

    # If nothing was detected as relevant, keep all sampled frames (graceful degradation)
    if not relevant:
        relevant = list(range(0, total_frames, sample_every))

    if max_frames is not None:
        # Evenly sub-sample if too many
        if len(relevant) > max_frames:
            indices = np.linspace(0, len(relevant) - 1, max_frames, dtype=int)
            relevant = [relevant[i] for i in indices]

    return relevant
