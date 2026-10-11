"""
test_pipeline.py
----------------
Smoke-test for RasoDataForge pipeline (no UI, no real video required).

Tests:
  1. frame_filter.filter_relevant_frames — runs on a synthetic video
  2. detector.run_detection_pipeline — runs on same synthetic video
  3. exporter.export_to_zip — checks zip is created and contains annotations.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import pytest

# Make sure we import from local src
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.frame_filter import filter_relevant_frames
from pipeline.detector import run_detection_pipeline, DetectionResult
from pipeline.exporter import export_to_zip, build_coco_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Create a tiny 30-frame synthetic video (640×480, 10 fps, random noise)."""
    tmp_dir = tmp_path_factory.mktemp("video")
    video_path = str(tmp_dir / "test_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
    rng = np.random.default_rng(42)
    for _ in range(30):
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    assert os.path.exists(video_path), "Synthetic video was not created"
    return video_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFrameFilter:
    def test_returns_list_of_ints(self, synthetic_video):
        frames = filter_relevant_frames(
            synthetic_video, description="person", sample_every=5,
            relevance_threshold=0.99,  # high threshold → will fallback to sampled frames
        )
        assert isinstance(frames, list)
        assert all(isinstance(f, int) for f in frames)
        assert len(frames) > 0

    def test_max_frames_respected(self, synthetic_video):
        frames = filter_relevant_frames(
            synthetic_video, description="car", sample_every=2, max_frames=5,
        )
        assert len(frames) <= 5

    def test_empty_description_fallback(self, synthetic_video):
        frames = filter_relevant_frames(
            synthetic_video, description="", sample_every=5,
        )
        assert len(frames) > 0


class TestDetectionPipeline:
    def test_returns_tuple(self, synthetic_video):
        frames = list(range(0, 10, 2))  # frames 0, 2, 4, 6, 8
        result, annotated = run_detection_pipeline(
            synthetic_video, frame_indices=frames, model_size="yolov8n.pt"
        )
        assert isinstance(result, DetectionResult)
        assert isinstance(annotated, list)
        assert len(annotated) == len(frames)

    def test_annotated_frames_are_images(self, synthetic_video):
        frames = [0, 5, 10]
        _, annotated = run_detection_pipeline(
            synthetic_video, frame_indices=frames, model_size="yolov8n.pt"
        )
        for img in annotated:
            assert isinstance(img, np.ndarray)
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_category_map_populated(self, synthetic_video):
        frames = [0]
        result, _ = run_detection_pipeline(
            synthetic_video, frame_indices=frames, model_size="yolov8n.pt"
        )
        assert len(result.category_map) > 0


class TestExporter:
    def test_zip_created(self, synthetic_video, tmp_path):
        frames = [0, 5, 10]
        result, annotated = run_detection_pipeline(
            synthetic_video, frame_indices=frames, model_size="yolov8n.pt"
        )
        zip_path = str(tmp_path / "output.zip")
        out = export_to_zip(
            result=result,
            annotated_frames=annotated,
            frame_indices=frames,
            video_path=synthetic_video,
            description="test objects",
            output_path=zip_path,
        )
        assert os.path.exists(out)
        assert zipfile.is_zipfile(out)

    def test_zip_contains_coco_json(self, synthetic_video, tmp_path):
        frames = [0, 5, 10]
        result, annotated = run_detection_pipeline(
            synthetic_video, frame_indices=frames, model_size="yolov8n.pt"
        )
        zip_path = str(tmp_path / "output2.zip")
        export_to_zip(
            result=result,
            annotated_frames=annotated,
            frame_indices=frames,
            video_path=synthetic_video,
            description="test objects",
            output_path=zip_path,
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "annotations.json" in names
            data = json.loads(zf.read("annotations.json"))
            for key in ("info", "categories", "images", "annotations"):
                assert key in data

    def test_coco_json_structure(self, synthetic_video):
        frames = [0, 5]
        result, _ = run_detection_pipeline(
            synthetic_video, frame_indices=frames, model_size="yolov8n.pt"
        )
        coco = build_coco_json(result, synthetic_video, "smoke test")
        assert "info" in coco
        assert "categories" in coco
        assert isinstance(coco["categories"], list)
        assert isinstance(coco["images"], list)
        assert isinstance(coco["annotations"], list)
