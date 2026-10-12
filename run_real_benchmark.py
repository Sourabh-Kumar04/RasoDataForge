"""
run_real_benchmark.py
---------------------
Runs the full RasoDataForge pipeline on /tmp/real_bench.mp4 and logs real measured numbers.
"""

import os
import sys
import time
import json
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.frame_filter import filter_relevant_frames
from pipeline.detector import run_detection_pipeline
from pipeline.exporter import build_coco_json, export_to_zip

VIDEO_PATH = "/tmp/real_bench.mp4"
DESCRIPTION = "person walking"
ZIP_OUTPUT = "/tmp/real_bench_export.zip"

def main():
    print(f"=== RasoDataForge Real-Video Benchmark ===")
    print(f"Video path: {VIDEO_PATH}")
    print(f"Description: '{DESCRIPTION}'")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"{VIDEO_PATH} does not exist!")

    # Video properties
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"Video Specs: {width}x{height} @ {fps:.2f} fps, {total_frames} frames ({duration:.2f}s)")

    # Overall timer start
    e2e_t0 = time.time()

    # Step 1: Smart frame filtering
    print("\n--- Step 1: filter_relevant_frames ---")
    filter_t0 = time.time()
    kept_frames = filter_relevant_frames(VIDEO_PATH, description=DESCRIPTION)
    filter_t1 = time.time()
    filter_time = filter_t1 - filter_t0

    num_kept = len(kept_frames)
    reduction_pct = (1.0 - (num_kept / total_frames)) * 100.0

    print(f"Filter Time: {filter_time:.3f} s")
    print(f"Frames Kept: {num_kept} / {total_frames}")
    print(f"Reduction: {reduction_pct:.2f}%")

    # Step 2: Detection pipeline (YOLOv8 + ByteTrack)
    print("\n--- Step 2: run_detection_pipeline ---")
    det_t0 = time.time()
    detection_result, annotated_frames = run_detection_pipeline(
        video_path=VIDEO_PATH,
        frame_indices=kept_frames,
        model_size="yolov8n.pt",
        conf_threshold=0.30,
    )
    det_t1 = time.time()
    det_time = det_t1 - det_t0
    det_fps = num_kept / det_time if det_time > 0 else 0

    total_boxes = sum(len(fr["annotations"]) for fr in detection_result.frame_results)
    
    unique_tracks = set()
    for fr in detection_result.frame_results:
        for ann in fr["annotations"]:
            tid = ann.get("track_id", -1)
            if tid != -1:
                unique_tracks.add(tid)
    num_unique_tracks = len(unique_tracks)

    print(f"Detection Time: {det_time:.3f} s")
    print(f"Detection Throughput: {det_fps:.2f} fps")
    print(f"Total Bounding Boxes: {total_boxes}")
    print(f"Unique Track IDs: {num_unique_tracks}")

    # Step 3: Build COCO JSON
    print("\n--- Step 3: build_coco_json ---")
    coco_t0 = time.time()
    coco_dict = build_coco_json(detection_result, VIDEO_PATH, DESCRIPTION)
    coco_t1 = time.time()
    coco_time = coco_t1 - coco_t0
    coco_annotations_count = len(coco_dict["annotations"])

    print(f"COCO JSON Build Time: {coco_time:.3f} s")
    print(f"COCO Annotations Count: {coco_annotations_count}")
    print(f"COCO Images Count: {len(coco_dict['images'])}")
    print(f"COCO Categories: {coco_dict['categories']}")

    # Step 4: Export to ZIP
    print("\n--- Step 4: export_to_zip ---")
    zip_t0 = time.time()
    out_zip = export_to_zip(
        result=detection_result,
        annotated_frames=annotated_frames,
        frame_indices=kept_frames,
        video_path=VIDEO_PATH,
        description=DESCRIPTION,
        output_path=ZIP_OUTPUT,
    )
    zip_t1 = time.time()
    zip_time = zip_t1 - zip_t0
    zip_size_mb = os.path.getsize(out_zip) / (1024 * 1024)

    print(f"ZIP Export Time: {zip_time:.3f} s")
    print(f"ZIP File: {out_zip} ({zip_size_mb:.2f} MB)")

    # Overall timer end
    e2e_t1 = time.time()
    e2e_time = e2e_t1 - e2e_t0

    print("\n==========================================")
    print("           BENCHMARK RESULTS              ")
    print("==========================================")
    print(f"Total Frames:             {total_frames}")
    print(f"Frames Kept:              {num_kept}")
    print(f"Filter Reduction:         {reduction_pct:.2f}%")
    print(f"Filter Elapsed:           {filter_time:.3f}s")
    print(f"Detection Pipeline Time:  {det_time:.3f}s")
    print(f"Detection FPS:            {det_fps:.2f} fps")
    print(f"Total Boxes:              {total_boxes}")
    print(f"Unique Track IDs:         {num_unique_tracks}")
    print(f"COCO Annotation Count:    {coco_annotations_count}")
    print(f"Export ZIP Time:          {zip_time:.3f}s")
    print(f"End-to-End Time:          {e2e_time:.3f}s")
    print("==========================================")

    # Save raw json for record
    results = {
        "video": {
            "path": VIDEO_PATH,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "duration_s": duration,
        },
        "description": DESCRIPTION,
        "total_frames": total_frames,
        "frames_kept": num_kept,
        "reduction_pct": reduction_pct,
        "filter_time_s": filter_time,
        "detection_time_s": det_time,
        "detection_fps": det_fps,
        "total_boxes": total_boxes,
        "unique_track_ids": num_unique_tracks,
        "coco_annotation_count": coco_annotations_count,
        "zip_time_s": zip_time,
        "end_to_end_time_s": e2e_time,
        "zip_size_mb": zip_size_mb,
    }

    with open("/tmp/real_benchmark_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
