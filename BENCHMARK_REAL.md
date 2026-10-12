# RasoDataForge Real-Video Benchmark Results

Real-world end-to-end benchmark executed on `/tmp/real_bench.mp4` with prompt `"person walking"`.

## Hardware & Environment
- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Compute Backend:** CUDA (PyTorch 2.x)
- **Model:** YOLOv8n (`yolov8n.pt`) + ByteTrack
- **Input Video:** `/tmp/real_bench.mp4` (Creative Commons / Public Domain workplace pedestrian video)

## Video Specifications
- **Resolution:** 1920 × 1080 (1080p)
- **Frame Rate:** 59.94 FPS
- **Duration:** 45.03 seconds
- **Total Input Frames:** 2,699

## Measured Pipeline Metrics

| Metric | Measured Value | Unit / Note |
|---|---|---|
| **Total Frames** | 2,699 | Input frames |
| **Frames Kept by Filter** | 2,147 | Frames with detected activity |
| **Filter Reduction** | 20.45 | % of idle frames skipped |
| **Filter Elapsed Time** | 26.414 | seconds |
| **Detection Time** | 21.797 | seconds (YOLOv8n + ByteTrack) |
| **Detection Throughput** | 98.50 | frames per second (FPS) |
| **Total Bounding Boxes** | 2,744 | detections across kept frames |
| **Unique Track IDs** | 9 | persistent ByteTrack object IDs |
| **COCO Annotation Count** | 2,744 | annotations in `annotations.json` |
| **COCO Image Count** | 2,147 | indexed images in COCO export |
| **ZIP Export Time** | 0.195 | seconds |
| **ZIP Archive Size** | 3.75 | MB (`/tmp/real_bench_export.zip`) |
| **End-to-End Elapsed Time** | 48.413 | seconds |
