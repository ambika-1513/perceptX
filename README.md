# PerceptX 🚗

> Real-time multi-object detection, segmentation & tracking pipeline for autonomous driving perception.

Built as part of my Computer Vision & ADAS learning journey.

**Stack:** YOLOv8 · Segment Anything Model (SAM) · ByteTrack · OpenCV · PyTorch

---

## What it does

PerceptX processes dashcam footage frame by frame through 3 stages:

```
Video Frame
    │
    ▼
[YOLOv8]      →  Detects objects (cars, trucks, pedestrians, traffic lights...)
    │
    ▼
[ByteTrack]   →  Assigns persistent ID to each object across frames
    │
    ▼
[SAM]         →  Generates pixel-perfect mask for each detected object
    │
    ▼
Annotated Output Video
```

## Demo

<p align="center">
  <video src="https://github.com/user-attachments/assets/f71e790d-1db4-41bb-a766-fb4dbecb32b0" width="700" controls></video>
</p>

*20-second clip — bounding boxes (YOLOv8), persistent track IDs (ByteTrack), and segmentation masks (SAM) overlaid on real dashcam footage.*

> 🎥 [Full 5-minute output video](https://drive.google.com/file/d/1dBlqbxjJFIQRakYPcRGPwWaQijI59lVV/view?usp=sharing) 





<!--
If the video tag above doesn't render for some reason, use a GIF instead:
<img src="assets/demo.gif" width="700" alt="PerceptX demo">
-->

## Why this matters for ADAS

- **Detection alone isn't enough** — a car behind a truck briefly disappears and reappears. Without tracking, the system thinks it's a new car every time.
- **Bounding boxes aren't enough** — knowing exact pixel boundaries of objects helps with distance estimation, lane occupancy, and collision prediction.
- **PerceptX combines all 3** into a single clean pipeline.

## Project Structure

```
perceptx/
├── pyproject.toml
├── perceptx/
│   ├── detect.py       # YOLOv8 detection + ADAS class filtering
│   ├── segment.py      # SAM batch segmentation using boxes as prompts
│   ├── track.py        # ByteTrack multi-object tracking
│   └── pipeline.py     # Orchestrates detect → track → segment
├── scripts/
│   └── run.py          # CLI entry point
└── data/               # Put input videos here
```

## Setup

```bash
git clone https://github.com/ambika-1513/perceptX.git
cd perceptX
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e .
```

## Usage

**Detection + Tracking only (fast):**
```bash
python scripts/run.py --video data/dashcam.mp4 --device cuda
```

**Full pipeline with SAM segmentation:**
```bash
python scripts/run.py --video data/dashcam.mp4 --segment --sam-checkpoint sam_vit_b_01ec64.pth
```

**CPU mode:**
```bash
python scripts/run.py --video data/dashcam.mp4 --device cpu
```

## Running on Google Colab (recommended — T4 GPU)

No local GPU needed. Open `PerceptX_Colab.ipynb` in Colab — it handles everything:
- GPU verification
- Repo clone + install
- SAM checkpoint download (cached to Drive)
- Video upload + inference
- Output preview + save

## Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| Detection | YOLOv8s | ~22MB | Auto-downloaded |
| Segmentation | SAM ViT-B | ~375MB | Download via Colab |
| Tracking | ByteTrack | — | Built into supervision |

## Performance

Measured on [GPU name, e.g. Colab T4]:

| Mode | FPS | Notes |
|------|-----|-------|
| Detection + Tracking | ~XX FPS | YOLOv8s, 640px input |
| Full pipeline (+ SAM) | ~XX FPS | SAM is the bottleneck — segmentation adds the most latency |

> Replace the XX values with your actual benchmark — run `run.py` on a fixed clip and time it (e.g. `frames / total_seconds`), or log it directly inside `pipeline.py`.

## Detected Classes

Cars · Trucks · Buses · Motorcycles · Bicycles · Persons · Traffic Lights · Stop Signs

---

*Part of my journey toward Computer Vision & Perception Engineering in the automotive/ADAS space.*
