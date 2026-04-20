# drunk_driving_detection
Embedded software on a scalable hardware stack to detect drunk driver behind the wheel in near real-time.

# DrunkDrivingScan — Real-Time Intoxication Analyzer

A Python + web application that uses your webcam and MediaPipe face mesh to analyze facial signals associated with intoxication.

## Signals Analyzed

| Signal | What it measures |
|--------|-----------------|
| **Eye Droopiness** | Eye Aspect Ratio (EAR) — heavy/droopy eyelids are a classic impairment indicator |
| **Blink Rate** | Blinks per minute — drunk people blink slower (<10 bpm) or erratically |
| **Head Stability** | Roll & pitch angle + variance — unsteady head movement |
| **Facial Symmetry** | Left/right landmark distances from midline — alcohol causes subtle asymmetry |

## Setup

### 1. Prerequisites
- Python 3.9–3.11 (MediaPipe works best on these)
- A webcam

### 2. Install dependencies
```bash
cd drunk_detector
pip install -r requirements.txt
```

> **macOS / Linux**: You may need `pip3` instead of `pip`.
> **Windows**: Use a virtual environment to avoid conflicts.

### 3. Run
```bash
python app.py
```

### 4. Open in browser
Navigate to: **http://localhost:5000**

Click **"ACTIVATE CAMERA"** and allow camera access.

---

## How it works

1. Your browser captures webcam frames every 250ms
2. Each frame is JPEG-compressed and sent as base64 to the Flask backend
3. MediaPipe Face Mesh detects 468 3D facial landmarks
4. Four signal algorithms run on the landmarks (see `app.py`)
5. A weighted overall score (0–100) is returned to the frontend
6. The UI updates in real time with per-signal breakdowns

## Score thresholds

| Score | Verdict |
|-------|---------|
| 0–24  | 🟢 Sober |
| 25–49 | 🟡 Mild Signs |
| 50–74 | 🟠 Moderate Impairment |
| 75+   | 🔴 Severe Impairment |

## ⚠️ Disclaimer

This is a computer vision heuristic tool — **NOT** a medically validated or legally admissible sobriety test. Scores are affected by lighting, glasses, fatigue, medical conditions, and camera angle.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: mediapipe` | Run `pip install mediapipe` |
| Camera not working | Allow camera in browser permissions |
| "Backend Offline" in UI | Make sure `python app.py` is running |
| Poor detection | Ensure good frontal lighting, remove glasses if possible |
| mediapipe error on Python 3.12 | Use Python 3.10 or 3.11 |
