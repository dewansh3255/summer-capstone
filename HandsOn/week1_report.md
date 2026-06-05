# Hands-On Report — Week 1
**Submitted by:** Dewansh Khandelwal
**Deadline:** Next Friday
**Project:** DepthVR + Delsys — Multimodal Gesture Recognition (Mélange Research Lab)

---

## Overview

As assigned in the meeting, this report documents hands-on exploration of the two data modalities that form the core of the research pipeline:

1. **Computer Vision** — MediaPipe + OpenCV pipeline for hand gesture image processing
2. **EMG (Electromyography)** — NinaPro DB1 dataset preprocessing and analysis

---

## Modality 1 — Computer Vision

### Objective
Build a pipeline that:
- Takes raw hand gesture photographs as input
- Detects hand landmarks using MediaPipe
- Removes the background using OpenCV
- Computes Chamfer Distance to quantify shape similarity between gestures

### Dataset Used
Self-collected photos — 3 hand gestures photographed against a white background using a phone camera:
- 👍 Thumbs Up
- ✌️ Victory / Peace Sign
- 🖐️ Open Palm

Two separate sets of photos were taken and processed independently to test consistency.

### Tools & Libraries
| Tool | Version | Purpose |
|---|---|---|
| MediaPipe | 0.10.35 | Hand landmark detection (21 joints) |
| OpenCV (cv2) | 4.13.0 | Image processing, background removal |
| NumPy | — | Array math |
| SciPy | 1.17.1 | Chamfer Distance computation |

### Pipeline Steps

#### Step 1 — Hand Landmark Detection (MediaPipe)
MediaPipe's `HandLandmarker` model detects **21 joints** on the hand — wrist, knuckles, and fingertips — returning normalized (x, y) coordinates between 0 and 1. These are converted to pixel coordinates by multiplying with image dimensions.

The model used (`hand_landmarker.task`) is a pre-trained Google model (7.6 MB) loaded locally — no internet required at inference time.

#### Step 2 — Background Removal (OpenCV)
Using the 21 detected landmark points:
1. A **convex hull** is computed — the tightest polygon wrapping all joints
2. The hull is **expanded by 18%** outward from its centroid to avoid clipping fingertips
3. A **binary mask** is created (white = hand, black = background)
4. **Morphological dilation** fills any small gaps in the mask
5. The hand is pasted onto a **white background canvas**

**Fallback:** When MediaPipe fails to detect a hand (poor lighting, busy background), an **HSV skin-colour segmentation** is used as a backup.

#### Step 3 — Chamfer Distance
The binary mask's edge pixels are extracted using **Canny edge detection**, giving a set of boundary points representing the hand's outline shape.

Chamfer Distance between two shapes A and B is:
```
CD(A, B) = ( mean(min distance from each A point to B) +
             mean(min distance from each B point to A) ) / 2
```
**Lower = more similar shapes. Zero = identical shapes.**

### Results

#### Photo Set 1
| Pair | Chamfer Distance |
|---|---|
| Thumbs Up ↔ Open Palm | **45.18 px** (most similar) |
| Victory ↔ Open Palm | 58.01 px |
| Thumbs Up ↔ Victory | **73.35 px** (most different) |

#### Photo Set 2
| Pair | Chamfer Distance |
|---|---|
| Victory ↔ Open Palm | **33.16 px** (most similar) |
| Thumbs Up ↔ Victory | 53.43 px |
| Thumbs Up ↔ Open Palm | **66.95 px** (most different) |

### Key Observation
> The Chamfer Distance values **changed between the two photo sets** even though the same gestures were performed. This is because Chamfer Distance measures the actual pixel shape in that specific image — slight differences in wrist angle, finger spread, and photo distance all alter the edge points.

**This confirms why the real study collects 3 orientations per gesture** — a single photo does not capture the full variability of a gesture's shape.

### Output Files
```
HandsOn/output/
├── thumbs_up_landmarks.jpg     — MediaPipe skeleton overlay
├── thumbs_up_bg_removed.jpg    — Hand isolated on white background
├── thumbs_up_mask.jpg          — Binary hand mask
├── victory_landmarks.jpg
├── victory_bg_removed.jpg
├── victory_mask.jpg
├── open_palm_landmarks.jpg
├── open_palm_bg_removed.jpg
├── open_palm_mask.jpg
└── comparison_grid.jpg         — All gestures × all stages side-by-side
```

### Difficulty Faced
- **MediaPipe version conflict**: MediaPipe 0.10.35 (Python 3.14) removed the classic `mp.solutions` API. Had to rewrite using the new `mp.tasks.vision` Tasks API and download the model file manually.
- **Victory sign detection miss** (Photo Set 1): MediaPipe failed on the Victory gesture due to non-white background. The skin-colour fallback still produced a usable result, but with reduced accuracy.

---

## Modality 2 — EMG (Electromyography)

### Objective
Load and preprocess an open-source EMG gesture dataset, understand the raw signal format, and apply standard preprocessing steps to prepare data for downstream classification.

### Dataset Used
**NinaPro DB1** — Non-Invasive Adaptive Hand Prosthetics Database 1
- **Subject:** S1 (Subject 1 of 27)
- **File:** `S1_A1_E1.mat` (Exercise 1)
- **Source:** [ninapro.hevs.ch](http://ninapro.hevs.ch)

### Dataset Structure
| Field | Shape | Description |
|---|---|---|
| `emg` | (101,014 × 10) | Raw EMG from 10 forearm muscle sensors |
| `restimulus` | (101,014 × 1) | Gesture label at each time step (0=rest, 1–12=gesture) |
| `rerepetition` | (101,014 × 1) | Repetition number (1–10 per gesture) |
| `glove` | (101,014 × 22) | Finger angle data from a data glove (not used) |

**Recording:** 16.8 minutes total | **Sampling rate:** 100 Hz | **12 hand gestures**

### Gestures in Exercise 1
| ID | Gesture |
|---|---|
| G1 | Small Finger Flexion |
| G2 | Ring Finger Flexion |
| G3 | Middle Finger Flexion |
| G4 | Index Finger Flexion |
| G5 | Thumb Flexion |
| G6 | Thumb + Index (Pinch) |
| G7 | Thumb + Middle |
| G8 | Thumb + Ring |
| G9 | Thumb + Small |
| G10 | All Fingers Extended |
| G11 | Hook Grip |
| G12 | Lateral Grip |

### Preprocessing Pipeline

#### Step 1 — Bandpass Filter (20–45 Hz)
A 4th-order Butterworth bandpass filter is applied to isolate the frequency range containing genuine muscle signals.

| Frequency Range | Content | Action |
|---|---|---|
| Below 20 Hz | Motion artifacts (arm movement) | ❌ Removed |
| 20–45 Hz | Actual muscle electrical activity | ✅ Kept |
| Above 45 Hz | Electrical noise | ❌ Removed |

> Note: NinaPro DB1 samples at 100 Hz → Nyquist limit = 50 Hz. The highcut was adjusted to 45 Hz automatically.

#### Step 2 — Rectification
The filtered signal oscillates between positive and negative values. Taking the absolute value (`|signal|`) flips all negatives to positive, enabling meaningful amplitude analysis.

#### Step 3 — RMS Envelope (200ms window)
A sliding Root Mean Square window of 200ms (20 samples at 100 Hz) smooths the rectified signal into a clean **muscle activation envelope** — showing how strongly each muscle fires over time, without the high-frequency oscillation.

### Results

#### Segmentation (Repetition 1 only)
| Gesture | Samples | Duration |
|---|---|---|
| G1: Small Finger Flexion | 305 | 3.0s |
| G2: Ring Finger Flexion | 193 | 1.9s |
| G3: Middle Finger Flexion | 524 | 5.2s |
| G4: Index Finger Flexion | 348 | 3.5s |
| G5: Thumb Flexion | 452 | 4.5s |
| G6: Thumb + Index (Pinch) | 293 | 2.9s |
| G7: Thumb + Middle | 246 | 2.5s |
| G8: Thumb + Ring | 211 | 2.1s |
| G9: Thumb + Small | 197 | 2.0s |
| G10: All Fingers Extended | 458 | 4.6s |
| G11: Hook Grip | 186 | 1.9s |
| G12: Lateral Grip | 341 | 3.4s |

#### Activation Heatmap — Key Findings
The mean RMS amplitude per (gesture, channel) pair reveals distinct **muscle activation fingerprints**:

- **Channel 7** shows the highest activation for G4 (Index Finger), G5 (Thumb), and G6 (Pinch) — anatomically consistent with the extensor/flexor muscles controlling these fingers
- **Channel 1** spikes specifically for G7 (Thumb + Middle) — a unique pattern not seen in other gestures
- **Channel 2** activates broadly across many gestures — consistent with a general forearm flexor
- **Channels 4, 5, 6** show low activation throughout Exercise 1 — these muscles are more relevant for the wrist/forearm gestures in Exercise 2 and 3

> **Core insight:** Each gesture produces a unique combination of muscle activations across the 10 channels — this is the fundamental basis of EMG gesture recognition. A classifier can learn to map these "activation fingerprints" to gesture labels.

### Output Files
```
HandsOn/output/emg/
├── 1_raw_emg.png               — All 10 channels, raw signal (first 10 seconds)
├── 2_preprocessing_stages.png  — Raw → Bandpass → Rectified → RMS Envelope (Ch 1)
├── 3_per_gesture_envelopes.png — EMG envelope for each gesture, all 10 channels
└── 4_activation_heatmap.png    — Mean activation matrix: gestures × channels
```

### Difficulty Faced
- No major technical issues. NinaPro DB1 loaded cleanly via `scipy.io.loadmat()`.
- The `stimulus` vs `restimulus` distinction required attention — `restimulus` provides cleaner, time-corrected labels and should always be used over `stimulus`.

---

## Summary of Both Modalities

| | Computer Vision | EMG |
|---|---|---|
| **Data source** | Self-captured photos (phone) | NinaPro DB1, Subject 1 |
| **Key tool** | MediaPipe + OpenCV | SciPy signal processing |
| **Core output** | Background-removed gesture images + Chamfer Distance | Preprocessed RMS envelope + activation heatmap |
| **Main finding** | Chamfer Distance is sensitive to gesture orientation/scale | Each gesture has a unique muscle activation fingerprint |
| **Relevance to project** | Visual pipeline for the Meta Quest image data | Signal pipeline for the Delsys EMG sensor data |

---

## Next Steps (as discussed)

1. **Feature extraction from EMG** — compute time-domain features (MAV, ZCR, Waveform Length) from each gesture window
2. **Train a classifier** — SVM or Random Forest on extracted features, evaluate on NinaPro DB1
3. **Extend CV pipeline** — test on actual Meta Quest depth images from the VR system
4. **Data collection** — 5 participants, 10 gestures × 3 orientations using the DepthVR system

---

*All code is available in:*
- [`HandsOn/gesture_pipeline.ipynb`](HandsOn/gesture_pipeline.ipynb) — Computer Vision pipeline (Jupyter Notebook)
- [`HandsOn/emg_pipeline.ipynb`](HandsOn/emg_pipeline.ipynb) — EMG pipeline (Jupyter Notebook)
