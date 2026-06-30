# DepthVR — Occlusion-Robust Hand Gesture Recognition in VR

M.Tech summer research project (IIIT Delhi, Mélange Research Lab).

This project builds a **hand-gesture recognition system for standalone VR
headsets** (Meta Quest). A Unity app on the headset captures 3D hand-joint data
and streams it to a Python backend that converts each hand into a 3D point
cloud, compares gestures using **Chamfer Distance**, and classifies them with a
lightweight neural network. It continues — and completes — the approach from the
*C4GT IIIT Delhi* report on occlusion-robust gesture recognition, and is
evaluated using the out-of-distribution methodology from the *EMGBench* paper.

> A separate, parallel sub-task on **EMG muscle-fatigue analysis** lives in
> [`analysis/`](analysis/) and has its own README. This document is about the
> main gesture-recognition project.

---

## Table of contents

1. [The problem & the idea](#1-the-problem--the-idea)
2. [How the system works](#2-how-the-system-works)
3. [Repository layout](#3-repository-layout)
4. [Quick start & testing](#4-quick-start--testing)
5. [End-to-end workflow](#5-end-to-end-workflow)
6. [Research context](#6-research-context)
7. [Related sub-task: EMG analysis](#7-related-sub-task-emg-analysis)
8. [What is not in the repo](#8-what-is-not-in-the-repo)

---

## 1. The problem & the idea

Camera-based gesture recognition degrades badly when the hand is **partially
occluded** or seen from an **unusual angle** — the exact conditions that are
common in real VR use. The C4GT report proposed addressing this by representing
a hand as a **3D point cloud** and comparing gestures with **Chamfer Distance**,
an alignment-free shape-similarity metric that is robust to occlusion and
viewpoint changes.

This project takes that idea into VR:

- The **Meta Quest** already provides clean 3D hand-tracking (25 joints per hand),
  so we get a high-quality skeletal point cloud directly — no fragile RGB-D
  segmentation needed.
- We stream those joints to a backend, normalise them into a point cloud,
  measure gesture similarity with Chamfer Distance, and classify the gesture
  with a small neural network.
- We complete the **neural pre-classifier** that the C4GT report proposed but
  left unbuilt, and evaluate generalisation with EMGBench-style cross-validation.

The system recognises **10 hand gestures**: ASL_L, ASL_Y, Five, Four, One,
Spiderman, Spock, Three, Thumbs_Up, Two.

---

## 2. How the system works

```
┌──────────────────────────────────────────────────────────────┐
│  Meta Quest headset — Unity DepthVR app   (VR IP/frontend/)   │
│  • shows the user a gesture to perform                        │
│  • captures 25 hand joints/frame via XR hand tracking         │
│  • streams them over WebSocket (Wi-Fi)                        │
└───────────────────────────┬──────────────────────────────────┘
                            │  binary frames, ~72 fps
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Python backend   (backend/)                                  │
│                                                              │
│   Step 1  decode binary frames        protocol/               │
│   Step 2  joints → normalised cloud   pipeline/point_cloud.py │
│   Step 3  Chamfer Distance + record   pipeline/chamfer.py     │
│   Step 4  dataset + MLP classifier    pipeline/classifier.py  │
│                                                              │
│   → predicted gesture: "Thumbs_Up"                            │
└──────────────────────────────────────────────────────────────┘
```

The two halves:

- **`VR IP/frontend/`** — the Unity 6000.3.2f1 DepthVR application that runs on
  the Quest, presents gesture prompts, and streams hand data. Build/run
  instructions: [`VR IP/readme_depthvr.md`](VR%20IP/readme_depthvr.md).
- **`backend/`** — the Python pipeline that receives, processes, trains on, and
  classifies that data. Full details: [`backend/README.md`](backend/README.md).

A key design property: once gesture data has been recorded once, **all
development, training, and testing run fully offline** — no headset needed.

---

## 3. Repository layout

```
Summer/
├── README.md                  ← you are here (main project overview)
│
├── backend/                   The gesture-recognition Python pipeline
│   ├── server.py              WebSocket server (entry point)
│   ├── train.py               offline classifier training
│   ├── evaluate.py            offline evaluation (accuracy, LOSO-CV)
│   ├── protocol/              Step 1: binary frame decoder
│   ├── pipeline/              Steps 2-4: point cloud, Chamfer, dataset, MLP
│   ├── tests/                 179 tests
│   └── README.md              ← detailed backend documentation
│
├── VR IP/                     The VR frontend + EMG hardware integration
│   ├── frontend/              Unity DepthVR app for Meta Quest
│   ├── VR-Delsys-Integration-main/   Delsys EMG sensor SDK + examples
│   └── readme_depthvr.md      Quest build/run guide
│
├── analysis/                  PARALLEL SUB-TASK: EMG fatigue analysis
│   └── README.md              ← its own documentation (separate from this)
│
├── HandsOn/                   exploratory Jupyter notebooks
├── Reports/                   weekly progress reports
└── Meeting Records/           meeting notes
```

---

## 4. Quick start & testing

Everything below runs **without a VR headset** — the test suite uses synthetic
data, so the whole backend can be verified on any machine.

**Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

**Run the test suite (179 tests):**
```bash
python3 -m pytest tests/ -v
```
Expected result: **179 passed**. These cover binary-frame decoding, point-cloud
normalisation, Chamfer Distance, the session recorder, dataset splits
(LOSO-CV / time-series), and the MLP classifier (training, inference, save/load).

That single command is enough to confirm the entire backend works.

---

## 5. End-to-end workflow

The full lifecycle (detailed in [`backend/README.md`](backend/README.md)):

**1. Record gestures** — *needs the Quest running the DepthVR app:*
```bash
cd backend
python3 server.py --record --gesture Thumbs_Up --recordings-dir recordings/
# perform the gesture for ~5-10 s, Ctrl-C to stop. Repeat per gesture.
```

**2. Train the classifier** — *offline, no headset:*
```bash
python3 train.py --recordings-dir recordings/ --model-out models/clf.npz --loso
```

**3. Evaluate** — *offline:*
```bash
python3 evaluate.py --model models/clf.npz --recordings-dir recordings/ --loso
```

**4. Live recognition** — *needs the Quest:*
```bash
python3 server.py --model models/clf.npz --verbose
```

Set the backend's LAN IP in Unity (`WebSocketSender → serverUrl →
ws://<your-ip>:9002`); both devices must share a Wi-Fi network.

---

## 6. Research context

- **C4GT IIIT Delhi report** — *"A Robust Framework for Occluded Hand Gesture
  Recognition using 3D Landmark Detection and Point Cloud Analysis."* The source
  of the point-cloud + Chamfer Distance approach. This project adapts it to VR
  hand-tracking and implements its proposed neural pre-classifier.
- **EMGBench (Yang et al., CMU 2024)** — provides the out-of-distribution
  evaluation methodology (Leave-One-Session-Out cross-validation and
  train-test-split-for-time-series) used to measure how well the classifier
  generalises to new recording sessions.

Design decisions (pure-NumPy classifier, sparse 25-point clouds, the
Chamfer-vs-classifier split) are documented in
[`backend/README.md` §11](backend/README.md).

---

## 7. Related sub-task: EMG analysis

In parallel with the gesture project, the lab provided surface-EMG (sEMG)
recordings from a Delsys Trigno system for **muscle-activation and fatigue
analysis**. That work is self-contained in [`analysis/`](analysis/) with its own
README, pipeline, and 106 tests. It shares the repo (and the long-term goal of
multimodal sensing) but is independent of the gesture backend.

See [`analysis/README.md`](analysis/README.md) for what it does, the science
behind it, and how to test it.

---

## 8. What is not in the repo

To keep the repository lean and license-clean, the following are gitignored:

- **Research papers** (`*.pdf`) — copyrighted.
- **Large Unity binaries** — Unity's `Library/` build cache, gesture-demo
  videos, and the Delsys SDK DLLs. The Unity **source and scripts are
  committed**.
- **Raw EMG recordings** (`analysis/data/*.csv`) — large; placed locally.
- **Generated artifacts** — trained models, recordings, plots.

Everything required to understand the code and run the test suites is in the
repository.

---

### One-command verification for a reviewer

```bash
cd backend && pip install -r requirements.txt && python3 -m pytest tests/ -q
```
Expected: **179 passed**. For the EMG sub-task, see
[`analysis/README.md`](analysis/README.md).
