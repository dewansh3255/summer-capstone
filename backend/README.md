# DepthVR Backend Pipeline

Python backend for the DepthVR gesture recognition system.
This server receives live hand joint data from a Meta Quest headset via WebSocket,
processes it through a four-stage pipeline, and classifies hand gestures in real time.

Built as part of a summer research project at IIIT Delhi (Mélange Research Lab),
continuing the work from the C4GT report *"A Robust Framework for Occluded Hand
Gesture Recognition using 3D Landmark Detection and Point Cloud Analysis"* and
evaluated using the out-of-distribution benchmarking methodology from the
*EMGBench* paper (Yang et al., CMU 2024).

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [System Architecture](#2-system-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites and Setup](#4-prerequisites-and-setup)
5. [Pipeline — How It Works](#5-pipeline--how-it-works)
   - [Step 1: WebSocket Server and Binary Deserializer](#step-1-websocket-server-and-binary-deserializer)
   - [Step 2: Point Cloud Generation](#step-2-point-cloud-generation)
   - [Step 3: Chamfer Distance and Session Recorder](#step-3-chamfer-distance-and-session-recorder)
   - [Step 4: Dataset, Training, and Classification](#step-4-dataset-training-and-classification)
6. [Complete Workflow — From Quest to Results](#6-complete-workflow--from-quest-to-results)
7. [Server Modes and CLI Reference](#7-server-modes-and-cli-reference)
8. [Offline Scripts](#8-offline-scripts)
9. [Gesture Vocabulary](#9-gesture-vocabulary)
10. [Binary Frame Protocol](#10-binary-frame-protocol)
11. [Design Decisions and Research Notes](#11-design-decisions-and-research-notes)
12. [Running Tests](#12-running-tests)
13. [Future Work](#13-future-work)

---

## 1. Project Context

This backend is one component of a larger system:

```
Meta Quest (Unity DepthVR app)
    ↓ WiFi / WebSocket
Python Backend  ← you are here
    ↓
Gesture classification result
```

The Unity frontend (`VR IP/frontend/`) runs on the Quest, presents gesture
demonstration videos to the user, captures 25 hand joint positions per frame
via XR hand tracking, and streams them to this server over WebSocket.

The backend's job is to:
1. Receive and decode those binary frames
2. Convert joint positions into normalised 3D point clouds
3. Save recordings for offline dataset building
4. Train and run a gesture classifier

The Delsys EMG sensor integration (`VR IP/VR-Delsys-Integration-main/`) is a
parallel hardware component managed separately — it can be synchronised with
this pipeline's timestamp data for future multimodal fusion work.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Meta Quest (Unity DepthVR)                                 │
│  HandLandmarkLogger.cs                                      │
│    → 25 joint positions (x,y,z,depth) per frame            │
│    → Binary serialised via FrameSerializer.cs               │
│    → Sent over WebSocket (ws://<PC_IP>:9002)                │
└──────────────────────────┬──────────────────────────────────┘
                           │ WiFi
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  server.py   (asyncio WebSocket server)                     │
│                                                             │
│  Step 1 ✅  protocol/frame_deserializer.py                  │
│             Binary bytes → Frame → HandData → JointData     │
│                                                             │
│  Step 2 ✅  pipeline/point_cloud.py                         │
│             HandData → normalised PointCloud (N,3) float32  │
│             centre by mean + scale to unit sphere           │
│                                                             │
│  Step 3 ✅  pipeline/recorder.py                            │
│             PointCloud → .npy files on disk                 │
│             (hardware-independence layer)                   │
│                                                             │
│  Step 4 ✅  pipeline/classifier.py                          │
│             PointCloud → GestureClassifier.predict_label()  │
│             → "Thumbs_Up" / "Spock" / etc.                  │
└─────────────────────────────────────────────────────────────┘
                           │ offline
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Offline analysis (no Quest needed)                         │
│                                                             │
│  pipeline/chamfer.py                                        │
│    chamfer_distance(), pairwise_chamfer_matrix()            │
│    find_optimal_threshold()  (C4GT §4.5 elbow method)       │
│                                                             │
│  pipeline/dataset.py                                        │
│    load_dataset(), tsts_split(), loso_splits()              │
│    (EMGBench TSTS + LOSO-CV evaluation methodology)         │
│                                                             │
│  train.py    → train model, save .npz                       │
│  evaluate.py → per-gesture accuracy, confusion matrix,      │
│                LOSO-CV table, threshold analysis            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
backend/
├── server.py                   WebSocket server — entry point for live use
├── train.py                    Offline training script
├── evaluate.py                 Offline evaluation / reporting script
├── requirements.txt            Python dependencies
├── PROTOCOL.md                 Binary frame format specification
├── README.md                   This file
│
├── protocol/
│   ├── __init__.py
│   └── frame_deserializer.py   Step 1: binary packet → Python objects
│
├── pipeline/
│   ├── __init__.py
│   ├── point_cloud.py          Step 2: HandData → normalised PointCloud
│   ├── recorder.py             Step 3: save/load PointCloud sessions to disk
│   ├── chamfer.py              Step 3: Chamfer Distance + threshold analysis
│   ├── dataset.py              Step 4: load recordings into GestureDataset
│   └── classifier.py           Step 4: MLP gesture classifier
│
├── tests/
│   ├── test_deserializer.py    26 tests — binary protocol correctness
│   ├── test_point_cloud.py     25 tests — normalisation, occlusion handling
│   ├── test_recorder.py        27 tests — save/load, metadata
│   ├── test_chamfer.py         37 tests — distance, thresholding, pairs
│   ├── test_dataset.py         33 tests — vocabulary, splits
│   └── test_classifier.py      31 tests — training, inference, persistence
│
├── recordings/                 Created at runtime — recorded gesture sessions
│   └── <timestamp>_<gesture>/
│       ├── session.json        Metadata (gesture, fps, duration)
│       └── frames/
│           ├── 000000.npy      One frame = one (N,3) float32 point cloud
│           └── ...
│
└── models/                     Created at runtime — saved classifiers
    └── gesture_clf.npz
```

---

## 4. Prerequisites and Setup

**Python version:** 3.10 or later (3.14 tested)

**Install dependencies:**

```bash
cd backend
pip3 install -r requirements.txt
```

Dependencies:

| Package | Version | Purpose |
|---|---|---|
| websockets | 12.0 | Async WebSocket server |
| numpy | 1.26.4 | All numerical operations |
| scipy | 1.13.0 | Chamfer Distance (cdist) |
| open3d | 0.18.0 | Point cloud utilities (future use) |
| pandas | 2.2.2 | Tabular analysis of recordings |
| matplotlib | 3.9.0 | Threshold curve plotting |
| seaborn | 0.13.2 | Confusion matrix visualisation |
| tqdm | 4.66.4 | Progress bars for long computations |

**Network requirement for live use:**
Both the Quest and the laptop running this server must be on the **same WiFi network**.
Find your laptop's LAN IP (e.g. `192.168.1.100`) and set it in Unity:

```
Unity Inspector → WebSocketSender → serverUrl → ws://192.168.1.100:9002
```

---

## 5. Pipeline — How It Works

### Step 1: WebSocket Server and Binary Deserializer

**Files:** `server.py`, `protocol/frame_deserializer.py`

The Quest sends one binary packet per Unity `LateUpdate()` frame (~72 fps).
`frame_deserializer.py` parses the exact byte layout produced by C#'s
`BinaryWriter` in `FrameSerializer.cs`.

**Packet structure (32-byte header + blobs):**

```
Offset  Type    Field
0       int32   FrameIndex
4       int64   Timestamp (Stopwatch ticks)
12      uint16  ColorWidth, ColorHeight
16      uint16  DepthWidth, DepthHeight
20      int32   colorLen
24      int32   depthLen
28      int32   handsLen
32      bytes   hands blob  → left hand then right hand
32+N    bytes   color blob  (currently empty)
32+N+M  bytes   depth blob  (currently empty)
```

**Hand blob structure (850 bytes = 2 hands × 425 bytes each):**

For each of 25 joints:
```
1 byte  bool    valid (1 = tracked)
4 bytes float32 x  (camera-local space, metres)
4 bytes float32 y
4 bytes float32 z
4 bytes float32 depth (Euclidean distance from camera)
```

The parsed result is a `Frame` object containing two `HandData` objects
(left and right), each holding 25 `JointData` entries.

Full spec: see `PROTOCOL.md`.

---

### Step 2: Point Cloud Generation

**File:** `pipeline/point_cloud.py`

Converts a `HandData` object (25 joints) into a normalised `PointCloud`.

**Design choice — Option A (sparse skeleton):**
The 25 XR joint positions are used directly as the point cloud.
Each joint is a real measurement — no synthetic interpolated points.

**Option B (dense interpolation) is documented but not implemented:**
If future experiments show 25 points are insufficient for discrimination,
bone-segment interpolation can be added. See `point_cloud.py` docstring
for full implementation notes and tradeoffs.

**Normalisation pipeline:**

```
raw (N,3) positions in metres
    │
    ▼  subtract mean
centred cloud
    │
    ▼  divide by max L2 norm
normalised cloud — fits inside unit sphere, centred at origin
```

This makes Chamfer Distance invariant to how far the hand is from the
headset camera. The raw `centroid` and `scale` are preserved on the
`PointCloud` object so the transform is invertible.

Frames with fewer than 10 valid joints (heavily occluded hand) return
`None` — the caller skips these frames cleanly.

---

### Step 3: Chamfer Distance and Session Recorder

**Files:** `pipeline/chamfer.py`, `pipeline/recorder.py`

**Chamfer Distance:**

The symmetric Chamfer Distance between two point clouds X and Y is:

```
CD(X,Y) = (1/|X|) Σ_{x∈X} min_{y∈Y} ‖x−y‖²
         + (1/|Y|) Σ_{y∈Y} min_{x∈X} ‖y−x‖²
```

Implemented via `scipy.spatial.distance.cdist` — one vectorised M×N
matrix call, no Python loops over individual points.

Key functions:
- `chamfer_distance(a, b)` — distance between two (N,3) arrays
- `pairwise_chamfer_matrix(clouds)` — full K×K distance matrix
- `extract_pairs(clouds, labels)` — all pairwise distances with same/different labels
- `find_optimal_threshold(distances, labels)` — F1-maximising threshold sweep
  (C4GT report Section 4.5 elbow method, 500 candidate thresholds)

**Session Recorder:**

The `Recorder` class saves each `PointCloud` frame as an individual `.npy`
file under `recordings/<timestamp>_<gesture>/frames/`, plus a `session.json`
with metadata. This is the **hardware-independence layer** — once recorded,
all further development and testing is entirely offline.

```
recordings/
└── 2026-06-18_143022_Thumbs_Up/
    ├── session.json    {"gesture_label": "Thumbs_Up", "frame_count": 432, "avg_fps": 72.1}
    └── frames/
        ├── 000000.npy  shape: (25, 3) float32
        ├── 000001.npy
        └── ...
```

---

### Step 4: Dataset, Training, and Classification

**Files:** `pipeline/dataset.py`, `pipeline/classifier.py`

**Dataset:**

`load_dataset(recordings_dir)` scans all recorded sessions and returns a
flat `GestureDataset` with:
- `features`: (N, 75) float32 — each frame's 25 joints flattened to a vector
- `labels`: (N,) int32 — integer class index
- `label_names`: list of gesture label strings per frame
- `session_ids`: which recording session each frame came from (for splits)

Two split strategies (both from the EMGBench paper):

**TSTS (Train-Test Split for Time Series):**
Splits each session's frames at a ratio along the time axis.
Test data is always temporally after training data — prevents leakage.

**LOSO-CV (Leave-One-Session-Out Cross-Validation):**
For each session, train on all other sessions and test on the held-out one.
This is the standard OOD evaluation for measuring generalisation to new
recording conditions.

**Classifier:**

A two-layer MLP (`GestureClassifier`) implemented in pure NumPy:

```
Input (75) → FC(128) → ReLU → Dropout(0.3)
           → FC(64)  → ReLU → Dropout(0.3)
           → FC(10)  → Softmax
```

Architecture matches C4GT report Section 4.7 `BroadGestureClassifier`,
with dropout added for regularisation on small datasets.

This is the **broad pre-clustering stage** from the C4GT report that was
left unimplemented due to time constraints. It reduces Chamfer Distance
computation cost from O(K) to O(K/C) per query (C = number of classes = 10).

Training uses mini-batch SGD with cross-entropy loss and He initialisation.
Model is saved as a `.npz` file loadable with `GestureClassifier.load()`.

---

## 6. Complete Workflow — From Quest to Results

### Phase 1 — Data Collection (requires Quest)

Record one session per gesture. Each recording should be 5–10 seconds
(~360–720 frames at 72 fps).

```bash
# In terminal 1: start server in record mode
python3 server.py --record --gesture Thumbs_Up --recordings-dir recordings/

# Put on Quest, perform Thumbs_Up gesture
# Press Ctrl-C to stop — session saved automatically

# Repeat for all 10 gestures:
python3 server.py --record --gesture ASL_L       --recordings-dir recordings/
python3 server.py --record --gesture ASL_Y       --recordings-dir recordings/
python3 server.py --record --gesture Five        --recordings-dir recordings/
python3 server.py --record --gesture Four        --recordings-dir recordings/
python3 server.py --record --gesture One         --recordings-dir recordings/
python3 server.py --record --gesture Spiderman   --recordings-dir recordings/
python3 server.py --record --gesture Spock       --recordings-dir recordings/
python3 server.py --record --gesture Three       --recordings-dir recordings/
python3 server.py --record --gesture Two         --recordings-dir recordings/
```

### Phase 2 — Offline Training (no Quest needed)

```bash
python3 train.py \
    --recordings-dir recordings/ \
    --model-out models/gesture_clf.npz \
    --epochs 150 \
    --lr 1e-3 \
    --loso
```

Example output:
```
[1/4] Loading dataset from 'recordings/' ...
  Loaded 4320 frames
  Class distribution:
    ASL_L          432 frames
    ASL_Y          438 frames
    ...

[2/4] Splitting dataset (train=80%, test=20%) ...
  Train: 3456 frames   Test: 864 frames

[3/4] Training MLP classifier (150 epochs, lr=0.001) ...
  Epoch    1/150  loss=2.3026  acc=0.102
  Epoch   10/150  loss=1.8432  acc=0.612
  ...
  Epoch  150/150  loss=0.2103  acc=0.943

  Final train accuracy : 0.943 (94.3%)
  Final test  accuracy : 0.891 (89.1%)

[4/4] Saving model to 'models/gesture_clf.npz' ...

[LOSO-CV] Leave-one-session-out cross-validation ...
  Held-out: 2026-06-18_143022_Thumbs_Up   acc=0.882
  ...
  LOSO-CV mean accuracy: 0.873 (87.3%)
```

### Phase 3 — Evaluation Report (no Quest needed)

```bash
python3 evaluate.py \
    --model models/gesture_clf.npz \
    --recordings-dir recordings/ \
    --loso \
    --threshold
```

### Phase 4 — Live Inference (requires Quest)

```bash
python3 server.py \
    --model models/gesture_clf.npz \
    --verbose
```

Server logs the predicted gesture label for each incoming frame:
```
14:32:05  DEBUG    Frame  1042 → predicted: Thumbs_Up
14:32:05  DEBUG    Frame  1043 → predicted: Thumbs_Up
```

---

## 7. Server Modes and CLI Reference

```
python3 server.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `9002` | WebSocket port (must match Unity `serverUrl`) |
| `--verbose` | off | Log every frame's joint counts and predictions |
| `--record` | off | Enable recording mode — saves frames to disk |
| `--gesture` | — | Gesture label for recording (required if `--record`) |
| `--recordings-dir` | `recordings/` | Root directory for saved sessions |
| `--model` | — | Path to `.npz` classifier for live inference |

**Combinations:**

```bash
# Just receive and parse frames (no save, no classify):
python3 server.py

# Record a session:
python3 server.py --record --gesture Spock

# Live classification with a trained model:
python3 server.py --model models/gesture_clf.npz --verbose

# Record AND classify simultaneously:
python3 server.py --record --gesture Spock --model models/gesture_clf.npz
```

---

## 8. Offline Scripts

### train.py

Trains the MLP classifier from recorded sessions.

```
python3 train.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--recordings-dir` | `recordings/` | Source recordings directory |
| `--model-out` | `models/gesture_clf.npz` | Output model path |
| `--epochs` | `150` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch-size` | `32` | Mini-batch size |
| `--train-ratio` | `0.8` | Fraction of each session used for training |
| `--loso` | off | Also run leave-one-session-out CV |
| `--threshold` | off | Also compute optimal Chamfer Distance threshold |

### evaluate.py

Produces a full evaluation report from a saved model.

```
python3 evaluate.py --model <path> [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to `.npz` model file |
| `--recordings-dir` | `recordings/` | Source recordings directory |
| `--train-ratio` | `0.8` | Same ratio used during training |
| `--loso` | off | Run LOSO-CV evaluation |
| `--threshold` | off | Run Chamfer Distance threshold analysis |

---

## 9. Gesture Vocabulary

The 10 gestures match the video files in `VR IP/frontend/Assets/Gesture Videos/`.
Each gesture was recorded at three orientations (0°, 45°, 90°) in DepthVR.
The classifier treats all orientations of the same gesture as one class —
orientation robustness is built into the unit-sphere normalisation in Step 2.

| Index | Label | Description |
|---|---|---|
| 0 | `ASL_L` | American Sign Language letter L |
| 1 | `ASL_Y` | American Sign Language letter Y |
| 2 | `Five` | Open hand, all five fingers extended |
| 3 | `Four` | Four fingers extended, thumb tucked |
| 4 | `One` | Index finger pointing up |
| 5 | `Spiderman` | Web-shooting pose (index + pinky extended) |
| 6 | `Spock` | Vulcan salute (split between middle and ring) |
| 7 | `Three` | Three fingers extended |
| 8 | `Thumbs_Up` | Thumbs up, fist closed |
| 9 | `Two` | Index and middle fingers extended |

**Important:** gesture label strings must match exactly when recording.
`Thumbs_Up` not `ThumbsUp`. Unrecognised labels are skipped by `load_dataset()`.

---

## 10. Binary Frame Protocol

Full specification: see `PROTOCOL.md`.

Quick summary:
- All values little-endian (C# `BinaryWriter` default)
- 32-byte header: `int32 frameIndex`, `int64 timestamp`, 4× `uint16` dimensions, 3× `int32` blob lengths
- Hands blob: 850 bytes = 2 hands × 25 joints × 17 bytes/joint
- Per joint: `uint8 valid` + `float32 x, y, z, depth`
- Color and depth blobs currently empty (length 0)

---

## 11. Design Decisions and Research Notes

### Why pure NumPy (no PyTorch)?

The classifier is implemented entirely in NumPy with no PyTorch dependency.
For a 75-dim input and 10 output classes, a forward pass through the MLP
takes microseconds on CPU. Adding PyTorch would increase the deployment
footprint substantially for no performance benefit at this scale.
PyTorch3D is noted in `chamfer.py` as a future option if the dataset
grows to millions of pairs requiring GPU acceleration.

### Why 25-point sparse point clouds (Option A)?

The C4GT report used dense point clouds (~10,000–20,000 points) from an
RGB-D depth camera. Our source is the Quest's skeletal hand tracking,
which produces 25 clean, labelled joint positions — already a higher-level
representation than raw depth pixels.

Chamfer Distance is valid on any point set size. The relative ordering
(same-gesture pairs have smaller CD than different-gesture pairs) is
preserved even with 25 points, which is all that classification requires.

Option B (interpolating points along bone segments) is documented in
`pipeline/point_cloud.py` and can be enabled if 25-point clouds prove
insufficiently discriminative. Key tradeoff: bone interpolation introduces
a free parameter (points per bone segment) with no principled default,
and biases the centroid toward longer bones.

### TSTS vs LOSO evaluation

Following EMGBench (Section 4.3):
- **TSTS** (temporal split): fast sanity check. Tests whether the model
  generalises across time within the same recording session. Used as the
  primary metric during `train.py`.
- **LOSO-CV**: the real OOD test. Tests whether the model generalises to
  recording conditions it has never seen. Use this for research reporting.

### Chamfer Distance threshold vs classifier

The pipeline has two classification mechanisms:
1. **MLP classifier** (Step 4): fast, ~microseconds per frame, classifies
   into one of 10 gesture classes directly.
2. **Chamfer Distance threshold** (Step 3): geometric, alignment-free,
   uses the elbow-point threshold from `find_optimal_threshold()`.

The intended production use is the hybrid approach from C4GT Section 4.7:
- Classifier predicts coarse gesture class (10× fewer Chamfer comparisons)
- Chamfer Distance verifies within the predicted class

The `find_optimal_threshold()` and `evaluate.py --threshold` outputs give
the threshold value for this verification step.

---

## 12. Running Tests

```bash
cd backend
python3 -m pytest tests/ -v
```

179 tests across 6 test files, all passing. No hardware required.

| File | Tests | What is covered |
|---|---|---|
| `test_deserializer.py` | 26 | Binary protocol: header parsing, joint coordinates, edge cases |
| `test_point_cloud.py` | 25 | Normalisation, occlusion handling, degenerate inputs |
| `test_recorder.py` | 27 | Save/load sessions, metadata, ordering, error states |
| `test_chamfer.py` | 37 | Distance correctness, symmetry, threshold sweep, pair extraction |
| `test_dataset.py` | 33 | Vocabulary, feature extraction, TSTS and LOSO splits |
| `test_classifier.py` | 31 | Forward pass, training convergence, save/load, LOSO integration |

---

## 13. Future Work

The following items are documented for the next development phase:

**Multimodal fusion with Delsys EMG:**
The `VR IP/VR-Delsys-Integration-main/Python/` directory contains a working
Python SDK for the Delsys Trigno EMG sensor. Synchronising EMG muscle
activation signals with the Quest hand joint timestamps would enable a
fusion architecture: vision-based joints for clear-view scenarios + EMG
for occluded hand states.

**PointNet / DGCNN as classifier backbone:**
The current MLP treats the 25 joints as an unordered flat vector, losing
the graph structure of the hand skeleton. Replacing the MLP with PointNet
(Qi et al. 2017) or DGCNN would let the model learn spatial relationships
between joints explicitly. This is the upgrade path noted in C4GT §4.7.

**Few-shot fine-tuning (EMGBench adaptation methodology):**
EMGBench showed that fine-tuning a pretrained model with just 8–30 seconds
of data from a new user achieves near-full accuracy. The `GestureClassifier`
supports this through its standard `fit()` interface — implement a
`finetune(small_dataset, epochs=50)` wrapper that uses a reduced learning
rate to adapt the pretrained model without catastrophic forgetting.

**Option B: bone interpolation for denser point clouds:**
If LOSO-CV results show poor gesture separability, implement the bone-segment
interpolation described in `pipeline/point_cloud.py`. Suggested starting
point: 5 points per bone × 20 bone segments ≈ 125 total points per cloud.

**Real-time Chamfer verification layer:**
Wire the `find_optimal_threshold()` result from `train.py --threshold` into
the live `server.py` pipeline as a second-pass verification step after the
MLP prediction. This implements the full C4GT hybrid architecture.
