# Summer Research Project — Gesture Recognition & EMG Fatigue Analysis

M.Tech summer research project (IIIT Delhi, Mélange Research Lab) on
**hand-gesture recognition** and **muscle-fatigue analysis** from wearable
sensors. This repository contains two complete, independently testable software
systems plus the VR data-collection frontend.

> **For the reviewer:** every component here can be tested **without any
> special hardware and without an API key**. The test suites use synthetic
> signals and a built-in offline interpreter. Jump to
> [How to test everything](#how-to-test-everything).

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Background & motivation](#2-background--motivation)
3. [Repository structure](#3-repository-structure)
4. [The two systems at a glance](#4-the-two-systems-at-a-glance)
5. [Prerequisites](#5-prerequisites)
6. [How to test everything](#6-how-to-test-everything)
7. [System 1 — Gesture recognition backend](#7-system-1--gesture-recognition-backend)
8. [System 2 — EMG fatigue analysis](#8-system-2--emg-fatigue-analysis)
9. [What needs hardware vs what does not](#9-what-needs-hardware-vs-what-does-not)
10. [What is intentionally not in the repo](#10-what-is-intentionally-not-in-the-repo)

---

## 1. What this project is

The project has two threads that share a common goal — understanding hand and
muscle activity from wearable sensors:

- **Gesture recognition (`backend/` + `VR IP/`)** — a Meta Quest VR app streams
  3D hand-joint data over WebSocket to a Python backend that converts it into
  point clouds, compares gestures using Chamfer Distance, and classifies them
  with a small neural network. This continues the work of the *C4GT IIIT Delhi*
  report on occlusion-robust gesture recognition, and is evaluated with the
  out-of-distribution methodology from the *EMGBench* paper.

- **EMG fatigue analysis (`analysis/`)** — a pipeline that takes raw surface-EMG
  (sEMG) recordings from a Delsys Trigno system, extracts muscle-activation and
  fatigue features, and uses an LLM (or an offline rule-based interpreter) to
  produce a readable report on which muscles activated and which fatigued.

Both are built the same way: small, documented, well-tested Python modules with
a clear phase-by-phase structure.

---

## 2. Background & motivation

The work is grounded in foundational literature (papers are not committed — see
[section 10](#10-what-is-intentionally-not-in-the-repo)):

- **C4GT IIIT Delhi report** — occlusion-robust hand-gesture recognition using
  3D landmarks, point clouds, and Chamfer Distance. The gesture backend
  continues this work and implements its proposed-but-unfinished neural
  pre-classifier.
- **EMGBench (Yang et al., CMU 2024)** — out-of-distribution benchmarking for
  EMG. We borrow its LOSO-CV and train-test-split-for-time-series evaluation.
- **sEMG fatigue literature** (Clancy 2023; Sun 2022; Carvalho 2023; Srinidhi
  2025; Cerqueira 2024) — the EMG analysis pipeline implements the established
  amplitude-estimation, fatigue-feature (RMS↑ + median-frequency↓), and
  onset-detection methods from these papers.

---

## 3. Repository structure

```
Summer/
├── README.md                  ← you are here (start here)
│
├── backend/                   SYSTEM 1: gesture recognition pipeline
│   ├── server.py              WebSocket server (entry point)
│   ├── train.py               offline classifier training
│   ├── evaluate.py            offline evaluation / reporting
│   ├── protocol/              binary frame deserializer (Quest -> Python)
│   ├── pipeline/              point cloud, Chamfer distance, dataset, classifier
│   ├── tests/                 179 tests
│   ├── PROTOCOL.md            binary wire-format spec
│   └── README.md              detailed backend docs
│
├── analysis/                  SYSTEM 2: EMG fatigue analysis
│   ├── report.py              end-to-end CLI (entry point)
│   ├── emg_pipeline/          parser, preprocess, features, analysis, llm
│   ├── emg_pipeline/tests/    106 tests
│   ├── data/                  place Delsys .csv recordings here (gitignored)
│   └── README.md              detailed EMG docs + scientific basis
│
├── VR IP/                     VR frontend + EMG hardware integration
│   ├── frontend/              Unity 6000.3.2f1 DepthVR app (Meta Quest)
│   ├── VR-Delsys-Integration-main/   Delsys EMG sensor SDK + examples
│   └── readme_depthvr.md      Quest build/run instructions
│
├── HandsOn/                   exploratory Jupyter notebooks
├── Reports/                   weekly progress reports
└── Meeting Records/           meeting notes
```

Each system has its **own detailed README** (`backend/README.md`,
`analysis/README.md`). This top-level file is the map and the testing guide.

---

## 4. The two systems at a glance

### System 1 — Gesture recognition (`backend/`)

```
Meta Quest (Unity) ──WebSocket──► server.py
                                    Step 1: decode binary frames        protocol/
                                    Step 2: joints -> point cloud       pipeline/point_cloud.py
                                    Step 3: Chamfer distance + recorder  pipeline/chamfer.py, recorder.py
                                    Step 4: dataset + MLP classifier     pipeline/dataset.py, classifier.py
```
Recognises 10 hand gestures (ASL_L, ASL_Y, Five, Four, One, Spiderman, Spock,
Three, Thumbs_Up, Two). Pure-NumPy classifier, no GPU needed.

### System 2 — EMG fatigue analysis (`analysis/`)

```
Delsys .csv ──► report.py
                  Phase A: parse + filter            emg_pipeline/delsys_parser.py, preprocess.py
                  Phase B: features + fatigue metrics emg_pipeline/features.py
                  Phase C: onset detection + plots    emg_pipeline/analysis.py
                  Phase D: summary + LLM report       emg_pipeline/summary.py, llm.py
```
Detects the canonical fatigue signature (RMS amplitude rising **and** median
frequency falling) per sensor and explains it in plain language.

---

## 5. Prerequisites

- **Python 3.10 or newer** (developed/tested on 3.14)
- `pip` for installing dependencies
- No GPU, no VR headset, and no API key are needed to run the tests.

Each system has its own `requirements.txt`. Install them independently
(ideally in a virtual environment):

```bash
# System 1 — gesture backend
cd backend
pip install -r requirements.txt

# System 2 — EMG analysis
cd ../analysis
pip install -r requirements.txt
```

---

## 6. How to test everything

Both systems ship with comprehensive test suites that run on **synthetic data**
— no recordings, no hardware, no API key required. **285 tests total.**

### Test System 1 (gesture backend) — 179 tests

```bash
cd backend
python3 -m pytest tests/ -v
```

Covers: binary protocol decoding, point-cloud normalisation, Chamfer Distance,
the session recorder, dataset splits (LOSO-CV / time-series), and MLP training
+ inference + persistence.

### Test System 2 (EMG analysis) — 106 tests

```bash
cd analysis
python3 -m pytest emg_pipeline/tests/ -v
```

Covers: the Delsys CSV parser, filtering, all features (RMS, MAV, MDF, MNF, …),
fatigue-trend detection on a synthetic fatiguing signal, onset detection,
plotting, the summary builder, and the LLM interpretation path (via a mock
client — no API key).

### One-liner to run both

```bash
( cd backend && python3 -m pytest tests/ -q ) && \
( cd analysis && python3 -m pytest emg_pipeline/tests/ -q )
```

Expected result: **179 passed** then **106 passed**.

---

## 7. System 1 — Gesture recognition backend

Full details in [`backend/README.md`](backend/README.md). Quick tour:

### Run the tests
```bash
cd backend && python3 -m pytest tests/ -v
```

### Try it without a headset (offline)
The classifier and all processing run on saved data. To collect new data you
need a Meta Quest running the DepthVR app (see `VR IP/readme_depthvr.md`), but
the code, tests, and training/evaluation logic are fully exercisable offline.

Typical workflow once recordings exist:
```bash
# Record gesture sessions from the Quest (needs headset):
python3 server.py --record --gesture Thumbs_Up --recordings-dir recordings/

# Train + evaluate (offline, no headset):
python3 train.py    --recordings-dir recordings/ --model-out models/clf.npz --loso
python3 evaluate.py --model models/clf.npz --recordings-dir recordings/ --loso
```

---

## 8. System 2 — EMG fatigue analysis

Full details in [`analysis/README.md`](analysis/README.md). Quick tour:

### Run the tests
```bash
cd analysis && python3 -m pytest emg_pipeline/tests/ -v
```

### Generate a real fatigue report (offline, no API key)
Place the Delsys `.csv` recordings in `analysis/data/`, then:

```bash
cd analysis
python3 report.py \
    --task data/Avnish_push1_01.csv \
    --mvc  data/Avnish_push1_leftdynamo_01.csv data/Avnish_push1_righydynamo_01.csv \
    --task-name "push task"
```

This prints a per-sensor activation + fatigue report using the built-in
**rule-based interpreter** — no OpenAI key needed. Example output (real data):

```
EMG fatigue analysis — push task
Overall: 7 of 16 sensor(s) show the fatigue signature.
Most fatigued: Sensor 99.
- Sensor 99: mean activation 40%MVC (peak 174%MVC); FATIGUE detected:
  amplitude rose +56% while MDF fell from 65 to 53 Hz (-19%).
  ...
```

### Optional: use OpenAI for a richer narrative
When an API key is available later:
```bash
pip install openai
export OPENAI_API_KEY=sk-...
python3 report.py --task data/Avnish_push1_01.csv \
    --mvc data/Avnish_push1_leftdynamo_01.csv data/Avnish_push1_righydynamo_01.csv \
    --llm --plots output/
```
Without `--llm` the offline rule-based interpreter is used, so the feature is
entirely optional.

### Optional: name the muscles
We do not currently have the sensor→muscle mapping, so reports refer to
"Sensor 1 … Sensor 99". When the electrode placement is known, drop in a JSON
mapping and the report uses real muscle names:
```bash
python3 report.py --task data/Avnish_push1_01.csv --labels labels.json
# labels.json:  {"1": "Biceps Brachii", "2": "Triceps", ...}
```

---

## 9. What needs hardware vs what does not

| Activity | Hardware / key needed? |
|---|---|
| Run all 285 tests | **No** — synthetic data |
| Train/evaluate the gesture classifier on existing recordings | No |
| Generate an EMG fatigue report (rule-based) | No (just the `.csv` files) |
| Record new gesture data | Meta Quest headset + DepthVR app |
| Record new EMG data | Delsys Trigno system |
| LLM-written EMG narrative | OpenAI API key (optional; rule-based fallback otherwise) |

---

## 10. What is intentionally not in the repo

To keep the repository lean and license-clean, the following are gitignored:

- **Research papers** (`*.pdf`) — copyrighted; obtain from the publishers.
- **Raw EMG recordings** (`analysis/data/*.csv`) — large; place them locally in
  `analysis/data/` to run real-data reports. The tests do not need them.
- **Large Unity/EMG binaries** — Unity's `Library/`, gesture-demo videos, the
  Delsys SDK DLLs, and large recording datasets. The Unity **source** and
  scripts are committed.
- **Generated artifacts** — trained models, plots (`output/`), caches.

Everything required to understand the code and run the full test suites **is**
in the repository.

---

## Summary for the reviewer

1. `cd backend && pip install -r requirements.txt && python3 -m pytest tests/ -q` → **179 passed**
2. `cd analysis && pip install -r requirements.txt && python3 -m pytest emg_pipeline/tests/ -q` → **106 passed**
3. Read `backend/README.md` and `analysis/README.md` for the deep dives.
4. With the Delsys `.csv` files in `analysis/data/`, run the EMG report command
   in [section 8](#8-system-2--emg-fatigue-analysis) for a real fatigue analysis
   — no API key required.
