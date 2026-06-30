# EMG Muscle-Activation & Fatigue Analysis

A pipeline that takes raw surface-EMG (sEMG) recordings from a **Delsys Trigno**
system and turns them into a readable report on **which muscles activated** and
**which fatigued** — with an optional LLM layer that writes the report in plain
language.

This is a parallel sub-task of the summer project, independent of the gesture
`backend/`. It analyses multi-sensor EMG to quantify muscle activation and
fatigue using established signal-processing methods, then interprets the results.

> **For the reviewer:** everything here runs **without any hardware and without
> an API key** — the test suite uses synthetic signals, and the report generator
> has a built-in offline interpreter. Jump to [How to test](#how-to-test).

---

## Table of contents

1. [What this does & why](#1-what-this-does--why)
2. [The scientific idea (in one paragraph)](#2-the-scientific-idea-in-one-paragraph)
3. [How to test](#3-how-to-test)
4. [Generate a real fatigue report](#4-generate-a-real-fatigue-report)
5. [The data](#5-the-data)
6. [How the pipeline works (4 phases)](#6-how-the-pipeline-works-4-phases)
7. [Scientific basis (papers)](#7-scientific-basis-papers)
8. [Using it from Python](#8-using-it-from-python)
9. [Notes on data & papers](#9-notes-on-data--papers)

---

## 1. What this does & why

Muscles fatigue during sustained or repeated effort. Detecting that fatigue
objectively is useful for ergonomics, rehabilitation, sports science, and injury
prevention. Surface EMG measures the electrical activity of muscles, and decades
of research have established **how fatigue shows up in the signal**. This project
implements that established analysis end-to-end:

1. **Read** the messy multi-sensor Delsys CSV correctly.
2. **Clean** the signal (filtering).
3. **Extract** the features that quantify activation and fatigue.
4. **Interpret** those features into a human-readable report.

The novel part is step 4: instead of leaving a researcher to read dense feature
tables, we summarise the numbers and let an **LLM explain them** — or fall back
to a built-in rule-based explainer when no LLM is available.

---

## 2. The scientific idea (in one paragraph)

When a muscle fatigues during a sustained contraction, two things happen to its
EMG signal at the same time: the **amplitude rises** (more motor units recruited
— measured by RMS) and the **frequency content falls** (slower muscle-fibre
conduction — measured by the median frequency, MDF). So the fatigue signature we
look for, per muscle, is **RMS going up while MDF goes down** over the course of
the task. This pipeline computes both trends and flags the muscles that show it.

---

## 3. How to test

```bash
cd analysis
pip install -r requirements.txt
python3 -m pytest emg_pipeline/tests/ -v
```

Expected result: **106 passed**.

The tests use synthetic signals with known properties (e.g. a deliberately
"fatiguing" signal whose amplitude rises and frequency falls, which the detector
must flag), so they run on any machine with **no recordings, no hardware, and no
API key**. The LLM path is tested through a mock client. A real-data integration
test runs automatically only if the Delsys CSVs are present, and skips otherwise.

---

## 4. Generate a real fatigue report

Place the Delsys `.csv` recordings in `analysis/data/`, then:

```bash
cd analysis
python3 report.py \
    --task data/Avnish_push1_01.csv \
    --mvc  data/Avnish_push1_leftdynamo_01.csv data/Avnish_push1_righydynamo_01.csv \
    --task-name "push task"
```

This prints a per-sensor activation + fatigue report using the built-in
**rule-based interpreter — no API key needed**. Real example output:

```
EMG fatigue analysis — push task
Overall: 7 of 16 sensor(s) show the fatigue signature.
Most fatigued: Sensor 99.
- Sensor 99: mean activation 40%MVC (peak 174%MVC); FATIGUE detected:
  amplitude rose +56% while MDF fell from 65 to 53 Hz (-19%).
- Sensor 66: ... FATIGUE detected: amplitude rose +62% while MDF fell ...
  ...
```

**Optional — richer narrative with OpenAI** (only when a key is available):
```bash
pip install openai
export OPENAI_API_KEY=sk-...
python3 report.py --task data/Avnish_push1_01.csv \
    --mvc data/Avnish_push1_leftdynamo_01.csv data/Avnish_push1_righydynamo_01.csv \
    --llm --plots output/
```
Without `--llm`, the offline interpreter is used — the OpenAI step is entirely
optional.

**Optional — name the muscles.** We do not yet have the sensor→muscle mapping,
so reports say "Sensor 1 … Sensor 99". When electrode placement is known, supply
a JSON mapping and reports use real names:
```bash
python3 report.py --task data/Avnish_push1_01.csv --labels labels.json
# labels.json:  {"1": "Biceps Brachii", "2": "Triceps", ...}
```

### report.py options

| Flag | Default | Description |
|---|---|---|
| `--task` | (required) | Main task recording CSV to analyse |
| `--mvc` | — | One or more max-effort (dynamometer) CSVs for %MVC normalisation |
| `--task-name` | filename | Human-readable task name for the report |
| `--labels` | — | JSON file mapping sensor id → muscle name |
| `--window-ms` | 250 | Feature window length (ms) |
| `--overlap` | 0.5 | Window overlap fraction |
| `--notch-hz` | 50 | Power-line notch: 50 (India) or 60 (Americas) |
| `--llm` | off | Use OpenAI for interpretation (needs `OPENAI_API_KEY`) |
| `--model` | gpt-4o-mini | OpenAI model |
| `--plots` | — | Directory to write trend/ranking plots into |

---

## 5. The data

Delsys Trigno Discover exports, **16 Avanti sensors** (EMG @ ~1259 Hz,
accelerometer/gyroscope @ ~148 Hz). A recording session typically includes:

- **baseline rest** — noise floor / resting activity
- **left / right dynamometer** — maximum voluntary contraction (MVC) references,
  used to normalise amplitude to **%MVC** (effort relative to the muscle's max)
- **push task** — the sustained effort analysed for fatigue

Sensors are identified by hardware id (`1, 2, …, 99`). A **sensor→muscle mapping
is optional**: all analysis works per-sensor, and muscle names can be dropped in
later without changing anything else.

---

## 6. How the pipeline works (4 phases)

```
Phase A  Parsing + preprocessing      emg_pipeline/delsys_parser.py, preprocess.py
Phase B  Feature extraction           emg_pipeline/features.py
Phase C  Analysis + visualisation     emg_pipeline/analysis.py
Phase D  Summary + LLM report         emg_pipeline/summary.py, llm.py
                                       report.py  (end-to-end CLI)
```

### Phase A — Parsing + preprocessing
- `delsys_parser.py` — a **robust parser** for the Trigno Discover CSV, which is
  not a normal table: it has 8 header rows, and **every channel has its own time
  column** because EMG (~1259 Hz) and the motion sensors (~148 Hz) sample at
  different rates. A single CSV row is therefore *not* one instant in time; each
  channel is read independently. Extracts one clean EMG channel per sensor.
- `preprocess.py` — best-practice conditioning (Clancy et al. 2023): zero-phase
  Butterworth band-pass (20–450 Hz) + power-line notch (50 Hz, configurable).

### Phase B — Feature extraction
- `features.py` — slides a window (default 250 ms, 50% overlap) over each sensor
  and computes:
  - **time-domain:** RMS, MAV, integrated EMG, zero-crossings, waveform length
  - **frequency-domain:** mean frequency (MNF), **median frequency (MDF)**, total power
  - **%MVC** when a dynamometer reference is supplied
- **Fatigue metrics:** fits a line to RMS(t) and MDF(t) over the whole task;
  flags fatigue when **RMS slope > 0 and MDF slope < 0** (the signature from §2).
  Reported start/end values are the *fitted* regression endpoints (robust),
  so the numbers always agree with the trend direction.

### Phase C — Analysis + visualisation
- `analysis.py` — muscle **activation onset detection** (Carvalho et al. 2023:
  linear envelope + adaptive threshold + optional Teager-Kaiser energy operator)
  and **plots**: RMS-over-time, MDF-over-time, a fatigue-ranking bar chart, and a
  per-channel overview (EMG + envelope + shaded activations).

### Phase D — Summary + LLM report
- `summary.py` — distils the dense feature tables into a compact JSON summary
  (per sensor: activation level, %MVC, MDF trend, fatigue flag). **Every number
  is computed by the pipeline; the LLM never invents values.**
- `llm.py` —
  - a domain-grounded prompt encoding the fatigue rules + a guardrail to only
    interpret supplied numbers,
  - `OpenAIClient` (real OpenAI; key from `OPENAI_API_KEY`),
  - `rule_based_report` (deterministic offline interpreter; the default),
  - `interpret_fatigue(summary, client=None)` — the entry point.
- `report.py` — ties it all together into one CLI command.

---

## 7. Scientific basis (papers)

The methods implement established sEMG literature (papers not committed — see
[§9](#9-notes-on-data--papers)):

| Topic | Source |
|---|---|
| Amplitude estimation best practices (filtering, RMS/MAV, normalisation) | Clancy et al. 2023, *J. Electromyogr. Kinesiol.* |
| Fatigue features (RMS ↑, median/mean frequency ↓) | Sun et al. 2022, *Front. Syst. Neurosci.* |
| Activation onset detection (threshold methods, TKEO) | Carvalho et al. 2023, *J. NeuroEng. Rehabil.* |
| Applied fatigue methodology (%MVC, MDF slope) | Srinidhi et al. 2025, *Sci. Reports* |
| Fatigue labelling / ground truth | Cerqueira et al. 2024, *Sensors* |
| sEMG biophysics & interpretation | McManus et al. 2020, *Front. Neurol.* |

---

## 8. Using it from Python

```python
from emg_pipeline import (
    parse_delsys_csv, preprocess_recording,
    compute_mvc_reference, combine_mvc_references,
    extract_recording_features, fatigue_summary_frame,
    build_recording_summary, interpret_fatigue,
)

# 1. MVC reference from the dynamometer (max-effort) recordings
left  = preprocess_recording(parse_delsys_csv("data/Avnish_push1_leftdynamo_01.csv"))
right = preprocess_recording(parse_delsys_csv("data/Avnish_push1_righydynamo_01.csv"))
mvc   = combine_mvc_references(compute_mvc_reference(left), compute_mvc_reference(right))

# 2. Features on the main task, normalised to %MVC
push  = preprocess_recording(parse_delsys_csv("data/Avnish_push1_01.csv"))
feats = extract_recording_features(push, mvc_references=mvc)

# 3. Per-sensor fatigue table (most-fatigued first)
print(fatigue_summary_frame(feats))

# 4. Human-readable report (rule-based; pass an OpenAIClient to use the LLM)
summary = build_recording_summary(feats, task_name="push task")
print(interpret_fatigue(summary))
```

---

## 9. Notes on data & papers

`.gitignore` excludes `data/*.csv` (large raw recordings) and `*.pdf`
(copyrighted papers), as well as generated `output/` plots and caches. Place the
Delsys CSV exports in `analysis/data/` to run the pipeline on real recordings;
the test suite does not need them.

**Dependencies:** `pip install -r requirements.txt`. The `openai` package is
listed but only needed for the `--llm` path; everything else (including the full
test suite and the rule-based report) works without it.
