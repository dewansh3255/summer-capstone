# EMG Analysis Pipeline

Muscle activation and fatigue analysis from **Delsys Trigno Discover** surface
EMG recordings, with an LLM-based interpretation layer.

This package is independent of the gesture-recognition `backend/`. It analyses
multi-sensor sEMG data to quantify muscle activation and fatigue, and (later)
uses an LLM to turn the extracted features into a readable report.

## Scientific basis

The methodology is grounded in the papers in `analysis/` (not committed —
copyrighted):

| Topic | Source |
|---|---|
| Amplitude estimation best practices (filtering, RMS/MAV, normalisation) | Clancy et al. 2023, *J. Electromyogr. Kinesiol.* |
| Fatigue features (RMS ↑, median/mean frequency ↓) | Sun et al. 2022, *Front. Syst. Neurosci.* |
| Activation onset detection (threshold methods, TKEO) | Carvalho et al. 2023, *J. NeuroEng. Rehabil.* |
| Applied fatigue methodology (%MVC, MDF slope) | Srinidhi et al. 2025, *Sci. Reports* |
| Fatigue labelling / ground truth | Cerqueira et al. 2024, *Sensors* |

**The fatigue signature:** during a sustained contraction, a fatiguing muscle
shows **rising RMS amplitude** together with a **falling median/mean frequency**.
That dual trend is the primary target of the feature and analysis stages.

## The data

Delsys Trigno Discover exports, 16 Avanti sensors (EMG @ ~1259 Hz, ACC/GYRO @
~148 Hz). A recording session typically includes:

- **baseline rest** — noise floor / resting activity
- **left / right dynamometer** — maximum voluntary contraction (MVC) references
  used to normalise amplitude to %MVC
- **push task** — the sustained effort analysed for fatigue

Sensors are identified by their hardware id (`1, 2, ..., 99`). A
**sensor → muscle mapping is optional**: all analysis works per-sensor. When a
mapping becomes available it is dropped in as metadata and the reports simply
swap sensor ids for muscle names.

## Pipeline phases

```
Phase A ✅  Parsing + preprocessing      delsys_parser.py, preprocess.py
Phase B ✅  Feature extraction           features.py
Phase C     Analysis + visualisation     analysis.py     [next]
Phase D     LLM interpretation           llm.py
```

### Phase A (done)

- `delsys_parser.py` — robust parser for the Trigno Discover wide-CSV format.
  Handles the 8 header rows, per-channel independent time columns, multi-rate
  channels, and trailing empty cells. Extracts one `EMGChannel` per sensor.
- `preprocess.py` — Clancy best-practice conditioning: zero-phase Butterworth
  band-pass (20–450 Hz) + power-line notch (50 Hz, India mains; configurable).

### Phase B (done)

- `features.py` — sliding-window feature extraction (default 250 ms, 50%
  overlap) computing per window:
  - **time-domain:** RMS, MAV, integrated EMG, zero-crossings, waveform length
  - **frequency-domain:** mean frequency (MNF), median frequency (MDF), total power
  - **%MVC** when a dynamometer/max-effort reference is supplied
- **MVC reference** from the dynamometer recordings (`compute_mvc_reference`,
  `combine_mvc_references` — per-sensor max across left/right).
- **Fatigue-trend metrics** (`compute_fatigue_metrics`): linear fit of RMS(t)
  and MDF(t) over the task. `fatigue_detected = rms_slope > 0 AND mdf_slope < 0`
  — the canonical fatigue signature.

  Note: `*_slope` (linear regression over all windows) is the robust trend
  measure used for detection; `*_pct_change` (endpoint-to-endpoint) is reported
  for convenience but is sensitive to end-window spikes, so the two can
  occasionally disagree. Detection always uses the regression slope.

## Usage

```python
from emg_pipeline import (
    parse_delsys_csv, preprocess_recording,
    compute_mvc_reference, combine_mvc_references,
    extract_recording_features, fatigue_summary_frame,
)

# 1. MVC reference from the dynamometer (max-effort) recordings
left  = preprocess_recording(parse_delsys_csv("data/Avnish_push1_leftdynamo_01.csv"))
right = preprocess_recording(parse_delsys_csv("data/Avnish_push1_righydynamo_01.csv"))
mvc   = combine_mvc_references(compute_mvc_reference(left), compute_mvc_reference(right))

# 2. Features on the main task, normalised to %MVC
push  = preprocess_recording(parse_delsys_csv("data/Avnish_push1_01.csv"))
feats = extract_recording_features(push, mvc_references=mvc)

# 3. Per-sensor fatigue trend table (most-fatigued first)
summary = fatigue_summary_frame(feats)
print(summary)
```

## Setup & tests

```bash
pip install -r requirements.txt
python3 -m pytest emg_pipeline/tests/ -v
```

61 tests cover parsing (synthetic + real-data integration), filtering, and
feature extraction / fatigue metrics. Tests use synthetic signals so they run
without the real recordings; the real-data test skips automatically when
`data/` is absent.

## Data & papers are not committed

`.gitignore` excludes `data/*.csv` (large) and `*.pdf` (copyrighted). Place the
Delsys CSV exports in `analysis/data/` to run the pipeline on real recordings.
