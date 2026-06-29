"""
features.py
-----------
Phase B: windowed feature extraction + fatigue-trend metrics.

For each EMG channel we slide a fixed window (default 250 ms, 50% overlap) and
compute, per window:

  Time-domain (amplitude / activation):
    - RMS  : root-mean-square amplitude        (rises with fatigue & effort)
    - MAV  : mean absolute value
    - IEMG : integrated EMG (sum of |x|)
    - ZC   : zero crossings (with deadzone)    (frequency-content proxy)
    - WL   : waveform length (sum |dx|)

  Frequency-domain (fatigue gold standard):
    - MNF  : mean power frequency              (falls with fatigue)
    - MDF  : median frequency                  (falls with fatigue)
    - total power

Then, over the whole task, we fit a line to RMS(t) and MDF(t) to obtain the
fatigue-trend metrics. The canonical fatigue signature (Sun et al. 2022) is:

      RMS slope  > 0   (amplitude rising)
      MDF slope  < 0   (spectrum compressing toward lower frequencies)

References: Clancy 2023 (amplitude), Sun 2022 (fatigue features),
Phinyomark 2012 (feature definitions).

Window length and overlap are parameters. 250 ms is standard for fatigue work;
shorter windows track fast dynamic contractions better at the cost of noisier
frequency estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .delsys_parser import EMGChannel, DelsysRecording


DEFAULT_WINDOW_MS = 250.0
DEFAULT_OVERLAP   = 0.5
# ZC / slope-sign deadzone as a fraction of the window's std, to ignore
# baseline noise crossings (Phinyomark 2012 recommends a small threshold).
ZC_THRESHOLD_FRAC = 0.01

FEATURE_COLUMNS = [
    "t_center", "rms", "mav", "iemg", "zc", "wl", "mnf", "mdf", "total_power",
]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level per-window feature functions
# ──────────────────────────────────────────────────────────────────────────────

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x))) if len(x) else 0.0


def iemg(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x))) if len(x) else 0.0


def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x)))) if len(x) > 1 else 0.0


def zero_crossings(x: np.ndarray, threshold: float = 0.0) -> int:
    """
    Count sign changes where the amplitude change exceeds `threshold`.
    The threshold (deadzone) suppresses spurious crossings from baseline noise.
    """
    if len(x) < 2:
        return 0
    s = np.sign(x)
    s[s == 0] = 1
    sign_change = s[:-1] != s[1:]
    amp_ok = np.abs(x[:-1] - x[1:]) >= threshold
    return int(np.sum(sign_change & amp_ok))


def power_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided power spectral density via the periodogram (|rfft|^2).
    Returns (freqs, psd). A Hann window reduces spectral leakage, which matters
    for MDF/MNF stability on short windows.
    """
    n = len(x)
    if n < 2 or fs <= 0:
        return np.array([]), np.array([])
    w = np.hanning(n)
    xw = x * w
    spec = np.abs(np.fft.rfft(xw)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """MNF = sum(f * P) / sum(P)."""
    total = np.sum(psd)
    return float(np.sum(freqs * psd) / total) if total > 0 else 0.0


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """MDF = frequency at which cumulative power reaches half the total power."""
    total = np.sum(psd)
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(psd)
    idx = int(np.searchsorted(cumulative, total / 2.0))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────────────────────────────────────

def _iter_windows(n: int, win: int, step: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) sample indices for each full window."""
    if win <= 0 or step <= 0:
        return
    start = 0
    while start + win <= n:
        yield start, start + win
        start += step


# ──────────────────────────────────────────────────────────────────────────────
# Output containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelFeatures:
    """
    Windowed features for one EMG channel.

    Attributes
    ----------
    sensor_id : str
    fs : float
    window_ms, overlap : float
        Windowing parameters used.
    df : pandas.DataFrame
        One row per window, columns = FEATURE_COLUMNS (+ 'pct_mvc' if a MVC
        reference was supplied).
    """
    sensor_id: str
    fs: float
    window_ms: float
    overlap: float
    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"ChannelFeatures(sensor={self.sensor_id!r}, "
            f"windows={len(self.df)}, window_ms={self.window_ms:g})"
        )


@dataclass
class FatigueMetrics:
    """
    Whole-task fatigue-trend metrics for one channel (from a linear fit of each
    feature against window-centre time).

    Positive rms_slope and negative mdf_slope together indicate fatigue.
    """
    sensor_id: str
    n_windows: int
    duration_s: float
    rms_start: float
    rms_end: float
    rms_slope: float            # amplitude units per second
    rms_pct_change: float       # % change end vs start
    mdf_start: float            # Hz
    mdf_end: float              # Hz
    mdf_slope: float            # Hz per second  (negative => fatigue)
    mdf_pct_change: float       # % change end vs start
    mnf_slope: float            # Hz per second
    fatigue_detected: bool      # rms_slope > 0 AND mdf_slope < 0

    def as_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "n_windows": self.n_windows,
            "duration_s": round(self.duration_s, 2),
            "rms_start": self.rms_start,
            "rms_end": self.rms_end,
            "rms_slope": self.rms_slope,
            "rms_pct_change": self.rms_pct_change,
            "mdf_start": self.mdf_start,
            "mdf_end": self.mdf_end,
            "mdf_slope": self.mdf_slope,
            "mdf_pct_change": self.mdf_pct_change,
            "mnf_slope": self.mnf_slope,
            "fatigue_detected": self.fatigue_detected,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_channel_features(
    channel: EMGChannel,
    window_ms: float = DEFAULT_WINDOW_MS,
    overlap: float = DEFAULT_OVERLAP,
    mvc_reference: Optional[float] = None,
) -> ChannelFeatures:
    """
    Compute windowed features for one EMG channel.

    Parameters
    ----------
    channel : EMGChannel
        Preferably already preprocessed (band-pass + notch).
    window_ms : float
        Window length in milliseconds.
    overlap : float
        Fractional overlap in [0, 1). 0.5 = 50% overlap.
    mvc_reference : float, optional
        Per-channel MVC RMS for %MVC normalisation. If given, a 'pct_mvc'
        column is added (100 * rms / mvc_reference).

    Returns
    -------
    ChannelFeatures
    """
    fs = channel.fs
    sig = channel.signal
    t = channel.time

    win = int(round(window_ms / 1000.0 * fs))
    if win < 2:
        win = 2
    step = max(1, int(round(win * (1.0 - overlap))))

    rows = []
    for a, b in _iter_windows(len(sig), win, step):
        seg = sig[a:b]
        t_center = float(t[a] + (t[b - 1] - t[a]) / 2.0)
        zc_thr = ZC_THRESHOLD_FRAC * (np.std(seg) if np.std(seg) > 0 else 0.0)
        freqs, psd = power_spectrum(seg, fs)
        rows.append((
            t_center,
            rms(seg),
            mav(seg),
            iemg(seg),
            zero_crossings(seg, zc_thr),
            waveform_length(seg),
            mean_frequency(freqs, psd),
            median_frequency(freqs, psd),
            float(np.sum(psd)),
        ))

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    if mvc_reference is not None and mvc_reference > 0:
        df["pct_mvc"] = 100.0 * df["rms"] / mvc_reference

    return ChannelFeatures(
        sensor_id=channel.sensor_id,
        fs=fs,
        window_ms=window_ms,
        overlap=overlap,
        df=df,
    )


def extract_recording_features(
    recording: DelsysRecording,
    window_ms: float = DEFAULT_WINDOW_MS,
    overlap: float = DEFAULT_OVERLAP,
    mvc_references: Optional[dict[str, float]] = None,
) -> dict[str, ChannelFeatures]:
    """
    Extract windowed features for every channel in a recording.

    Parameters
    ----------
    mvc_references : dict[sensor_id, float], optional
        Per-sensor MVC RMS for %MVC normalisation (see compute_mvc_reference).

    Returns
    -------
    dict[sensor_id, ChannelFeatures]
    """
    out: dict[str, ChannelFeatures] = {}
    for sid, ch in recording.channels.items():
        mvc = mvc_references.get(sid) if mvc_references else None
        out[sid] = extract_channel_features(
            ch, window_ms=window_ms, overlap=overlap, mvc_reference=mvc,
        )
    return out


def features_to_long_frame(
    feats: dict[str, ChannelFeatures],
) -> pd.DataFrame:
    """
    Combine per-sensor feature tables into one tidy long-form DataFrame with a
    'sensor_id' column. Convenient for plotting and for building the LLM
    feature summary in Phase D.
    """
    frames = []
    for sid, cf in feats.items():
        f = cf.df.copy()
        f.insert(0, "sensor_id", sid)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["sensor_id", *FEATURE_COLUMNS])
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# MVC reference (from dynamometer / max-effort recordings)
# ──────────────────────────────────────────────────────────────────────────────

def compute_mvc_reference(
    recording: DelsysRecording,
    window_ms: float = DEFAULT_WINDOW_MS,
    overlap: float = DEFAULT_OVERLAP,
    percentile: float = 100.0,
) -> dict[str, float]:
    """
    Estimate each channel's MVC (maximum voluntary contraction) RMS from a
    max-effort recording (e.g. a dynamometer task).

    The MVC reference is the peak windowed RMS. Using percentile<100 (e.g. 95)
    makes it robust to single-window spikes/artifacts.

    Parameters
    ----------
    recording : DelsysRecording
        A preprocessed max-effort recording.
    percentile : float
        Percentile of windowed RMS to use as the reference (100 = max).

    Returns
    -------
    dict[sensor_id, float]
        Per-sensor MVC RMS reference.
    """
    refs: dict[str, float] = {}
    feats = extract_recording_features(recording, window_ms=window_ms, overlap=overlap)
    for sid, cf in feats.items():
        if len(cf.df) == 0:
            refs[sid] = 0.0
        else:
            refs[sid] = float(np.percentile(cf.df["rms"].values, percentile))
    return refs


def combine_mvc_references(
    *ref_dicts: dict[str, float],
) -> dict[str, float]:
    """
    Combine multiple MVC reference dicts (e.g. left + right dynamometer) by
    taking the per-sensor maximum. Each sensor's true MVC is the largest effort
    observed across the max-effort recordings.
    """
    combined: dict[str, float] = {}
    for d in ref_dicts:
        for sid, val in d.items():
            combined[sid] = max(combined.get(sid, 0.0), val)
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Fatigue-trend metrics
# ──────────────────────────────────────────────────────────────────────────────

def _linfit(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Least-squares (slope, intercept) of y vs t.
    Returns (0, mean(y)) for <2 points or zero time span.
    """
    if len(t) < 2 or np.ptp(t) == 0:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    slope, intercept = np.polyfit(t, y, 1)
    return float(slope), float(intercept)


def compute_fatigue_metrics(features: ChannelFeatures) -> FatigueMetrics:
    """
    Fit RMS(t) and MDF(t) to obtain whole-task fatigue-trend metrics.

    fatigue_detected is True when amplitude rises (rms_slope > 0) AND the
    spectrum compresses (mdf_slope < 0) — the canonical fatigue signature.

    The reported start/end values are the FITTED endpoints of the regression
    line (value at the first and last window time), NOT the raw first/last
    window. Single-window endpoints are extremely noisy over a multi-minute
    recording and can contradict the overall trend; the fitted endpoints are
    robust and always consistent with the slope sign. Percentage changes are
    derived from these fitted endpoints for the same reason.
    """
    df = features.df
    sid = features.sensor_id

    if len(df) < 2:
        return FatigueMetrics(
            sensor_id=sid, n_windows=len(df), duration_s=0.0,
            rms_start=0.0, rms_end=0.0, rms_slope=0.0, rms_pct_change=0.0,
            mdf_start=0.0, mdf_end=0.0, mdf_slope=0.0, mdf_pct_change=0.0,
            mnf_slope=0.0, fatigue_detected=False,
        )

    t   = df["t_center"].values
    t0, t1 = float(t[0]), float(t[-1])

    rms_slope, rms_b = _linfit(t, df["rms"].values)
    mdf_slope, mdf_b = _linfit(t, df["mdf"].values)
    mnf_slope, _     = _linfit(t, df["mnf"].values)

    # Fitted (robust) endpoints
    rms_start = rms_slope * t0 + rms_b
    rms_end   = rms_slope * t1 + rms_b
    mdf_start = mdf_slope * t0 + mdf_b
    mdf_end   = mdf_slope * t1 + mdf_b

    def pct(a, b):
        return float(100.0 * (b - a) / a) if a != 0 else 0.0

    return FatigueMetrics(
        sensor_id=sid,
        n_windows=len(df),
        duration_s=t1 - t0,
        rms_start=rms_start,
        rms_end=rms_end,
        rms_slope=rms_slope,
        rms_pct_change=pct(rms_start, rms_end),
        mdf_start=mdf_start,
        mdf_end=mdf_end,
        mdf_slope=mdf_slope,
        mdf_pct_change=pct(mdf_start, mdf_end),
        mnf_slope=mnf_slope,
        fatigue_detected=(rms_slope > 0 and mdf_slope < 0),
    )


def fatigue_summary_frame(
    feats: dict[str, ChannelFeatures],
) -> pd.DataFrame:
    """
    Compute fatigue metrics for every channel and return them as one DataFrame
    (one row per sensor), sorted by mdf_slope ascending so the most-fatigued
    channels (steepest spectral decline) appear first.
    """
    rows = [compute_fatigue_metrics(cf).as_dict() for cf in feats.values()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mdf_slope").reset_index(drop=True)
    return df
