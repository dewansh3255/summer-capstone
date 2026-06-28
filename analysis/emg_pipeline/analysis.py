"""
analysis.py
-----------
Phase C: activation onset detection + visualisation.

Two capabilities:

1. Muscle activation onset/offset detection (Carvalho et al. 2023):
   - Linear envelope (rectify + low-pass) of the conditioned EMG.
   - Single-threshold detector: activation when the envelope exceeds
     baseline_mean + k * baseline_std for at least a minimum duration.
   - Optional Teager-Kaiser Energy Operator (TKEO) pre-emphasis, which
     improves onset sharpness by amplifying instantaneous amplitude*frequency
     (recommended by Carvalho et al. for robust onset timing).

2. Plotting (saved as PNGs; headless 'Agg' backend so it runs on a server):
   - RMS over time per sensor (amplitude trend)
   - MDF over time per sensor (spectral / fatigue trend)
   - Fatigue ranking bar chart across all sensors
   - Per-channel overview (raw + envelope + detected onsets)

All plotting functions WRITE FILES and return the output path; they never call
plt.show(), so they are safe in notebooks, scripts, and CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("Agg")            # headless backend — must precede pyplot import
import matplotlib.pyplot as plt

from .delsys_parser import EMGChannel
from .features import ChannelFeatures


# ──────────────────────────────────────────────────────────────────────────────
# Envelope + TKEO
# ──────────────────────────────────────────────────────────────────────────────

def teager_kaiser_energy(x: np.ndarray) -> np.ndarray:
    """
    Teager-Kaiser Energy Operator: psi[n] = x[n]^2 - x[n-1]*x[n+1].

    Emphasises segments of high instantaneous amplitude AND frequency, sharpening
    EMG burst boundaries. Output length matches input (edges zero-padded).
    """
    if len(x) < 3:
        return np.zeros_like(x)
    psi = np.zeros_like(x, dtype=np.float64)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return psi


def linear_envelope(
    signal: np.ndarray,
    fs: float,
    lowpass_hz: float = 5.0,
    order: int = 2,
) -> np.ndarray:
    """
    Linear envelope = full-wave rectification followed by a low-pass Butterworth
    filter (zero-phase). The standard way to obtain a smooth EMG amplitude
    envelope for onset detection and activation visualisation.
    """
    if fs <= 0 or len(signal) == 0:
        return np.abs(signal).astype(np.float64)
    rectified = np.abs(signal)
    nyquist = 0.5 * fs
    cutoff = min(lowpass_hz, nyquist * 0.99) / nyquist
    b, a = butter(order, cutoff, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(rectified) <= padlen:
        return rectified.astype(np.float64)
    return filtfilt(b, a, rectified).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Onset detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivationEvent:
    """A single detected muscle activation (onset -> offset)."""
    onset_s: float
    offset_s: float

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s

    def __repr__(self) -> str:
        return f"Activation({self.onset_s:.3f}s -> {self.offset_s:.3f}s, {self.duration_s:.3f}s)"


def detect_onsets(
    channel: EMGChannel,
    k: float = 3.0,
    min_duration_ms: float = 50.0,
    baseline_s: float = 0.5,
    lowpass_hz: float = 5.0,
    use_tkeo: bool = True,
    threshold: Optional[float] = None,
) -> list[ActivationEvent]:
    """
    Detect activation onsets/offsets using a single adaptive threshold on the
    linear envelope (Carvalho et al. 2023, single-threshold method).

    Parameters
    ----------
    channel : EMGChannel
        Preferably preprocessed (band-pass + notch).
    k : float
        Threshold = baseline_mean + k * baseline_std. Higher k = stricter.
    min_duration_ms : float
        Minimum time the envelope must stay above threshold to count as an
        activation (rejects brief noise spikes).
    baseline_s : float
        Duration at the start of the signal used to estimate baseline noise
        statistics. Use a quiet/rest segment for best results.
    lowpass_hz : float
        Envelope low-pass cutoff.
    use_tkeo : bool
        Apply TKEO pre-emphasis before enveloping (sharper onsets).
    threshold : float, optional
        Explicit envelope threshold. Overrides the adaptive baseline threshold
        when provided.

    Returns
    -------
    list[ActivationEvent]
        Detected activations in chronological order.
    """
    sig = channel.signal
    fs = channel.fs
    t = channel.time
    if len(sig) == 0 or fs <= 0:
        return []

    proc = teager_kaiser_energy(sig) if use_tkeo else sig
    env = linear_envelope(proc, fs, lowpass_hz=lowpass_hz)

    # Adaptive threshold from the baseline window
    if threshold is None:
        n_base = max(1, int(baseline_s * fs))
        n_base = min(n_base, len(env))
        base = env[:n_base]
        threshold = float(base.mean() + k * base.std())

    above = env >= threshold
    min_samples = max(1, int(min_duration_ms / 1000.0 * fs))

    events: list[ActivationEvent] = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if (j - i) >= min_samples:
                events.append(ActivationEvent(onset_s=float(t[i]),
                                              offset_s=float(t[min(j, n) - 1])))
            i = j
        else:
            i += 1
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_rms_trend(
    feats: dict[str, ChannelFeatures],
    out_path: str | Path,
    use_pct_mvc: bool = False,
    title: str = "RMS amplitude over time",
) -> Path:
    """
    Plot RMS (or %MVC) vs time for every sensor on one axis.
    Returns the saved file path.
    """
    p = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(11, 6))
    col = "pct_mvc" if use_pct_mvc else "rms"
    ylabel = "%MVC" if use_pct_mvc else "RMS (mV)"
    for sid, cf in feats.items():
        if col in cf.df.columns and len(cf.df):
            ax.plot(cf.df["t_center"], cf.df[col], label=f"S{sid}", linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_mdf_trend(
    feats: dict[str, ChannelFeatures],
    out_path: str | Path,
    title: str = "Median frequency (MDF) over time — declining = fatigue",
) -> Path:
    """Plot MDF vs time for every sensor (declining trend indicates fatigue)."""
    p = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(11, 6))
    for sid, cf in feats.items():
        if "mdf" in cf.df.columns and len(cf.df):
            ax.plot(cf.df["t_center"], cf.df["mdf"], label=f"S{sid}", linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Median frequency (Hz)")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_fatigue_ranking(
    summary_df,
    out_path: str | Path,
    title: str = "Fatigue ranking (MDF slope; more negative = more fatigue)",
) -> Path:
    """
    Horizontal bar chart of per-sensor MDF slope from a fatigue summary frame
    (see features.fatigue_summary_frame). Sensors flagged fatigue_detected are
    coloured distinctly.
    """
    p = _ensure_dir(out_path)
    df = summary_df.copy()
    labels = [f"S{s}" for s in df["sensor_id"]]
    slopes = df["mdf_slope"].values
    colors = ["#d62728" if f else "#7f7f7f"
              for f in df.get("fatigue_detected", [False] * len(df))]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(df))))
    ax.barh(labels, slopes, color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("MDF slope (Hz/s)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_channel_overview(
    channel: EMGChannel,
    out_path: str | Path,
    events: Optional[list[ActivationEvent]] = None,
    max_seconds: Optional[float] = None,
) -> Path:
    """
    Per-channel diagnostic plot: conditioned EMG + linear envelope, with any
    detected activation events shaded. `max_seconds` limits the plotted span
    for very long recordings (plots only the first max_seconds).
    """
    p = _ensure_dir(out_path)
    sig = channel.signal
    t = channel.time
    fs = channel.fs

    if max_seconds is not None and fs > 0:
        n = int(max_seconds * fs)
        sig, t = sig[:n], t[:n]

    env = linear_envelope(sig, fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, sig, color="#1f77b4", linewidth=0.4, alpha=0.7, label="EMG")
    ax.plot(t, env, color="#ff7f0e", linewidth=1.2, label="envelope")
    ax.plot(t, -env, color="#ff7f0e", linewidth=1.2)

    if events:
        span = (t[0], t[-1]) if len(t) else (0, 0)
        for ev in events:
            if ev.onset_s <= span[1] and ev.offset_s >= span[0]:
                ax.axvspan(ev.onset_s, ev.offset_s, color="#2ca02c", alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG (mV)")
    ax.set_title(f"Sensor {channel.sensor_id} — EMG + envelope"
                 + (f" + {len(events)} activations" if events else ""))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p
