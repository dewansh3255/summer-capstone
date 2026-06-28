"""
preprocess.py
-------------
Phase A (part 2): EMG signal conditioning.

Implements the "best practices" amplitude-estimation preprocessing chain from
Clancy et al. (2023), J. Electromyogr. Kinesiol., adapted for surface EMG
recorded with the Delsys Trigno system (~1259 Hz EMG sampling):

    raw EMG (mV)
        │
        ▼  1. Band-pass filter (20-450 Hz, 4th-order Butterworth, zero-phase)
           Removes motion-artifact / baseline wander below 20 Hz and
           high-frequency noise above the EMG band. Zero-phase (filtfilt)
           preserves the temporal location of bursts (important for onset
           detection in Phase C).
        │
        ▼  2. Power-line notch (50 Hz + optional harmonics, India mains)
           Removes mains interference. Narrow IIR notch (filtfilt).
        │
        ▼  conditioned EMG (mV)

Notes
-----
- High cutoff 450 Hz is valid: EMG fs ~1259 Hz -> Nyquist ~629 Hz.
- Rectification and RMS/MAV smoothing are NOT done here; they belong to the
  feature stage (Phase B), where window length is a tunable parameter. Keeping
  conditioning and feature extraction separate keeps the pipeline composable.
- All filters use filtfilt (forward-backward) for zero phase distortion.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

from .delsys_parser import EMGChannel, DelsysRecording


# Default conditioning parameters (overridable per call)
DEFAULT_BAND_LOW_HZ   = 20.0
DEFAULT_BAND_HIGH_HZ  = 450.0
DEFAULT_BAND_ORDER    = 4
DEFAULT_NOTCH_HZ      = 50.0     # India mains; use 60.0 for US/Americas
DEFAULT_NOTCH_Q       = 30.0


# ──────────────────────────────────────────────────────────────────────────────
# Primitive filters (operate on raw arrays)
# ──────────────────────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = DEFAULT_BAND_LOW_HZ,
    high: float = DEFAULT_BAND_HIGH_HZ,
    order: int = DEFAULT_BAND_ORDER,
) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Raw EMG samples.
    fs : float
        Sampling rate (Hz).
    low, high : float
        Passband edges (Hz). `high` is clipped to just below Nyquist if needed.
    order : int
        Butterworth order (applied once forward, once backward via filtfilt,
        so the effective order is doubled).

    Returns
    -------
    np.ndarray, shape (N,)
        Band-pass filtered signal (float64).
    """
    nyquist = 0.5 * fs
    high_clipped = min(high, nyquist * 0.99)
    low_clipped  = max(low, 1e-3)
    if low_clipped >= high_clipped:
        raise ValueError(
            f"Invalid band: low={low_clipped} >= high={high_clipped} (fs={fs})"
        )

    b, a = butter(order, [low_clipped / nyquist, high_clipped / nyquist], btype="band")
    # filtfilt needs signal length > 3 * max(len(a), len(b)); guard short signals.
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal.astype(np.float64, copy=True)
    return filtfilt(b, a, signal).astype(np.float64)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    freq: float = DEFAULT_NOTCH_HZ,
    q: float = DEFAULT_NOTCH_Q,
    harmonics: int = 1,
) -> np.ndarray:
    """
    Zero-phase IIR notch filter at `freq` and (optionally) its harmonics.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
    fs : float
        Sampling rate (Hz).
    freq : float
        Notch centre frequency (Hz), e.g. 50 (India) or 60 (Americas).
    q : float
        Quality factor; higher = narrower notch.
    harmonics : int
        Number of harmonics to notch (1 = fundamental only; 2 = +100 Hz; ...).
        Harmonics at or above Nyquist are skipped.

    Returns
    -------
    np.ndarray, shape (N,)
        Notch-filtered signal (float64).
    """
    nyquist = 0.5 * fs
    out = signal.astype(np.float64, copy=True)
    for k in range(1, harmonics + 1):
        f0 = freq * k
        if f0 >= nyquist * 0.99:
            break
        b, a = iirnotch(f0 / nyquist, q)
        padlen = 3 * max(len(a), len(b))
        if len(out) <= padlen:
            continue
        out = filtfilt(b, a, out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Channel / recording-level conditioning
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_channel(
    channel: EMGChannel,
    band_low: float = DEFAULT_BAND_LOW_HZ,
    band_high: float = DEFAULT_BAND_HIGH_HZ,
    band_order: int = DEFAULT_BAND_ORDER,
    notch_hz: Optional[float] = DEFAULT_NOTCH_HZ,
    notch_q: float = DEFAULT_NOTCH_Q,
    notch_harmonics: int = 1,
) -> EMGChannel:
    """
    Apply band-pass (and optional notch) conditioning to one EMG channel.

    Returns a NEW EMGChannel (the input is not mutated). Set notch_hz=None to
    skip notch filtering.
    """
    if channel.fs <= 0 or len(channel.signal) == 0:
        return replace(channel)

    sig = bandpass_filter(
        channel.signal, channel.fs,
        low=band_low, high=band_high, order=band_order,
    )
    if notch_hz is not None:
        sig = notch_filter(
            sig, channel.fs,
            freq=notch_hz, q=notch_q, harmonics=notch_harmonics,
        )

    return replace(channel, signal=sig)


def preprocess_recording(
    recording: DelsysRecording,
    band_low: float = DEFAULT_BAND_LOW_HZ,
    band_high: float = DEFAULT_BAND_HIGH_HZ,
    band_order: int = DEFAULT_BAND_ORDER,
    notch_hz: Optional[float] = DEFAULT_NOTCH_HZ,
    notch_q: float = DEFAULT_NOTCH_Q,
    notch_harmonics: int = 1,
) -> DelsysRecording:
    """
    Apply preprocess_channel to every EMG channel in a recording.

    Returns a NEW DelsysRecording with conditioned channels (input untouched).
    """
    new_channels = {
        sid: preprocess_channel(
            ch,
            band_low=band_low, band_high=band_high, band_order=band_order,
            notch_hz=notch_hz, notch_q=notch_q, notch_harmonics=notch_harmonics,
        )
        for sid, ch in recording.channels.items()
    }
    return replace(recording, channels=new_channels)
