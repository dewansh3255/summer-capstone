"""
summary.py
----------
Phase D (part 1): turn the dense feature tables into a compact, structured
summary that an LLM can interpret.

Why summarise?
--------------
A recording has millions of raw samples and thousands of feature windows per
sensor. An LLM cannot (and should not) read that. Instead we distil each sensor
down to a handful of decision-relevant numbers — start/end RMS, %MVC, the MDF
trend, the fatigue flag — exactly the quantities a domain expert would read off
a report. The LLM then reasons over this small JSON, not the raw signal.

Crucially, every number here is COMPUTED by our pipeline (Phases B/C). The LLM
never invents values; it only interprets these. That keeps the output
scientifically defensible.

The sensor->muscle mapping is optional: pass a `labels` dict to replace sensor
ids with muscle names; otherwise "Sensor <id>" is used.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .features import ChannelFeatures, compute_fatigue_metrics


def build_sensor_summary(
    features: ChannelFeatures,
    label: Optional[str] = None,
    activation_count: Optional[int] = None,
) -> dict:
    """
    Compact per-sensor summary dict (JSON-serialisable).

    Parameters
    ----------
    features : ChannelFeatures
        Windowed features for one sensor (Phase B).
    label : str, optional
        Muscle name to display instead of "Sensor <id>".
    activation_count : int, optional
        Number of detected activations (from analysis.detect_onsets), if
        available. Included only when provided.
    """
    fm = compute_fatigue_metrics(features)
    df = features.df

    amplitude = {
        "rms_start_mv": round(fm.rms_start, 5),
        "rms_end_mv": round(fm.rms_end, 5),
        "rms_pct_change": round(fm.rms_pct_change, 1),
        "rms_slope_mv_per_s": round(fm.rms_slope, 7),
    }
    if "pct_mvc" in df.columns and len(df):
        amplitude["mean_pct_mvc"] = round(float(df["pct_mvc"].mean()), 1)
        amplitude["peak_pct_mvc"] = round(float(df["pct_mvc"].max()), 1)

    summary = {
        "sensor_id": features.sensor_id,
        "label": label or f"Sensor {features.sensor_id}",
        "duration_s": round(fm.duration_s, 1),
        "n_windows": fm.n_windows,
        "amplitude": amplitude,
        "frequency": {
            "mdf_start_hz": round(fm.mdf_start, 1),
            "mdf_end_hz": round(fm.mdf_end, 1),
            "mdf_pct_change": round(fm.mdf_pct_change, 1),
            "mdf_slope_hz_per_s": round(fm.mdf_slope, 5),
            "mnf_slope_hz_per_s": round(fm.mnf_slope, 5),
        },
        "fatigue_detected": bool(fm.fatigue_detected),
    }
    if activation_count is not None:
        summary["activation_count"] = int(activation_count)

    return summary


def build_recording_summary(
    feats: dict[str, ChannelFeatures],
    labels: Optional[dict[str, str]] = None,
    activation_counts: Optional[dict[str, int]] = None,
    task_name: Optional[str] = None,
) -> dict:
    """
    Build the full recording summary across all sensors.

    Parameters
    ----------
    feats : dict[sensor_id, ChannelFeatures]
        Per-sensor features (Phase B output).
    labels : dict[sensor_id, str], optional
        Optional sensor->muscle name mapping.
    activation_counts : dict[sensor_id, int], optional
        Optional per-sensor activation counts (Phase C).
    task_name : str, optional
        Human-readable task name (e.g. "push task").

    Returns
    -------
    dict
        {
          task, n_sensors, n_fatigued, fatigued_sensors,
          most_fatigued (label with steepest negative MDF slope),
          sensors: [ per-sensor summary, ... ]
        }
    """
    labels = labels or {}
    activation_counts = activation_counts or {}

    sensors = [
        build_sensor_summary(
            cf,
            label=labels.get(sid),
            activation_count=activation_counts.get(sid),
        )
        for sid, cf in feats.items()
    ]

    fatigued = [s for s in sensors if s["fatigue_detected"]]

    # Most fatigued = steepest negative MDF slope among all sensors.
    most_fatigued = None
    if sensors:
        steepest = min(sensors, key=lambda s: s["frequency"]["mdf_slope_hz_per_s"])
        if steepest["frequency"]["mdf_slope_hz_per_s"] < 0:
            most_fatigued = steepest["label"]

    summary = {
        "n_sensors": len(sensors),
        "n_fatigued": len(fatigued),
        "fatigued_sensors": [s["label"] for s in fatigued],
        "most_fatigued": most_fatigued,
        "sensors": sensors,
    }
    if task_name:
        summary = {"task": task_name, **summary}
    return summary
