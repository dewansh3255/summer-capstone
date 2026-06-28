"""
Unit tests for Phase C: onset detection + plotting.

Onset tests use a synthetic burst signal (silence -> burst -> silence) with a
known activation window. Plotting tests verify files are written (headless Agg
backend), without asserting on pixel content.
"""

import numpy as np
import pandas as pd
import pytest

from emg_pipeline.delsys_parser import EMGChannel
from emg_pipeline.analysis import (
    teager_kaiser_energy, linear_envelope, detect_onsets, ActivationEvent,
    plot_rms_trend, plot_mdf_trend, plot_fatigue_ranking, plot_channel_overview,
)
from emg_pipeline.features import extract_channel_features


FS = 1000.0


def _burst_channel(sid="1"):
    """
    3-second signal: 1 s silence, 1 s 80 Hz burst, 1 s silence.
    Small baseline noise so the adaptive threshold has non-zero std.
    """
    n = int(3 * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(0)
    sig = 0.01 * rng.standard_normal(n)            # baseline noise
    burst = (t >= 1.0) & (t < 2.0)
    sig[burst] += np.sin(2 * np.pi * 80 * t[burst])  # activation
    return EMGChannel(sid, f"Sensor {sid}", t, sig, FS)


# ──────────────────────────────────────────────────────────────────────────────
# TKEO + envelope
# ──────────────────────────────────────────────────────────────────────────────

class TestTKEO:
    def test_length_preserved(self):
        x = np.sin(np.linspace(0, 10, 500))
        assert len(teager_kaiser_energy(x)) == len(x)

    def test_short_signal(self):
        assert np.all(teager_kaiser_energy(np.array([1.0, 2.0])) == 0)

    def test_higher_for_higher_freq(self):
        t = np.arange(1000) / FS
        lo = np.sin(2 * np.pi * 20 * t)
        hi = np.sin(2 * np.pi * 200 * t)
        # TKEO mean energy grows with frequency for equal amplitude
        assert teager_kaiser_energy(hi).mean() > teager_kaiser_energy(lo).mean()


class TestEnvelope:
    def test_envelope_nonnegative(self):
        x = np.sin(np.linspace(0, 50, 3000))
        env = linear_envelope(x, FS)
        assert np.all(env >= -1e-9)

    def test_envelope_length(self):
        x = np.sin(np.linspace(0, 50, 3000))
        assert len(linear_envelope(x, FS)) == len(x)

    def test_envelope_tracks_amplitude(self):
        t = np.arange(3000) / FS
        x = np.concatenate([np.zeros(1000), np.sin(2 * np.pi * 80 * t[:1000]), np.zeros(1000)])
        env = linear_envelope(x, FS)
        # envelope larger in the burst region than the silent region
        assert env[1500] > env[200]


# ──────────────────────────────────────────────────────────────────────────────
# Onset detection
# ──────────────────────────────────────────────────────────────────────────────

class TestOnsetDetection:
    def test_detects_single_activation(self):
        ev = detect_onsets(_burst_channel())
        assert len(ev) >= 1

    def test_activation_window_correct(self):
        ev = detect_onsets(_burst_channel())
        # The dominant event should overlap [1.0, 2.0] s
        main = max(ev, key=lambda e: e.duration_s)
        assert main.onset_s == pytest.approx(1.0, abs=0.1)
        assert main.offset_s == pytest.approx(2.0, abs=0.1)

    def test_returns_activation_events(self):
        ev = detect_onsets(_burst_channel())
        assert all(isinstance(e, ActivationEvent) for e in ev)

    def test_no_activation_in_silence(self):
        n = int(2 * FS)
        t = np.arange(n) / FS
        sig = 0.01 * np.random.default_rng(1).standard_normal(n)
        ch = EMGChannel("1", "S1", t, sig, FS)
        ev = detect_onsets(ch, k=4.0)
        # No sustained activation in pure noise
        assert len(ev) == 0

    def test_explicit_threshold(self):
        ch = _burst_channel()
        ev = detect_onsets(ch, threshold=0.05)
        assert len(ev) >= 1

    def test_min_duration_filters_short_activation(self):
        # The burst is ~1 s long. Requiring a 2 s minimum rejects it; a 0.5 s
        # minimum keeps it. This exercises the min-duration logic at a scale
        # above the envelope's time resolution (~100 ms for a 5 Hz envelope).
        ch = _burst_channel()
        assert len(detect_onsets(ch, min_duration_ms=2000)) == 0
        assert len(detect_onsets(ch, min_duration_ms=500)) >= 1

    def test_empty_channel(self):
        ch = EMGChannel("1", "S1", np.array([]), np.array([]), FS)
        assert detect_onsets(ch) == []

    def test_event_duration_property(self):
        e = ActivationEvent(1.0, 2.5)
        assert e.duration_s == pytest.approx(1.5)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (file creation)
# ──────────────────────────────────────────────────────────────────────────────

class TestPlotting:
    def _feats(self):
        return {
            "1": extract_channel_features(_burst_channel("1")),
            "2": extract_channel_features(_burst_channel("2")),
        }

    def test_plot_rms_trend(self, tmp_path):
        out = plot_rms_trend(self._feats(), tmp_path / "rms.png")
        assert out.exists() and out.stat().st_size > 0

    def test_plot_mdf_trend(self, tmp_path):
        out = plot_mdf_trend(self._feats(), tmp_path / "mdf.png")
        assert out.exists() and out.stat().st_size > 0

    def test_plot_fatigue_ranking(self, tmp_path):
        df = pd.DataFrame({
            "sensor_id": ["1", "2", "3"],
            "mdf_slope": [-0.5, -0.1, 0.2],
            "fatigue_detected": [True, True, False],
        })
        out = plot_fatigue_ranking(df, tmp_path / "rank.png")
        assert out.exists() and out.stat().st_size > 0

    def test_plot_channel_overview(self, tmp_path):
        ch = _burst_channel()
        ev = detect_onsets(ch)
        out = plot_channel_overview(ch, tmp_path / "overview.png", events=ev)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_channel_overview_max_seconds(self, tmp_path):
        ch = _burst_channel()
        out = plot_channel_overview(ch, tmp_path / "ov2.png", max_seconds=1.0)
        assert out.exists()

    def test_creates_parent_dir(self, tmp_path):
        out = plot_rms_trend(self._feats(), tmp_path / "nested" / "deep" / "rms.png")
        assert out.exists()
