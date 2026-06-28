"""
Unit tests for feature extraction and fatigue-trend metrics.

Uses synthetic signals with known mathematical properties so each feature and
the fatigue detector can be asserted precisely, with no dependency on the real
recordings.
"""

import numpy as np
import pandas as pd
import pytest

from emg_pipeline.delsys_parser import EMGChannel
from emg_pipeline.features import (
    rms, mav, iemg, waveform_length, zero_crossings,
    power_spectrum, mean_frequency, median_frequency,
    extract_channel_features, extract_recording_features,
    compute_mvc_reference, combine_mvc_references,
    compute_fatigue_metrics, fatigue_summary_frame,
    ChannelFeatures, FatigueMetrics, FEATURE_COLUMNS,
)
from emg_pipeline.delsys_parser import DelsysRecording


FS = 1000.0


def _sine(freq, dur_s, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(int(dur_s * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _channel(sig, fs=FS, sid="1"):
    t = np.arange(len(sig)) / fs
    return EMGChannel(sensor_id=sid, sensor_name=f"Sensor {sid}",
                      time=t, signal=sig, fs=fs)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level time-domain features
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeDomain:
    def test_rms_of_sine(self):
        # RMS of a sine of amplitude A is A/sqrt(2)
        x = _sine(50, 1.0, amp=2.0)
        assert rms(x) == pytest.approx(2.0 / np.sqrt(2), rel=1e-2)

    def test_mav_of_sine(self):
        # Mean abs value of a sine of amplitude A is 2A/pi
        x = _sine(50, 1.0, amp=1.0)
        assert mav(x) == pytest.approx(2.0 / np.pi, rel=1e-2)

    def test_rms_zero_signal(self):
        assert rms(np.zeros(100)) == 0.0

    def test_iemg_constant(self):
        x = np.ones(100)
        assert iemg(x) == pytest.approx(100.0)

    def test_waveform_length_ramp(self):
        x = np.arange(11, dtype=float)          # diffs all = 1, 10 of them
        assert waveform_length(x) == pytest.approx(10.0)

    def test_waveform_length_short(self):
        assert waveform_length(np.array([1.0])) == 0.0

    def test_zero_crossings_sine(self):
        # A 5 Hz sine over 1 s crosses zero ~10 times
        x = _sine(5, 1.0, amp=1.0)
        zc = zero_crossings(x, 0.0)
        assert 8 <= zc <= 12

    def test_zero_crossings_constant(self):
        assert zero_crossings(np.ones(100)) == 0

    def test_empty_inputs(self):
        e = np.array([])
        assert rms(e) == 0.0
        assert mav(e) == 0.0
        assert iemg(e) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Frequency-domain features
# ──────────────────────────────────────────────────────────────────────────────

class TestFrequencyDomain:
    def test_mnf_near_sine_frequency(self):
        x = _sine(100, 2.0, amp=1.0)
        freqs, psd = power_spectrum(x, FS)
        assert mean_frequency(freqs, psd) == pytest.approx(100, abs=8)

    def test_mdf_near_sine_frequency(self):
        x = _sine(100, 2.0, amp=1.0)
        freqs, psd = power_spectrum(x, FS)
        assert median_frequency(freqs, psd) == pytest.approx(100, abs=8)

    def test_higher_freq_has_higher_mnf(self):
        f1, p1 = power_spectrum(_sine(80, 2.0), FS)
        f2, p2 = power_spectrum(_sine(200, 2.0), FS)
        assert mean_frequency(f2, p2) > mean_frequency(f1, p1)

    def test_power_spectrum_empty(self):
        freqs, psd = power_spectrum(np.array([1.0]), FS)
        assert len(freqs) == 0 and len(psd) == 0

    def test_zero_power_returns_zero(self):
        freqs = np.array([1.0, 2.0, 3.0])
        psd = np.zeros(3)
        assert mean_frequency(freqs, psd) == 0.0
        assert median_frequency(freqs, psd) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Channel feature extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractChannelFeatures:
    def test_returns_channelfeatures(self):
        cf = extract_channel_features(_channel(_sine(100, 5.0)))
        assert isinstance(cf, ChannelFeatures)

    def test_columns_present(self):
        cf = extract_channel_features(_channel(_sine(100, 5.0)))
        for col in FEATURE_COLUMNS:
            assert col in cf.df.columns

    def test_window_count(self):
        # 5 s signal, 250 ms window, 50% overlap (125 ms step):
        # windows = floor((5000 - 250) / 125) + 1 = 39
        cf = extract_channel_features(
            _channel(_sine(100, 5.0)), window_ms=250, overlap=0.5
        )
        assert len(cf) == 39

    def test_t_center_increases(self):
        cf = extract_channel_features(_channel(_sine(100, 5.0)))
        t = cf.df["t_center"].values
        assert np.all(np.diff(t) > 0)

    def test_mvc_normalisation_column(self):
        cf = extract_channel_features(
            _channel(_sine(100, 5.0, amp=1.0)), mvc_reference=1.0
        )
        assert "pct_mvc" in cf.df.columns
        # RMS of unit sine ~0.707 -> ~70.7 %MVC
        assert cf.df["pct_mvc"].mean() == pytest.approx(70.7, abs=5)

    def test_no_mvc_no_column(self):
        cf = extract_channel_features(_channel(_sine(100, 5.0)))
        assert "pct_mvc" not in cf.df.columns

    def test_short_signal_few_windows(self):
        cf = extract_channel_features(_channel(_sine(100, 0.1)))
        assert len(cf) >= 0   # should not crash


# ──────────────────────────────────────────────────────────────────────────────
# MVC reference
# ──────────────────────────────────────────────────────────────────────────────

class TestMVC:
    def _recording(self, channels):
        return DelsysRecording(
            path=__import__("pathlib").Path("x.csv"),
            application="test", datetime="now",
            collection_length_s=1.0, channels=channels,
        )

    def test_mvc_reference_per_sensor(self):
        rec = self._recording({
            "1": _channel(_sine(100, 3.0, amp=2.0), sid="1"),
            "2": _channel(_sine(100, 3.0, amp=0.5), sid="2"),
        })
        refs = compute_mvc_reference(rec)
        # Sensor 1 had bigger amplitude -> bigger MVC RMS
        assert refs["1"] > refs["2"]
        assert refs["1"] == pytest.approx(2.0 / np.sqrt(2), rel=0.1)

    def test_combine_takes_max(self):
        left = {"1": 1.0, "2": 3.0}
        right = {"1": 2.0, "2": 1.0}
        combined = combine_mvc_references(left, right)
        assert combined["1"] == 2.0
        assert combined["2"] == 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Fatigue-trend metrics  (the core scientific claim)
# ──────────────────────────────────────────────────────────────────────────────

class TestFatigueMetrics:
    def _fatiguing_signal(self):
        """
        Synthesize the canonical fatigue signature:
          - frequency decreasing 150 Hz -> 80 Hz over the recording (MDF falls)
          - amplitude increasing 1.0 -> 2.0 over the recording (RMS rises)
        """
        fs = FS
        dur = 10.0
        n = int(dur * fs)
        t = np.arange(n) / fs
        freq = np.linspace(150, 80, n)          # falling instantaneous freq
        amp = np.linspace(1.0, 2.0, n)          # rising amplitude
        phase = 2 * np.pi * np.cumsum(freq) / fs
        return amp * np.sin(phase), fs

    def test_detects_fatigue(self):
        sig, fs = self._fatiguing_signal()
        cf = extract_channel_features(_channel(sig, fs=fs))
        fm = compute_fatigue_metrics(cf)
        assert isinstance(fm, FatigueMetrics)
        assert fm.rms_slope > 0          # amplitude rising
        assert fm.mdf_slope < 0          # spectrum compressing
        assert fm.fatigue_detected is True

    def test_rms_pct_change_positive(self):
        sig, fs = self._fatiguing_signal()
        cf = extract_channel_features(_channel(sig, fs=fs))
        fm = compute_fatigue_metrics(cf)
        assert fm.rms_pct_change > 0

    def test_mdf_pct_change_negative(self):
        sig, fs = self._fatiguing_signal()
        cf = extract_channel_features(_channel(sig, fs=fs))
        fm = compute_fatigue_metrics(cf)
        assert fm.mdf_pct_change < 0

    def test_no_fatigue_on_stationary_signal(self):
        # Constant amplitude + constant frequency -> no fatigue trend
        sig = _sine(100, 10.0, amp=1.0)
        cf = extract_channel_features(_channel(sig))
        fm = compute_fatigue_metrics(cf)
        assert fm.fatigue_detected is False

    def test_metrics_dict(self):
        sig, fs = self._fatiguing_signal()
        cf = extract_channel_features(_channel(sig, fs=fs))
        d = compute_fatigue_metrics(cf).as_dict()
        assert "rms_slope" in d and "mdf_slope" in d
        assert "fatigue_detected" in d

    def test_too_few_windows_safe(self):
        cf = extract_channel_features(_channel(_sine(100, 0.05)))
        fm = compute_fatigue_metrics(cf)
        assert fm.fatigue_detected is False


# ──────────────────────────────────────────────────────────────────────────────
# Recording-level helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestRecordingLevel:
    def _recording(self, channels):
        return DelsysRecording(
            path=__import__("pathlib").Path("x.csv"),
            application="test", datetime="now",
            collection_length_s=1.0, channels=channels,
        )

    def test_extract_recording_features(self):
        rec = self._recording({
            "1": _channel(_sine(100, 3.0), sid="1"),
            "2": _channel(_sine(120, 3.0), sid="2"),
        })
        feats = extract_recording_features(rec)
        assert set(feats.keys()) == {"1", "2"}
        assert all(isinstance(v, ChannelFeatures) for v in feats.values())

    def test_fatigue_summary_frame_sorted(self):
        rec = self._recording({
            "1": _channel(_sine(100, 3.0), sid="1"),
            "2": _channel(_sine(120, 3.0), sid="2"),
        })
        feats = extract_recording_features(rec)
        summary = fatigue_summary_frame(feats)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        # sorted by mdf_slope ascending
        assert summary["mdf_slope"].is_monotonic_increasing
