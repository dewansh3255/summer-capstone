"""
Unit tests for EMG preprocessing (filtering).

Uses synthetic signals with known spectral content so filter behaviour can be
asserted precisely, with no dependency on the real recordings.
"""

import numpy as np
import pytest

from emg_pipeline.delsys_parser import EMGChannel
from emg_pipeline.preprocess import (
    bandpass_filter,
    notch_filter,
    preprocess_channel,
)


FS = 1259.0


def _sine(freq, n, fs=FS, amp=1.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def _power_at(sig, fs, f0, bw=2.0):
    from numpy.fft import rfft, rfftfreq
    spec = np.abs(rfft(sig)) ** 2
    freqs = rfftfreq(len(sig), 1 / fs)
    mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    return spec[mask].mean() if mask.any() else 0.0


class TestBandpass:
    def test_removes_dc_offset(self):
        sig = _sine(100, 5000) + 5.0          # 5 mV DC offset
        out = bandpass_filter(sig, FS)
        assert abs(out.mean()) < 0.05

    def test_passes_inband_frequency(self):
        sig = _sine(100, 5000)                # 100 Hz is in [20,450]
        out = bandpass_filter(sig, FS)
        # In-band power largely retained
        assert _power_at(out, FS, 100) > 0.25 * _power_at(sig, FS, 100)

    def test_attenuates_low_frequency(self):
        low = _sine(5, 5000)                  # 5 Hz below 20 Hz passband
        out = bandpass_filter(low, FS)
        assert _power_at(out, FS, 5) < 0.1 * _power_at(low, FS, 5)

    def test_length_preserved(self):
        sig = _sine(100, 5000)
        assert len(bandpass_filter(sig, FS)) == len(sig)

    def test_output_float64(self):
        sig = _sine(100, 5000)
        assert bandpass_filter(sig, FS).dtype == np.float64

    def test_short_signal_returns_copy(self):
        sig = np.array([1.0, 2.0, 3.0])
        out = bandpass_filter(sig, FS)
        assert len(out) == 3

    def test_invalid_band_raises(self):
        sig = _sine(100, 5000)
        with pytest.raises(ValueError):
            bandpass_filter(sig, FS, low=450, high=20)


class TestNotch:
    def test_attenuates_50hz(self):
        sig = _sine(50, 5000) + _sine(150, 5000)
        out = notch_filter(sig, FS, freq=50)
        # 50 Hz strongly attenuated, 150 Hz preserved
        assert _power_at(out, FS, 50) < 0.2 * _power_at(sig, FS, 50)
        assert _power_at(out, FS, 150) > 0.5 * _power_at(sig, FS, 150)

    def test_harmonics(self):
        sig = _sine(50, 5000) + _sine(100, 5000)
        out = notch_filter(sig, FS, freq=50, harmonics=2)
        assert _power_at(out, FS, 50) < 0.2 * _power_at(sig, FS, 50)
        assert _power_at(out, FS, 100) < 0.2 * _power_at(sig, FS, 100)

    def test_length_preserved(self):
        sig = _sine(50, 5000)
        assert len(notch_filter(sig, FS)) == len(sig)


class TestPreprocessChannel:
    def _channel(self, sig):
        t = np.arange(len(sig)) / FS
        return EMGChannel(
            sensor_id="1", sensor_name="Test", time=t, signal=sig, fs=FS
        )

    def test_returns_new_channel(self):
        ch = self._channel(_sine(100, 5000))
        out = preprocess_channel(ch)
        assert out is not ch
        assert out.signal is not ch.signal

    def test_input_not_mutated(self):
        sig = _sine(100, 5000) + 5.0
        ch = self._channel(sig.copy())
        _ = preprocess_channel(ch)
        np.testing.assert_allclose(ch.signal, sig)   # original untouched

    def test_metadata_preserved(self):
        ch = self._channel(_sine(100, 5000))
        out = preprocess_channel(ch)
        assert out.sensor_id == ch.sensor_id
        assert out.fs == ch.fs
        assert len(out) == len(ch)

    def test_removes_offset_and_5060(self):
        sig = _sine(100, 5000) + _sine(50, 5000) + 3.0
        ch = self._channel(sig)
        out = preprocess_channel(ch)
        assert abs(out.signal.mean()) < 0.05            # DC gone
        assert _power_at(out.signal, FS, 50) < 0.3 * _power_at(sig, FS, 50)

    def test_notch_can_be_disabled(self):
        sig = _sine(50, 5000)
        ch = self._channel(sig)
        out = preprocess_channel(ch, notch_hz=None)
        # Without notch, 50 Hz (in passband) is largely retained
        assert _power_at(out.signal, FS, 50) > 0.25 * _power_at(sig, FS, 50)

    def test_empty_channel_safe(self):
        ch = EMGChannel("1", "Test", np.array([]), np.array([]), FS)
        out = preprocess_channel(ch)
        assert len(out) == 0
