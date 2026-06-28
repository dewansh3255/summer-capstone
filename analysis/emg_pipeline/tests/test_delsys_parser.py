"""
Unit tests for the Delsys Trigno Discover parser.

A tiny synthetic CSV that reproduces the real export's structure is written to
a temp file, so these tests do not depend on the (gitignored) real recordings.
The synthetic file has 2 sensors, each with an EMG channel (fast rate) and one
ACC channel (slow rate) so the multi-rate / trailing-empty-cell behaviour is
exercised.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from emg_pipeline.delsys_parser import parse_delsys_csv, EMGChannel, DelsysRecording


def _write_synthetic_csv(path: Path) -> None:
    """
    Build a 2-sensor Delsys-style CSV.

    Layout per sensor block (4 columns):
        EMG time, EMG (mV), ACC X time, ACC X (G)
    EMG has 6 samples; ACC has 3 samples (slower), so the last 3 ACC cells are
    empty — mirroring the real file where IMU channels run out of samples.
    """
    header = [
        "Application:, Trigno Discover (2.1.0.7)",
        "Date/Time:, 01-01-2026 10:00:00",
        "Collection Length (seconds):, 0.005",
        # sensor-name row: name only at each block start, rest blank
        "Avanti Sensor 1 (111), , , , Avanti Sensor 2 (222), , , ",
        "sensor mode: 65, , , , sensor mode: 50, , , ",
        # channel-name row
        "EMG 1 Time Series (s), EMG 1 (mV), ACC X Time Series (s), ACC X (G), "
        "EMG 1 Time Series (s), EMG 1 (mV), ACC X Time Series (s), ACC X (G)",
        # rate row
        ", 1000.0 Hz, , 500.0 Hz, , 1000.0 Hz, , 500.0 Hz",
        # period row
        ", 0.001 s, , 0.002 s, , 0.001 s, , 0.002 s",
    ]
    # 6 EMG samples; ACC only first 3 rows populated
    emg1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    emg1_v = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
    emg2_v = [0.5, -0.5, 0.6, -0.6, 0.7, -0.7]
    acc_t = [0.0, 0.002, 0.004]
    acc_v = [9.8, 9.7, 9.6]

    data_rows = []
    for i in range(6):
        s1_acc_t = acc_t[i] if i < 3 else ""
        s1_acc_v = acc_v[i] if i < 3 else ""
        row = [
            emg1[i], emg1_v[i], s1_acc_t, s1_acc_v,      # sensor 1
            emg1[i], emg2_v[i], s1_acc_t, s1_acc_v,      # sensor 2
        ]
        data_rows.append(", ".join(str(x) for x in row))

    path.write_text("\n".join(header + data_rows) + "\n")


@pytest.fixture
def synthetic_csv(tmp_path) -> Path:
    p = tmp_path / "synthetic_delsys.csv"
    _write_synthetic_csv(p)
    return p


class TestParsing:
    def test_returns_recording(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        assert isinstance(rec, DelsysRecording)

    def test_metadata(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        assert "Trigno Discover" in rec.application
        assert rec.collection_length_s == pytest.approx(0.005)

    def test_two_sensors_found(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        assert len(rec) == 2
        assert rec.sensor_ids() == ["1", "2"]

    def test_channels_are_emgchannel(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        for ch in rec.channels.values():
            assert isinstance(ch, EMGChannel)

    def test_emg_values_correct(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        np.testing.assert_allclose(
            rec.channels["1"].signal, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        )
        np.testing.assert_allclose(
            rec.channels["2"].signal, [0.5, -0.5, 0.6, -0.6, 0.7, -0.7]
        )

    def test_emg_times_correct(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        np.testing.assert_allclose(
            rec.channels["1"].time, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )

    def test_sampling_rate_parsed(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        assert rec.channels["1"].fs == pytest.approx(1000.0)

    def test_all_emg_samples_kept(self, synthetic_csv):
        """EMG has 6 samples; none should be dropped despite ACC running out."""
        rec = parse_delsys_csv(synthetic_csv)
        assert len(rec.channels["1"]) == 6
        assert len(rec.channels["2"]) == 6

    def test_sensor_name_preserved(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        assert "Sensor 1" in rec.channels["1"].sensor_name
        assert "111" in rec.channels["1"].sensor_name

    def test_duration_property(self, synthetic_csv):
        rec = parse_delsys_csv(synthetic_csv)
        assert rec.channels["1"].duration_s == pytest.approx(0.5)


class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_delsys_csv(tmp_path / "nope.csv")

    def test_too_short_raises(self, tmp_path):
        p = tmp_path / "short.csv"
        p.write_text("Application:, x\nDate:, y\n")
        with pytest.raises(ValueError):
            parse_delsys_csv(p)

    def test_no_emg_columns_raises(self, tmp_path):
        p = tmp_path / "noemg.csv"
        lines = [
            "Application:, Trigno Discover",
            "Date/Time:, x",
            "Collection Length (seconds):, 1.0",
            "Avanti Sensor 1 (111), ",
            "sensor mode: 50, ",
            "ACC X Time Series (s), ACC X (G)",
            ", 500 Hz",
            ", 0.002 s",
            "0.0, 9.8",
        ]
        p.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="No EMG"):
            parse_delsys_csv(p)


class TestRealData:
    """Integration tests against the real recordings, skipped if absent."""

    DATA = Path(__file__).resolve().parents[2] / "data"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[2] / "data" / "Avnish_push1_01.csv").exists(),
        reason="real Delsys data not present",
    )
    def test_real_push_file(self):
        rec = parse_delsys_csv(self.DATA / "Avnish_push1_01.csv")
        assert len(rec) == 16
        assert rec.sensor_ids()[-1] == "99"
        ch = rec.channels["1"]
        assert ch.fs == pytest.approx(1259.26, abs=1.0)
        assert len(ch) > 400_000
