"""
delsys_parser.py
----------------
Parser for Delsys Trigno Discover wide-CSV exports.

Why a custom parser is needed
-----------------------------
The Trigno Discover CSV is NOT a normal table:

  1. Eight header rows precede the data:
       row 1: Application:, Trigno Discover (x.x.x.x)
       row 2: Date/Time:, ...
       row 3: Collection Length (seconds):, NNN
       row 4: sensor names   ("Avanti Sensor 1 (75923)", then blanks per block)
       row 5: sensor modes   ("sensor mode: 65", ...)
       row 6: channel names  ("EMG 1 Time Series (s)", "EMG 1 (mV)", "ACC X ...", ...)
       row 7: sampling rates (", 1259.2593 Hz, , 148.1481 Hz, ...")
       row 8: sample period  (", 0.000794 s, , 0.00675 s, ...")

  2. Each channel has its OWN time column, because channels sample at
     different rates (EMG ~1259 Hz, ACC/GYRO ~148 Hz). Consequently a single
     CSV *row* does NOT correspond to one instant in time: at row 5000 the EMG
     column is at t=3.96 s while the ACC column is at t=33.7 s. Every
     (time, value) column pair must therefore be read INDEPENDENTLY and have
     its trailing empty cells dropped.

  3. The slower (IMU) channels run out of samples partway down the file,
     leaving empty cells in their columns for the remainder of the rows.

  4. A block of ~22 annotation/event columns may be appended on the right
     (Type, Name, Label, Time, ... RMS, PKPK, ...). These are ignored here.

Design
------
Rather than assume a fixed 14-column stride per sensor, we PARSE the channel
header row and locate every EMG value column by name ("EMG" + "(mV)"). Each
EMG value column's time column is the column immediately to its left, and its
owning sensor is found by forward-filling the sensor-name row. This is robust
to sensors with different channel configurations (e.g. sensor mode 65 vs 50).

This module extracts EMG channels only. ACC/GYRO extraction can be added the
same way if IMU-based analysis (e.g. movement/orientation) is needed later.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


N_HEADER_ROWS = 8          # data begins on row 9 (1-indexed)
SENSOR_ROW    = 3          # 0-indexed: "Avanti Sensor N (id)"
MODE_ROW      = 4          # 0-indexed: "sensor mode: NN"
CHANNEL_ROW   = 5          # 0-indexed: "EMG 1 (mV)", "ACC X (G)", ...
RATE_ROW      = 6          # 0-indexed: "1259.2593 Hz", ...


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EMGChannel:
    """
    One sensor's EMG channel.

    Attributes
    ----------
    sensor_id : str
        Sensor identifier as labelled in the file, e.g. "1", "11", "99".
    sensor_name : str
        Full sensor name from the header, e.g. "Avanti Sensor 1 (75923)".
    time : np.ndarray, shape (N,), float64
        Per-sample timestamps in seconds (independent of other channels).
    signal : np.ndarray, shape (N,), float64
        EMG amplitude in millivolts (mV).
    fs : float
        Sampling rate in Hz, parsed from the header (e.g. 1259.2593).
    """
    sensor_id:   str
    sensor_name: str
    time:        np.ndarray
    signal:      np.ndarray
    fs:          float

    def __len__(self) -> int:
        return len(self.signal)

    @property
    def duration_s(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) else 0.0

    def __repr__(self) -> str:
        return (
            f"EMGChannel(sensor={self.sensor_id!r}, n={len(self):,}, "
            f"fs={self.fs:.1f}Hz, dur={self.duration_s:.1f}s)"
        )


@dataclass
class DelsysRecording:
    """
    A parsed Delsys Trigno Discover recording.

    Attributes
    ----------
    path : Path
        Source CSV path.
    application : str
        Software string from row 1.
    datetime : str
        Recording timestamp from row 2.
    collection_length_s : float
        Declared collection length in seconds (row 3).
    channels : dict[str, EMGChannel]
        EMG channels keyed by sensor_id, in file order.
    """
    path:                 Path
    application:          str
    datetime:             str
    collection_length_s:  float
    channels:             dict[str, EMGChannel] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.channels)

    def sensor_ids(self) -> list[str]:
        return list(self.channels.keys())

    def __repr__(self) -> str:
        return (
            f"DelsysRecording('{self.path.name}', "
            f"{len(self.channels)} EMG channels, "
            f"{self.collection_length_s:.1f}s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Header parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(cell: str) -> str:
    """Strip whitespace and stray carriage returns from a CSV cell."""
    return cell.replace("\r", "").strip()


def _parse_rate(cell: str) -> Optional[float]:
    """Extract a float Hz value from a cell like '1259.2593 Hz'. None if absent."""
    m = re.search(r"([0-9]+\.?[0-9]*)\s*Hz", cell)
    return float(m.group(1)) if m else None


def _forward_fill(row: list[str]) -> list[str]:
    """
    Forward-fill non-empty values across a row.
    The sensor-name row only names the first column of each sensor block; all
    subsequent columns in that block are blank and inherit the sensor name.
    """
    out, last = [], ""
    for cell in row:
        c = _clean(cell)
        if c:
            last = c
        out.append(last)
    return out


def _sensor_id_from_name(name: str) -> str:
    """
    Extract a short sensor id from a name like 'Avanti Sensor 11 (70277)' -> '11'.
    Falls back to the full cleaned name if the pattern is not found.
    """
    m = re.search(r"Sensor\s+(\w+)", name)
    return m.group(1) if m else name


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_delsys_csv(path: str | Path) -> DelsysRecording:
    """
    Parse a Delsys Trigno Discover CSV into a DelsysRecording of EMG channels.

    Parameters
    ----------
    path : str | Path
        Path to the .csv export.

    Returns
    -------
    DelsysRecording
        Metadata + one EMGChannel per sensor (EMG channels only).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the header structure is not recognised (no EMG columns found).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # ── 1. Read the 8 header rows ─────────────────────────────────────────────
    header_rows: list[list[str]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            header_rows.append(row)
            if i >= N_HEADER_ROWS - 1:
                break

    if len(header_rows) < N_HEADER_ROWS:
        raise ValueError(f"File too short to be a Delsys export: {path}")

    application = _clean(header_rows[0][1]) if len(header_rows[0]) > 1 else ""
    datetime    = _clean(header_rows[1][1]) if len(header_rows[1]) > 1 else ""
    try:
        collection_length = float(_clean(header_rows[2][1]))
    except (IndexError, ValueError):
        collection_length = 0.0

    sensor_names = _forward_fill(header_rows[SENSOR_ROW])
    channel_row  = [_clean(c) for c in header_rows[CHANNEL_ROW]]
    rate_row     = header_rows[RATE_ROW]

    # ── 2. Locate EMG value columns by name ───────────────────────────────────
    # An EMG value column's header contains "EMG" and "(mV)" (not "Time Series").
    emg_value_cols: list[int] = []
    for idx, name in enumerate(channel_row):
        if "EMG" in name and "mV" in name and "Time Series" not in name:
            emg_value_cols.append(idx)

    if not emg_value_cols:
        raise ValueError(
            f"No EMG (mV) columns found in header of {path.name}. "
            "Is this a Trigno Discover export?"
        )

    # For each EMG value column, its time column is the column to its left.
    col_specs = []  # (sensor_id, sensor_name, time_col, value_col, fs)
    for vcol in emg_value_cols:
        tcol = vcol - 1
        # The sensor-name row may be truncated by trailing empty cells. Since it
        # is forward-filled, any column past its end still belongs to the LAST
        # named sensor, so fall back to the last entry rather than a placeholder.
        if vcol < len(sensor_names):
            sensor_name = sensor_names[vcol]
        elif sensor_names:
            sensor_name = sensor_names[-1]
        else:
            sensor_name = f"col{vcol}"
        sensor_id   = _sensor_id_from_name(sensor_name)
        fs          = _parse_rate(rate_row[vcol]) if vcol < len(rate_row) else None
        col_specs.append((sensor_id, sensor_name, tcol, vcol, fs))

    # ── 3. Stream the data rows, collecting each EMG column independently ──────
    max_col = max(vcol for _, _, _, vcol, _ in col_specs)
    time_acc:  dict[int, list[float]] = {tcol: [] for _, _, tcol, _, _ in col_specs}
    value_acc: dict[int, list[float]] = {vcol: [] for _, _, _, vcol, _ in col_specs}

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for _ in range(N_HEADER_ROWS):      # skip header
            next(reader, None)
        for row in reader:
            if len(row) <= max_col:
                continue
            for _, _, tcol, vcol, _ in col_specs:
                t_raw = row[tcol].replace("\r", "").strip()
                v_raw = row[vcol].replace("\r", "").strip()
                # Slower channels run out of samples → empty cells. Skip those.
                if t_raw == "" or v_raw == "":
                    continue
                try:
                    t = float(t_raw)
                    v = float(v_raw)
                except ValueError:
                    continue
                time_acc[tcol].append(t)
                value_acc[vcol].append(v)

    # ── 4. Build EMGChannel objects ───────────────────────────────────────────
    channels: dict[str, EMGChannel] = {}
    for sensor_id, sensor_name, tcol, vcol, fs in col_specs:
        t = np.asarray(time_acc[tcol], dtype=np.float64)
        v = np.asarray(value_acc[vcol], dtype=np.float64)

        # Fall back to estimating fs from timestamps if the header lacked it.
        if fs is None and len(t) > 1:
            dt = np.median(np.diff(t))
            fs = float(1.0 / dt) if dt > 0 else 0.0
        elif fs is None:
            fs = 0.0

        channels[sensor_id] = EMGChannel(
            sensor_id=sensor_id,
            sensor_name=sensor_name,
            time=t,
            signal=v,
            fs=fs,
        )

    return DelsysRecording(
        path=path,
        application=application,
        datetime=datetime,
        collection_length_s=collection_length,
        channels=channels,
    )
