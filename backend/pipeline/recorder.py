"""
recorder.py
-----------
Saves gesture recording sessions to disk and loads them back.

This module is the hardware-independence layer for the pipeline.
Once a recording session is saved, the entire Chamfer Distance,
thresholding, and classification pipeline can be run and re-run
offline without the Quest headset.

Directory layout on disk
------------------------
    recordings/
    └── <session_name>/               e.g. "2026-06-18_ThumbsUp"
        ├── session.json              metadata (gesture label, timestamps, fps)
        └── frames/
            ├── 000000.npy            one file per captured frame (N,3) float32
            ├── 000001.npy
            └── ...

Usage — recording (called from server.py during a live session)
---------------------------------------------------------------
    from pipeline.recorder import Recorder

    rec = Recorder("recordings", gesture_label="ThumbsUp")
    rec.start()

    # inside the WebSocket frame loop:
    if pc is not None:
        rec.add(pc)

    rec.stop()   # flushes metadata, closes session

Usage — playback (offline, no Quest needed)
-------------------------------------------
    from pipeline.recorder import load_session, list_sessions

    sessions = list_sessions("recordings")
    clouds   = load_session("recordings/2026-06-18_ThumbsUp")
    # clouds is a list of (N,3) float32 numpy arrays
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.point_cloud import PointCloud


# ──────────────────────────────────────────────────────────────────────────────
# Recorder — used live during a WebSocket session
# ──────────────────────────────────────────────────────────────────────────────

class Recorder:
    """
    Records a sequence of PointCloud frames for one gesture to disk.

    Parameters
    ----------
    base_dir : str | Path
        Root directory where all sessions are stored.
    gesture_label : str
        Human-readable gesture name, e.g. "ThumbsUp", "ASL_L", "Spock".
        Used in the session folder name and stored in metadata.
    session_name : str, optional
        Override the auto-generated session name.
        Default: "<YYYY-MM-DD_HHMMSS>_<gesture_label>"
    """

    def __init__(
        self,
        base_dir: str | Path,
        gesture_label: str,
        session_name: Optional[str] = None,
    ):
        self.gesture_label = gesture_label
        self._base_dir = Path(base_dir)

        if session_name is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            session_name = f"{ts}_{gesture_label}"

        self.session_dir = self._base_dir / session_name
        self._frames_dir = self.session_dir / "frames"

        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._stop_time:  Optional[float] = None
        self._active = False

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Create directories and begin recording."""
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.monotonic()
        self._active = True

    def add(self, pc: PointCloud) -> None:
        """
        Save one PointCloud frame to disk.
        Only the normalised (N,3) point array is saved — metadata lives
        in session.json.  This keeps individual files tiny (<1 KB each).

        Raises RuntimeError if called before start() or after stop().
        """
        if not self._active:
            raise RuntimeError("Recorder.add() called outside of start()/stop() block.")

        filename = self._frames_dir / f"{self._frame_count:06d}.npy"
        np.save(str(filename), pc.points)
        self._frame_count += 1

    def stop(self) -> dict:
        """
        Stop recording, write session.json, and return the metadata dict.
        Safe to call multiple times (idempotent after first call).
        """
        if not self._active:
            return {}

        self._active = False
        self._stop_time = time.monotonic()
        duration = self._stop_time - self._start_time
        fps = self._frame_count / duration if duration > 0 else 0.0

        meta = {
            "gesture_label":  self.gesture_label,
            "frame_count":    self._frame_count,
            "duration_s":     round(duration, 3),
            "avg_fps":        round(fps, 2),
            "recorded_at":    datetime.now().isoformat(),
            "session_dir":    str(self.session_dir),
        }

        meta_path = self.session_dir / "session.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __repr__(self) -> str:
        state = "recording" if self._active else "stopped"
        return (
            f"Recorder(gesture='{self.gesture_label}', "
            f"frames={self._frame_count}, state={state})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Playback — offline, no hardware needed
# ──────────────────────────────────────────────────────────────────────────────

def load_session(session_dir: str | Path) -> tuple[list[np.ndarray], dict]:
    """
    Load all point cloud frames from a saved session.

    Parameters
    ----------
    session_dir : str | Path
        Path to a session directory (contains session.json + frames/).

    Returns
    -------
    clouds : list of np.ndarray, each shape (N, 3) float32
        Ordered list of normalised point clouds, one per saved frame.
    meta : dict
        Contents of session.json (gesture_label, frame_count, fps, etc.)

    Raises
    ------
    FileNotFoundError
        If session_dir does not exist or contains no frames.
    """
    session_dir = Path(session_dir)
    frames_dir  = session_dir / "frames"
    meta_path   = session_dir / "session.json"

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    frame_files = sorted(frames_dir.glob("*.npy"))
    if not frame_files:
        raise FileNotFoundError(f"No frame files found in: {frames_dir}")

    clouds = [np.load(str(f)).astype(np.float32) for f in frame_files]
    return clouds, meta


def list_sessions(base_dir: str | Path) -> list[dict]:
    """
    List all saved sessions under base_dir, sorted by recording time.

    Returns
    -------
    list of dicts, each containing session metadata plus 'path' key.
    Returns an empty list if base_dir does not exist.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    sessions = []
    for session_dir in sorted(base_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        meta_path = session_dir / "session.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["path"] = str(session_dir)
            sessions.append(meta)

    return sessions
