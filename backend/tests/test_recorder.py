"""
test_recorder.py
----------------
Unit tests for the Recorder (save) and load_session / list_sessions (load).
Uses a temporary directory so no real filesystem state is left behind.

Run with:
    cd backend
    python3 -m pytest tests/ -v
"""

import json
import numpy as np
import pytest

from pipeline.recorder import Recorder, load_session, list_sessions
from pipeline.point_cloud import PointCloud


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_pc(n: int = 25, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    return PointCloud(
        points=pts,
        centroid=np.zeros(3, dtype=np.float32),
        scale=1.0,
        valid_count=n,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Recorder
# ──────────────────────────────────────────────────────────────────────────────

class TestRecorder:
    def test_creates_directories(self, tmp_path):
        rec = Recorder(tmp_path, "ThumbsUp", session_name="test_session")
        rec.start()
        rec.stop()
        assert (tmp_path / "test_session").is_dir()
        assert (tmp_path / "test_session" / "frames").is_dir()

    def test_frame_files_created(self, tmp_path):
        rec = Recorder(tmp_path, "ThumbsUp", session_name="s1")
        rec.start()
        for i in range(5):
            rec.add(_make_pc(seed=i))
        rec.stop()
        frame_files = sorted((tmp_path / "s1" / "frames").glob("*.npy"))
        assert len(frame_files) == 5

    def test_frame_filenames_zero_padded(self, tmp_path):
        rec = Recorder(tmp_path, "ASL_L", session_name="s2")
        rec.start()
        rec.add(_make_pc())
        rec.add(_make_pc(seed=1))
        rec.stop()
        names = [f.name for f in sorted((tmp_path / "s2" / "frames").glob("*.npy"))]
        assert names == ["000000.npy", "000001.npy"]

    def test_frame_count_property(self, tmp_path):
        rec = Recorder(tmp_path, "Spock", session_name="s3")
        rec.start()
        for i in range(7):
            rec.add(_make_pc(seed=i))
        assert rec.frame_count == 7
        rec.stop()
        assert rec.frame_count == 7

    def test_session_json_written(self, tmp_path):
        rec = Recorder(tmp_path, "Victory", session_name="s4")
        rec.start()
        rec.add(_make_pc())
        meta = rec.stop()
        assert (tmp_path / "s4" / "session.json").exists()

    def test_session_json_contents(self, tmp_path):
        rec = Recorder(tmp_path, "Five", session_name="s5")
        rec.start()
        rec.add(_make_pc())
        rec.add(_make_pc(seed=1))
        meta = rec.stop()
        assert meta["gesture_label"] == "Five"
        assert meta["frame_count"] == 2
        assert meta["duration_s"] >= 0.0
        assert "recorded_at" in meta

    def test_stop_returns_meta_dict(self, tmp_path):
        rec = Recorder(tmp_path, "One", session_name="s6")
        rec.start()
        result = rec.stop()
        assert isinstance(result, dict)

    def test_stop_is_idempotent(self, tmp_path):
        rec = Recorder(tmp_path, "Two", session_name="s7")
        rec.start()
        rec.add(_make_pc())
        rec.stop()
        # Second stop should not raise or create extra files
        rec.stop()
        frame_files = list((tmp_path / "s7" / "frames").glob("*.npy"))
        assert len(frame_files) == 1

    def test_add_before_start_raises(self, tmp_path):
        rec = Recorder(tmp_path, "Three", session_name="s8")
        with pytest.raises(RuntimeError):
            rec.add(_make_pc())

    def test_add_after_stop_raises(self, tmp_path):
        rec = Recorder(tmp_path, "Four", session_name="s9")
        rec.start()
        rec.stop()
        with pytest.raises(RuntimeError):
            rec.add(_make_pc())

    def test_saved_array_values_match_input(self, tmp_path):
        pc = _make_pc(seed=99)
        rec = Recorder(tmp_path, "Test", session_name="s10")
        rec.start()
        rec.add(pc)
        rec.stop()
        loaded = np.load(str(tmp_path / "s10" / "frames" / "000000.npy"))
        np.testing.assert_array_equal(loaded, pc.points)

    def test_repr_contains_gesture_and_state(self, tmp_path):
        rec = Recorder(tmp_path, "Spiderman", session_name="s11")
        rec.start()
        r = repr(rec)
        assert "Spiderman" in r
        assert "recording" in r
        rec.stop()
        r2 = repr(rec)
        assert "stopped" in r2

    def test_auto_session_name_contains_gesture(self, tmp_path):
        rec = Recorder(tmp_path, "Spock")
        rec.start()
        rec.stop()
        sessions = [d.name for d in tmp_path.iterdir() if d.is_dir()]
        assert any("Spock" in s for s in sessions)


# ──────────────────────────────────────────────────────────────────────────────
# load_session
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadSession:
    def _record(self, tmp_path, label: str, n_frames: int, session_name: str):
        rec = Recorder(tmp_path, label, session_name=session_name)
        rec.start()
        for i in range(n_frames):
            rec.add(_make_pc(seed=i))
        rec.stop()
        return tmp_path / session_name

    def test_returns_correct_number_of_clouds(self, tmp_path):
        path = self._record(tmp_path, "ThumbsUp", 10, "sess")
        clouds, _ = load_session(path)
        assert len(clouds) == 10

    def test_clouds_are_numpy_arrays(self, tmp_path):
        path = self._record(tmp_path, "ASL_Y", 3, "sess2")
        clouds, _ = load_session(path)
        for c in clouds:
            assert isinstance(c, np.ndarray)

    def test_clouds_dtype_float32(self, tmp_path):
        path = self._record(tmp_path, "Spock", 3, "sess3")
        clouds, _ = load_session(path)
        for c in clouds:
            assert c.dtype == np.float32

    def test_clouds_shape(self, tmp_path):
        path = self._record(tmp_path, "Five", 4, "sess4")
        clouds, _ = load_session(path)
        for c in clouds:
            assert c.ndim == 2
            assert c.shape[1] == 3

    def test_metadata_returned(self, tmp_path):
        path = self._record(tmp_path, "Victory", 5, "sess5")
        _, meta = load_session(path)
        assert meta["gesture_label"] == "Victory"
        assert meta["frame_count"] == 5

    def test_cloud_values_match_recorded(self, tmp_path):
        pc = _make_pc(seed=77)
        rec = Recorder(tmp_path, "Test", session_name="sess6")
        rec.start()
        rec.add(pc)
        rec.stop()
        clouds, _ = load_session(tmp_path / "sess6")
        np.testing.assert_array_equal(clouds[0], pc.points)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nonexistent")

    def test_empty_frames_directory_raises(self, tmp_path):
        session_dir = tmp_path / "empty_sess"
        (session_dir / "frames").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_session(session_dir)

    def test_clouds_in_correct_order(self, tmp_path):
        """Frames must be loaded in filename order (000000, 000001, …)."""
        rec = Recorder(tmp_path, "Ordered", session_name="sess7")
        rec.start()
        original_clouds = [_make_pc(seed=i) for i in range(5)]
        for pc in original_clouds:
            rec.add(pc)
        rec.stop()
        loaded, _ = load_session(tmp_path / "sess7")
        for original, loaded_cloud in zip(original_clouds, loaded):
            np.testing.assert_array_equal(original.points, loaded_cloud)


# ──────────────────────────────────────────────────────────────────────────────
# list_sessions
# ──────────────────────────────────────────────────────────────────────────────

class TestListSessions:
    def test_empty_base_dir_returns_empty_list(self, tmp_path):
        result = list_sessions(tmp_path)
        assert result == []

    def test_nonexistent_base_dir_returns_empty_list(self, tmp_path):
        result = list_sessions(tmp_path / "does_not_exist")
        assert result == []

    def test_lists_all_sessions(self, tmp_path):
        for name, label in [("s1", "ThumbsUp"), ("s2", "Spock"), ("s3", "Five")]:
            rec = Recorder(tmp_path, label, session_name=name)
            rec.start()
            rec.add(_make_pc())
            rec.stop()
        sessions = list_sessions(tmp_path)
        assert len(sessions) == 3

    def test_each_entry_has_path(self, tmp_path):
        rec = Recorder(tmp_path, "ASL_L", session_name="sl")
        rec.start()
        rec.add(_make_pc())
        rec.stop()
        sessions = list_sessions(tmp_path)
        assert "path" in sessions[0]

    def test_each_entry_has_gesture_label(self, tmp_path):
        rec = Recorder(tmp_path, "Victory", session_name="sv")
        rec.start()
        rec.add(_make_pc())
        rec.stop()
        sessions = list_sessions(tmp_path)
        assert sessions[0]["gesture_label"] == "Victory"

    def test_ignores_non_session_directories(self, tmp_path):
        """Directories without session.json should be silently skipped."""
        (tmp_path / "not_a_session").mkdir()
        rec = Recorder(tmp_path, "Real", session_name="real_sess")
        rec.start()
        rec.add(_make_pc())
        rec.stop()
        sessions = list_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0]["gesture_label"] == "Real"
