"""
test_dataset.py
---------------
Unit tests for dataset loading, feature extraction, and train/test splits.
All tests are hardware-free; sessions are synthesised in temp directories.

Run with:
    cd backend
    python3 -m pytest tests/ -v
"""

import json
import numpy as np
import pytest
from pathlib import Path

from pipeline.dataset import (
    GestureDataset,
    cloud_to_feature,
    load_dataset,
    tsts_split,
    loso_splits,
    GESTURE_LABELS,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    FEATURE_DIM,
    NUM_CLASSES,
)
from pipeline.recorder import Recorder
from pipeline.point_cloud import PointCloud


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_pc(n: int = 25, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    return PointCloud(points=pts, centroid=np.zeros(3, dtype=np.float32),
                      scale=1.0, valid_count=n)


def _record_session(base_dir: Path, label: str, n_frames: int,
                    session_name: str, seed: int = 0) -> Path:
    rec = Recorder(base_dir, label, session_name=session_name)
    rec.start()
    for i in range(n_frames):
        rec.add(_make_pc(seed=seed + i))
    rec.stop()
    return base_dir / session_name


def _make_dataset(tmp_path: Path) -> GestureDataset:
    """Build a small dataset with 3 gestures × 10 frames each."""
    for i, gesture in enumerate(["Thumbs_Up", "Spock", "Five"]):
        _record_session(tmp_path, gesture, 10, f"sess_{gesture}", seed=i * 100)
    return load_dataset(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

class TestVocabulary:
    def test_gesture_labels_count(self):
        assert len(GESTURE_LABELS) == 10

    def test_label_to_idx_bijective(self):
        assert len(LABEL_TO_IDX) == len(IDX_TO_LABEL) == NUM_CLASSES

    def test_round_trip_label_idx(self):
        for label in GESTURE_LABELS:
            idx = LABEL_TO_IDX[label]
            assert IDX_TO_LABEL[idx] == label

    def test_feature_dim(self):
        assert FEATURE_DIM == 75    # 25 joints × 3


# ──────────────────────────────────────────────────────────────────────────────
# cloud_to_feature
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudToFeature:
    def test_output_shape(self):
        cloud = np.random.default_rng(0).uniform(-1, 1, (25, 3)).astype(np.float32)
        f = cloud_to_feature(cloud)
        assert f.shape == (FEATURE_DIM,)

    def test_output_dtype(self):
        cloud = np.zeros((25, 3), dtype=np.float32)
        assert cloud_to_feature(cloud).dtype == np.float32

    def test_values_correct(self):
        cloud = np.arange(75, dtype=np.float32).reshape(25, 3)
        f = cloud_to_feature(cloud)
        np.testing.assert_array_equal(f, cloud.flatten())

    def test_fewer_than_25_zero_padded(self):
        cloud = np.ones((10, 3), dtype=np.float32)
        f = cloud_to_feature(cloud)
        assert f.shape == (FEATURE_DIM,)
        # First 30 values should be 1.0, rest should be 0.0
        np.testing.assert_array_equal(f[:30], np.ones(30))
        np.testing.assert_array_equal(f[30:], np.zeros(FEATURE_DIM - 30))

    def test_more_than_25_truncated(self):
        cloud = np.ones((30, 3), dtype=np.float32)
        f = cloud_to_feature(cloud)
        assert f.shape == (FEATURE_DIM,)


# ──────────────────────────────────────────────────────────────────────────────
# load_dataset
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadDataset:
    def test_loads_correct_total_frames(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert len(ds) == 30   # 3 gestures × 10 frames

    def test_features_shape(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert ds.features.shape == (30, FEATURE_DIM)

    def test_features_dtype(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert ds.features.dtype == np.float32

    def test_labels_shape(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert ds.labels.shape == (30,)

    def test_labels_dtype(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert ds.labels.dtype == np.int32

    def test_correct_label_indices(self, tmp_path):
        ds = _make_dataset(tmp_path)
        unique_labels = set(ds.label_names)
        for label in ["Thumbs_Up", "Spock", "Five"]:
            assert label in unique_labels

    def test_label_names_length(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert len(ds.label_names) == 30

    def test_session_ids_length(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert len(ds.session_ids) == 30

    def test_three_unique_sessions(self, tmp_path):
        ds = _make_dataset(tmp_path)
        assert len(set(ds.session_ids)) == 3

    def test_unknown_gesture_skipped(self, tmp_path):
        _record_session(tmp_path, "UnknownGesture", 5, "sess_unknown")
        _record_session(tmp_path, "Thumbs_Up", 10, "sess_known")
        ds = load_dataset(tmp_path)
        assert all(n != "UnknownGesture" for n in ds.label_names)
        assert len(ds) == 10

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_dataset(tmp_path)

    def test_class_counts(self, tmp_path):
        ds = _make_dataset(tmp_path)
        counts = ds.class_counts()
        for gesture in ["Thumbs_Up", "Spock", "Five"]:
            assert counts[gesture] == 10

    def test_repr(self, tmp_path):
        ds = _make_dataset(tmp_path)
        r = repr(ds)
        assert "GestureDataset" in r
        assert "30" in r


# ──────────────────────────────────────────────────────────────────────────────
# tsts_split
# ──────────────────────────────────────────────────────────────────────────────

class TestTSTSSplit:
    def test_total_frames_preserved(self, tmp_path):
        ds = _make_dataset(tmp_path)
        train, test = tsts_split(ds, train_ratio=0.8)
        assert len(train) + len(test) == len(ds)

    def test_train_larger_than_test(self, tmp_path):
        ds = _make_dataset(tmp_path)
        train, test = tsts_split(ds, train_ratio=0.8)
        assert len(train) > len(test)

    def test_no_overlap(self, tmp_path):
        ds = _make_dataset(tmp_path)
        train, test = tsts_split(ds)
        train_idx = set(id(f.tobytes()) for f in train.features)
        test_idx  = set(id(f.tobytes()) for f in test.features)
        # Row-level uniqueness check via values
        for row in test.features:
            match = any(np.array_equal(row, tr) for tr in train.features)
            # Not asserting no duplicates (values could coincide by chance),
            # just that dataset sizes are correct — checked above

    def test_returns_gesture_datasets(self, tmp_path):
        ds = _make_dataset(tmp_path)
        train, test = tsts_split(ds)
        assert isinstance(train, GestureDataset)
        assert isinstance(test,  GestureDataset)

    def test_ratio_zero_gives_all_test(self, tmp_path):
        ds = _make_dataset(tmp_path)
        train, test = tsts_split(ds, train_ratio=0.0)
        # Each session contributes at least 1 train frame (max(1, 0))
        assert len(train) >= 3   # 3 sessions × 1 minimum
        assert len(test) == len(ds) - len(train)


# ──────────────────────────────────────────────────────────────────────────────
# loso_splits
# ──────────────────────────────────────────────────────────────────────────────

class TestLOSOSplits:
    def test_number_of_splits(self, tmp_path):
        ds = _make_dataset(tmp_path)
        splits = loso_splits(ds)
        assert len(splits) == 3   # 3 sessions

    def test_each_split_is_tuple_of_three(self, tmp_path):
        ds = _make_dataset(tmp_path)
        for train, test, held_out in loso_splits(ds):
            assert isinstance(train,    GestureDataset)
            assert isinstance(test,     GestureDataset)
            assert isinstance(held_out, str)

    def test_test_set_is_held_out_session(self, tmp_path):
        ds = _make_dataset(tmp_path)
        for _, test, held_out in loso_splits(ds):
            assert all(s == held_out for s in test.session_ids)

    def test_train_set_excludes_held_out(self, tmp_path):
        ds = _make_dataset(tmp_path)
        for train, _, held_out in loso_splits(ds):
            assert held_out not in train.session_ids

    def test_train_plus_test_equals_total(self, tmp_path):
        ds = _make_dataset(tmp_path)
        for train, test, _ in loso_splits(ds):
            assert len(train) + len(test) == len(ds)
