"""
test_classifier.py
------------------
Unit tests for the MLP gesture classifier.

Tests cover: forward pass correctness, training convergence on a small
synthetic dataset, inference API, persistence (save/load), and
LOSO-CV evaluation using the dataset splits from test_dataset.

Run with:
    cd backend
    python3 -m pytest tests/ -v
"""

import numpy as np
import pytest
from pathlib import Path

from pipeline.classifier import GestureClassifier, TrainingResult
from pipeline.dataset import (
    GestureDataset,
    FEATURE_DIM,
    NUM_CLASSES,
    GESTURE_LABELS,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    cloud_to_feature,
    load_dataset,
    tsts_split,
    loso_splits,
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


def _synthetic_dataset(n_per_class: int = 30, n_classes: int = 3,
                        seed: int = 0) -> GestureDataset:
    """
    Build a linearly-separable synthetic dataset for fast training tests.
    Each class is a cluster of points centred at a different location.
    """
    rng = np.random.default_rng(seed)
    features = []
    labels   = []
    names    = []
    sessions = []

    for c in range(n_classes):
        centre = rng.uniform(-5, 5, FEATURE_DIM).astype(np.float32)
        noise  = rng.normal(0, 0.05, (n_per_class, FEATURE_DIM)).astype(np.float32)
        features.append(centre + noise)
        labels.extend([c] * n_per_class)
        names.extend([GESTURE_LABELS[c]] * n_per_class)
        sessions.extend([f"session_{c}"] * n_per_class)

    return GestureDataset(
        features=np.vstack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.int32),
        label_names=names,
        session_ids=sessions,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Constructor and repr
# ──────────────────────────────────────────────────────────────────────────────

class TestConstructor:
    def test_default_construction(self):
        clf = GestureClassifier()
        assert clf.input_dim   == FEATURE_DIM
        assert clf.num_classes == NUM_CLASSES

    def test_custom_dims(self):
        clf = GestureClassifier(input_dim=10, hidden_dim=16, num_classes=3)
        assert clf.input_dim   == 10
        assert clf.hidden_dim  == 16
        assert clf.num_classes == 3

    def test_untrained_repr(self):
        clf = GestureClassifier()
        assert "untrained" in repr(clf)

    def test_weight_shapes(self):
        clf = GestureClassifier(input_dim=75, hidden_dim=128, num_classes=10)
        assert clf.W1.shape == (75, 128)
        assert clf.W2.shape == (128, 64)
        assert clf.W3.shape == (64, 10)
        assert clf.b1.shape == (128,)
        assert clf.b2.shape == (64,)
        assert clf.b3.shape == (10,)

    def test_weights_dtype_float32(self):
        clf = GestureClassifier()
        for w in [clf.W1, clf.b1, clf.W2, clf.b2, clf.W3, clf.b3]:
            assert w.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass / predict_proba
# ──────────────────────────────────────────────────────────────────────────────

class TestForwardPass:
    def test_proba_shape_batch(self):
        clf = GestureClassifier()
        X = np.random.default_rng(0).uniform(-1, 1, (8, FEATURE_DIM)).astype(np.float32)
        probs = clf.predict_proba(X)
        assert probs.shape == (8, NUM_CLASSES)

    def test_proba_sums_to_one(self):
        clf = GestureClassifier()
        X = np.random.default_rng(1).uniform(-1, 1, (5, FEATURE_DIM)).astype(np.float32)
        probs = clf.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    def test_proba_non_negative(self):
        clf = GestureClassifier()
        X = np.random.default_rng(2).uniform(-1, 1, (5, FEATURE_DIM)).astype(np.float32)
        assert (clf.predict_proba(X) >= 0).all()

    def test_proba_1d_input(self):
        clf = GestureClassifier()
        x = np.random.default_rng(3).uniform(-1, 1, FEATURE_DIM).astype(np.float32)
        probs = clf.predict_proba(x)
        assert probs.shape == (1, NUM_CLASSES)

    def test_proba_dtype(self):
        clf = GestureClassifier()
        X = np.zeros((3, FEATURE_DIM), dtype=np.float32)
        assert clf.predict_proba(X).dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# predict / predict_label
# ──────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_shape(self):
        clf = GestureClassifier()
        X = np.random.default_rng(4).uniform(-1, 1, (10, FEATURE_DIM)).astype(np.float32)
        preds = clf.predict(X)
        assert preds.shape == (10,)

    def test_predict_dtype_int32(self):
        clf = GestureClassifier()
        X = np.zeros((3, FEATURE_DIM), dtype=np.float32)
        assert clf.predict(X).dtype == np.int32

    def test_predict_valid_class_indices(self):
        clf = GestureClassifier()
        X = np.random.default_rng(5).uniform(-1, 1, (20, FEATURE_DIM)).astype(np.float32)
        preds = clf.predict(X)
        assert ((preds >= 0) & (preds < NUM_CLASSES)).all()

    def test_predict_1d_input(self):
        clf = GestureClassifier()
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        preds = clf.predict(x)
        assert preds.shape == (1,)

    def test_predict_label_returns_string(self):
        clf = GestureClassifier()
        cloud = np.zeros((25, 3), dtype=np.float32)
        label = clf.predict_label(cloud)
        assert isinstance(label, str)

    def test_predict_label_in_vocabulary(self):
        clf = GestureClassifier()
        cloud = np.random.default_rng(6).uniform(-1, 1, (25, 3)).astype(np.float32)
        label = clf.predict_label(cloud)
        assert label in GESTURE_LABELS


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

class TestTraining:
    def test_returns_training_result(self):
        clf = GestureClassifier(num_classes=3)
        ds  = _synthetic_dataset(n_per_class=20, n_classes=3)
        result = clf.fit(ds, epochs=5, verbose=False)
        assert isinstance(result, TrainingResult)

    def test_loss_history_length(self):
        clf = GestureClassifier(num_classes=3)
        ds  = _synthetic_dataset(n_per_class=20, n_classes=3)
        result = clf.fit(ds, epochs=10, verbose=False)
        assert len(result.loss_history) == 10

    def test_loss_decreases_on_separable_data(self):
        clf = GestureClassifier(num_classes=3, seed=0)
        ds  = _synthetic_dataset(n_per_class=50, n_classes=3, seed=0)
        result = clf.fit(ds, epochs=50, verbose=False, lr=1e-2)
        # Loss should be lower at epoch 50 than epoch 1
        assert result.loss_history[-1] < result.loss_history[0]

    def test_accuracy_improves_on_separable_data(self):
        clf = GestureClassifier(num_classes=3, seed=1)
        ds  = _synthetic_dataset(n_per_class=50, n_classes=3, seed=1)
        acc_before = clf.score(ds)
        clf.fit(ds, epochs=100, verbose=False, lr=1e-2)
        acc_after  = clf.score(ds)
        assert acc_after > acc_before

    def test_trained_flag_set(self):
        clf = GestureClassifier(num_classes=3)
        ds  = _synthetic_dataset(n_per_class=10, n_classes=3)
        assert clf._trained is False
        clf.fit(ds, epochs=2, verbose=False)
        assert clf._trained is True

    def test_repr_after_training(self):
        clf = GestureClassifier(num_classes=3)
        ds  = _synthetic_dataset(n_per_class=10, n_classes=3)
        clf.fit(ds, epochs=2, verbose=False)
        assert "trained" in repr(clf)
        assert "untrained" not in repr(clf)

    def test_score_between_zero_and_one(self):
        clf = GestureClassifier(num_classes=3)
        ds  = _synthetic_dataset(n_per_class=20, n_classes=3)
        clf.fit(ds, epochs=10, verbose=False)
        acc = clf.score(ds)
        assert 0.0 <= acc <= 1.0

    def test_high_accuracy_on_easy_data(self):
        """Well-separated clusters should converge to near 100% accuracy."""
        clf = GestureClassifier(num_classes=3, seed=7)
        ds  = _synthetic_dataset(n_per_class=60, n_classes=3, seed=7)
        clf.fit(ds, epochs=200, lr=5e-3, verbose=False)
        assert clf.score(ds) > 0.90


# ──────────────────────────────────────────────────────────────────────────────
# Persistence (save / load)
# ──────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        clf = GestureClassifier(num_classes=3)
        clf.save(tmp_path / "model")
        assert (tmp_path / "model.npz").exists()

    def test_load_restores_weights(self, tmp_path):
        clf = GestureClassifier(num_classes=3, seed=0)
        clf.save(tmp_path / "model")
        clf2 = GestureClassifier.load(tmp_path / "model")
        np.testing.assert_array_equal(clf.W1, clf2.W1)
        np.testing.assert_array_equal(clf.W3, clf2.W3)

    def test_load_preserves_predictions(self, tmp_path):
        clf = GestureClassifier(num_classes=3, seed=2)
        ds  = _synthetic_dataset(n_per_class=20, n_classes=3)
        clf.fit(ds, epochs=10, verbose=False)
        clf.save(tmp_path / "model")

        clf2 = GestureClassifier.load(tmp_path / "model")
        preds1 = clf.predict(ds.features)
        preds2 = clf2.predict(ds.features)
        np.testing.assert_array_equal(preds1, preds2)

    def test_load_sets_trained_flag(self, tmp_path):
        clf = GestureClassifier(num_classes=3)
        clf.save(tmp_path / "m")
        clf2 = GestureClassifier.load(tmp_path / "m")
        assert clf2._trained is True

    def test_npz_extension_added_automatically(self, tmp_path):
        clf = GestureClassifier(num_classes=3)
        clf.save(tmp_path / "no_ext")
        assert (tmp_path / "no_ext.npz").exists()

    def test_load_without_npz_extension(self, tmp_path):
        clf = GestureClassifier(num_classes=3)
        clf.save(tmp_path / "model")
        # Load without explicit .npz
        clf2 = GestureClassifier.load(tmp_path / "model")
        assert isinstance(clf2, GestureClassifier)

    def test_roundtrip_hyperparameters(self, tmp_path):
        clf = GestureClassifier(input_dim=75, hidden_dim=64, num_classes=5)
        clf.save(tmp_path / "m2")
        clf2 = GestureClassifier.load(tmp_path / "m2")
        assert clf2.input_dim   == 75
        assert clf2.hidden_dim  == 64
        assert clf2.num_classes == 5


# ──────────────────────────────────────────────────────────────────────────────
# LOSO-CV integration test
# ──────────────────────────────────────────────────────────────────────────────

class TestLOSOCV:
    def test_loso_each_split_trains_and_scores(self):
        """
        LOSO-CV integration: for each held-out session, train on the rest
        and evaluate on the held-out session.  Uses synthetic data so the
        test is fast and hardware-free.

        Uses 3-class synthetic dataset with 3 sessions (one per class)
        and NUM_CLASSES=3 classifier so label indices are valid.
        """
        ds = _synthetic_dataset(n_per_class=30, n_classes=3, seed=42)
        splits = loso_splits(ds)
        assert len(splits) == 3   # one held-out session per unique session

        for train, test, held_out in splits:
            clf = GestureClassifier(
                input_dim=FEATURE_DIM,
                hidden_dim=64,
                num_classes=3,
                seed=0,
            )
            result = clf.fit(train, epochs=30, verbose=False, lr=5e-3)
            acc = clf.score(test)
            assert 0.0 <= acc <= 1.0
            assert isinstance(result, TrainingResult)
