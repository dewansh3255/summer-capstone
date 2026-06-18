"""
test_chamfer.py
---------------
Unit tests for Step 3: Chamfer Distance computation and statistical
thresholding.  All tests are hardware-free.

Run with:
    cd backend
    python3 -m pytest tests/ -v
"""

import numpy as np
import pytest

from pipeline.chamfer import (
    chamfer_distance,
    chamfer_distance_pc,
    pairwise_chamfer_matrix,
    find_optimal_threshold,
    extract_pairs,
    ThresholdResult,
    ThresholdMetrics,
)
from pipeline.point_cloud import PointCloud


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cloud(pts: list[tuple]) -> np.ndarray:
    return np.array(pts, dtype=np.float32)


def _pc(pts: list[tuple]) -> PointCloud:
    arr = _cloud(pts)
    return PointCloud(
        points=arr,
        centroid=np.zeros(3, dtype=np.float32),
        scale=1.0,
        valid_count=len(arr),
    )


def _random_cloud(n: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, 3)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# chamfer_distance — core function
# ──────────────────────────────────────────────────────────────────────────────

class TestChamferDistance:
    def test_identical_clouds_is_zero(self):
        a = _random_cloud(25, seed=0)
        assert chamfer_distance(a, a) == pytest.approx(0.0, abs=1e-7)

    def test_symmetric(self):
        a = _random_cloud(25, seed=1)
        b = _random_cloud(25, seed=2)
        assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a), abs=1e-7)

    def test_non_negative(self):
        a = _random_cloud(25, seed=3)
        b = _random_cloud(25, seed=4)
        assert chamfer_distance(a, b) >= 0.0

    def test_empty_cloud_returns_zero(self):
        a = _random_cloud(25)
        empty = np.zeros((0, 3), dtype=np.float32)
        assert chamfer_distance(a, empty) == 0.0
        assert chamfer_distance(empty, a) == 0.0

    def test_single_point_clouds(self):
        a = _cloud([(0.0, 0.0, 0.0)])
        b = _cloud([(1.0, 0.0, 0.0)])
        # CD = mean(1^2) + mean(1^2) = 2.0
        assert chamfer_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_known_value_two_points(self):
        # a = {(0,0,0)}, b = {(3,4,0)}  → distance² = 25
        # CD = 25/1 + 25/1 = 50
        a = _cloud([(0.0, 0.0, 0.0)])
        b = _cloud([(3.0, 4.0, 0.0)])
        assert chamfer_distance(a, b) == pytest.approx(50.0, abs=1e-5)

    def test_larger_separation_gives_larger_distance(self):
        base  = _random_cloud(25, seed=5)
        close = base + 0.01
        far   = base + 1.0
        assert chamfer_distance(base, far) > chamfer_distance(base, close)

    def test_different_sized_clouds(self):
        a = _random_cloud(10, seed=6)
        b = _random_cloud(25, seed=7)
        d = chamfer_distance(a, b)
        assert d >= 0.0
        # Symmetry still holds for different-size clouds
        assert chamfer_distance(b, a) == pytest.approx(d, abs=1e-6)

    def test_translated_cloud_higher_distance(self):
        a = _random_cloud(25, seed=8)
        b = a.copy()
        b_shifted = b + 5.0    # large shift
        assert chamfer_distance(a, b_shifted) > chamfer_distance(a, b)

    def test_result_is_python_float(self):
        a = _random_cloud(5)
        b = _random_cloud(5, seed=1)
        assert isinstance(chamfer_distance(a, b), float)


# ──────────────────────────────────────────────────────────────────────────────
# chamfer_distance_pc — PointCloud wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestChamferDistancePC:
    def test_same_cloud_is_zero(self):
        pts = [(0.1 * i, 0.0, 0.5) for i in range(10)]
        pc = _pc(pts)
        assert chamfer_distance_pc(pc, pc) == pytest.approx(0.0, abs=1e-7)

    def test_matches_raw_function(self):
        a = _pc([(float(i), 0.0, 0.0) for i in range(10)])
        b = _pc([(float(i), 0.1, 0.0) for i in range(10)])
        raw  = chamfer_distance(a.points, b.points)
        wrap = chamfer_distance_pc(a, b)
        assert raw == pytest.approx(wrap, abs=1e-7)


# ──────────────────────────────────────────────────────────────────────────────
# pairwise_chamfer_matrix
# ──────────────────────────────────────────────────────────────────────────────

class TestPairwiseMatrix:
    def test_shape(self):
        clouds = [_random_cloud(25, seed=i) for i in range(5)]
        D = pairwise_chamfer_matrix(clouds)
        assert D.shape == (5, 5)

    def test_diagonal_is_zero(self):
        clouds = [_random_cloud(25, seed=i) for i in range(4)]
        D = pairwise_chamfer_matrix(clouds)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-8)

    def test_symmetric(self):
        clouds = [_random_cloud(25, seed=i) for i in range(4)]
        D = pairwise_chamfer_matrix(clouds)
        np.testing.assert_allclose(D, D.T, atol=1e-7)

    def test_dtype_float64(self):
        clouds = [_random_cloud(5, seed=i) for i in range(3)]
        D = pairwise_chamfer_matrix(clouds)
        assert D.dtype == np.float64

    def test_single_cloud(self):
        D = pairwise_chamfer_matrix([_random_cloud(5)])
        assert D.shape == (1, 1)
        assert D[0, 0] == pytest.approx(0.0)

    def test_values_match_direct_calls(self):
        clouds = [_random_cloud(25, seed=i) for i in range(3)]
        D = pairwise_chamfer_matrix(clouds)
        for i in range(3):
            for j in range(3):
                expected = chamfer_distance(clouds[i], clouds[j])
                assert D[i, j] == pytest.approx(expected, abs=1e-7)

    def test_non_negative(self):
        clouds = [_random_cloud(10, seed=i) for i in range(5)]
        D = pairwise_chamfer_matrix(clouds)
        assert (D >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# find_optimal_threshold
# ──────────────────────────────────────────────────────────────────────────────

class TestFindOptimalThreshold:
    def _make_separable_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct a perfectly separable dataset:
          - Same-gesture pairs have CD drawn from U(0.0, 0.1)
          - Different-gesture pairs have CD drawn from U(0.5, 1.0)
        Optimal threshold should land in (0.1, 0.5) and achieve F1=1.0.
        """
        rng = np.random.default_rng(42)
        same_dists = rng.uniform(0.0,  0.1, 200)
        diff_dists = rng.uniform(0.5,  1.0, 200)
        distances  = np.concatenate([same_dists, diff_dists])
        labels     = np.concatenate([np.ones(200), np.zeros(200)]).astype(np.int32)
        return distances, labels

    def test_returns_threshold_result(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l)
        assert isinstance(result, ThresholdResult)

    def test_perfect_separation_gives_f1_one(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l, n_thresholds=500)
        assert result.optimal_f1 == pytest.approx(1.0, abs=1e-6)

    def test_optimal_threshold_in_gap(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l, n_thresholds=500)
        # Should land in the gap between 0.1 and 0.5
        assert 0.09 < result.optimal_threshold < 0.51

    def test_metrics_curve_length(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l, n_thresholds=100)
        assert len(result.metrics_curve) == 100

    def test_metrics_curve_entries_are_threshold_metrics(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l, n_thresholds=10)
        for m in result.metrics_curve:
            assert isinstance(m, ThresholdMetrics)

    def test_all_same_label_edge_case(self):
        """All pairs are same-gesture — threshold should be high, recall=1."""
        d = np.array([0.01, 0.02, 0.03, 0.04])
        l = np.ones(4, dtype=np.int32)
        result = find_optimal_threshold(d, l, n_thresholds=50)
        # At threshold >= max(d), all predicted same, recall = 1
        assert result.metrics_curve[-1].recall == pytest.approx(1.0)

    def test_all_different_label_edge_case(self):
        """All pairs are different-gesture — threshold should be 0, recall=1 for diff."""
        d = np.array([0.5, 0.6, 0.7])
        l = np.zeros(3, dtype=np.int32)
        # This is degenerate for F1 of 'same' class but should not raise
        result = find_optimal_threshold(d, l, n_thresholds=20)
        assert isinstance(result, ThresholdResult)

    def test_f1_in_valid_range(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l)
        for m in result.metrics_curve:
            assert 0.0 <= m.f1      <= 1.0
            assert 0.0 <= m.precision <= 1.0
            assert 0.0 <= m.recall    <= 1.0
            assert 0.0 <= m.accuracy  <= 1.0

    def test_confusion_matrix_counts_sum_to_total(self):
        d, l = self._make_separable_data()
        result = find_optimal_threshold(d, l, n_thresholds=10)
        total = len(l)
        for m in result.metrics_curve:
            assert m.tp + m.tn + m.fp + m.fn == total

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            find_optimal_threshold(np.array([0.1, 0.2]), np.array([1]))


# ──────────────────────────────────────────────────────────────────────────────
# extract_pairs
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractPairs:
    def test_pair_count(self):
        clouds = [_random_cloud(5, seed=i) for i in range(4)]
        labels = ["A", "A", "B", "B"]
        d, l = extract_pairs(clouds, labels)
        # 4 clouds → 4*3/2 = 6 pairs
        assert len(d) == 6
        assert len(l) == 6

    def test_same_label_pairs_marked_one(self):
        # 2 same-class, 2 different-class pairs
        clouds = [_random_cloud(5, seed=i) for i in range(3)]
        labels = ["A", "A", "B"]
        d, l = extract_pairs(clouds, labels)
        # Pairs: (A,A)=1, (A,B)=0, (A,B)=0
        assert list(l) == [1, 0, 0]

    def test_different_label_pairs_marked_zero(self):
        clouds = [_random_cloud(5, seed=i) for i in range(2)]
        labels = ["X", "Y"]
        d, l = extract_pairs(clouds, labels)
        assert l[0] == 0

    def test_distances_non_negative(self):
        clouds = [_random_cloud(10, seed=i) for i in range(4)]
        labels = ["A", "A", "B", "B"]
        d, l = extract_pairs(clouds, labels)
        assert (d >= 0).all()

    def test_same_cloud_pair_distance_zero(self):
        a = _random_cloud(10, seed=0)
        clouds = [a, a, _random_cloud(10, seed=1)]
        labels = ["G", "G", "H"]
        d, l = extract_pairs(clouds, labels)
        # First pair is (a, a) → CD = 0
        assert d[0] == pytest.approx(0.0, abs=1e-7)

    def test_output_dtypes(self):
        clouds = [_random_cloud(5, seed=i) for i in range(3)]
        labels = ["A", "B", "A"]
        d, l = extract_pairs(clouds, labels)
        assert d.dtype == np.float64
        assert l.dtype == np.int32

    def test_single_pair(self):
        clouds = [_random_cloud(5, seed=0), _random_cloud(5, seed=1)]
        labels = ["G1", "G2"]
        d, l = extract_pairs(clouds, labels)
        assert len(d) == 1
        assert l[0] == 0   # different labels
