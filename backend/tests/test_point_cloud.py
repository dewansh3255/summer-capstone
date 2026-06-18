"""
test_point_cloud.py
-------------------
Unit tests for Step 2: HandData → normalised PointCloud.

All tests are hardware-free — HandData objects are constructed directly
from synthetic joint values.

Run with:
    cd backend
    python3 -m pytest tests/ -v
"""

import numpy as np
import pytest

from protocol.frame_deserializer import HandData, JointData, JOINT_NAMES, NUM_JOINTS
from pipeline.point_cloud import build_point_cloud, PointCloud, MIN_VALID_JOINTS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_hand(positions: list[tuple], valid_flags: list[bool] = None) -> HandData:
    """
    Build a HandData from a list of (x, y, z) tuples.
    valid_flags defaults to all-True if not provided.
    positions must have exactly NUM_JOINTS entries.
    """
    assert len(positions) == NUM_JOINTS
    if valid_flags is None:
        valid_flags = [True] * NUM_JOINTS
    assert len(valid_flags) == NUM_JOINTS

    joints = [
        JointData(
            name=JOINT_NAMES[i],
            valid=valid_flags[i],
            x=float(positions[i][0]),
            y=float(positions[i][1]),
            z=float(positions[i][2]),
            depth=0.5,
        )
        for i in range(NUM_JOINTS)
    ]
    return HandData(joints=joints)


def _all_valid_hand(seed: int = 0) -> HandData:
    """25 joints with deterministic positions, all valid."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.1, 0.1, (NUM_JOINTS, 3))
    return _make_hand([tuple(p) for p in pts])


# ──────────────────────────────────────────────────────────────────────────────
# Tests: output type and basic structure
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_returns_point_cloud_object(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand, frame_index=5)
        assert isinstance(pc, PointCloud)

    def test_frame_index_stored(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand, frame_index=42)
        assert pc.frame_index == 42

    def test_points_shape(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.points.ndim == 2
        assert pc.points.shape[1] == 3

    def test_points_dtype_float32(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.points.dtype == np.float32

    def test_centroid_shape(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.centroid.shape == (3,)

    def test_centroid_dtype_float32(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.centroid.dtype == np.float32

    def test_valid_count_equals_25_when_all_valid(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.valid_count == NUM_JOINTS

    def test_len_equals_valid_count(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert len(pc) == pc.valid_count

    def test_repr_contains_useful_info(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        r = repr(pc)
        assert "PointCloud" in r
        assert "25" in r


# ──────────────────────────────────────────────────────────────────────────────
# Tests: normalisation correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalisation:
    def test_centred_at_origin(self):
        """After normalisation, mean of points must be ~0."""
        hand = _all_valid_hand(seed=1)
        pc = build_point_cloud(hand)
        mean = pc.points.mean(axis=0)
        np.testing.assert_allclose(mean, np.zeros(3), atol=1e-5)

    def test_fits_inside_unit_sphere(self):
        """All normalised points must have L2 norm <= 1.0 (with tolerance)."""
        hand = _all_valid_hand(seed=2)
        pc = build_point_cloud(hand)
        norms = np.linalg.norm(pc.points, axis=1)
        assert norms.max() <= 1.0 + 1e-5

    def test_at_least_one_point_on_unit_sphere(self):
        """The scaling ensures the furthest point lands exactly at norm=1."""
        hand = _all_valid_hand(seed=3)
        pc = build_point_cloud(hand)
        norms = np.linalg.norm(pc.points, axis=1)
        assert abs(norms.max() - 1.0) < 1e-5

    def test_scale_stored_correctly(self):
        """pc.scale should equal the max L2 norm of the centred raw points."""
        hand = _all_valid_hand(seed=4)
        raw = hand.to_point_cloud(valid_only=True)
        centred = raw - raw.mean(axis=0)
        expected_scale = float(np.linalg.norm(centred, axis=1).max())
        pc = build_point_cloud(hand)
        assert abs(pc.scale - expected_scale) < 1e-5

    def test_centroid_matches_raw_mean(self):
        """pc.centroid should equal the mean of the raw (pre-normalisation) points."""
        hand = _all_valid_hand(seed=5)
        raw = hand.to_point_cloud(valid_only=True)
        expected_centroid = raw.mean(axis=0)
        pc = build_point_cloud(hand)
        np.testing.assert_allclose(pc.centroid, expected_centroid, atol=1e-5)

    def test_normalisation_is_scale_invariant(self):
        """
        Two hands with identical joint topology but different absolute distances
        from the camera should produce the same normalised point cloud.
        """
        base_pts = [(0.01 * i, 0.0, 0.1) for i in range(NUM_JOINTS)]
        near_hand = _make_hand(base_pts)                      # ~0.1 m from camera
        far_pts   = [(0.01 * i, 0.0, 1.0) for i in range(NUM_JOINTS)]
        far_hand  = _make_hand(far_pts)                       # ~1.0 m from camera

        pc_near = build_point_cloud(near_hand)
        pc_far  = build_point_cloud(far_hand)

        # Normalised points should be identical (both centred + scaled to unit sphere)
        np.testing.assert_allclose(pc_near.points, pc_far.points, atol=1e-5)

    def test_two_different_gestures_produce_different_clouds(self):
        """Sanity: distinct hand shapes must produce distinct normalised clouds."""
        fist_pts  = [(0.0, 0.0, float(i) * 0.005) for i in range(NUM_JOINTS)]
        open_pts  = [(float(i) * 0.01, float(i) * 0.01, float(i) * 0.005)
                     for i in range(NUM_JOINTS)]
        pc_fist = build_point_cloud(_make_hand(fist_pts))
        pc_open = build_point_cloud(_make_hand(open_pts))
        assert not np.allclose(pc_fist.points, pc_open.points, atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: partial occlusion (some joints invalid)
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialOcclusion:
    def test_only_valid_joints_in_output(self):
        """Invalid joints must be excluded from the point cloud."""
        positions = [(float(i), 0.0, 0.5) for i in range(NUM_JOINTS)]
        valid_flags = [True] * 15 + [False] * 10
        hand = _make_hand(positions, valid_flags)
        pc = build_point_cloud(hand)
        assert pc.valid_count == 15
        assert len(pc.points) == 15

    def test_exactly_min_valid_joints_returns_cloud(self):
        """At the threshold boundary, a cloud should still be returned."""
        positions = [(float(i), 0.0, 0.5) for i in range(NUM_JOINTS)]
        valid_flags = [True] * MIN_VALID_JOINTS + [False] * (NUM_JOINTS - MIN_VALID_JOINTS)
        hand = _make_hand(positions, valid_flags)
        pc = build_point_cloud(hand)
        assert pc is not None
        assert pc.valid_count == MIN_VALID_JOINTS

    def test_below_min_valid_joints_returns_none(self):
        """Below the threshold, None should be returned."""
        positions = [(float(i), 0.0, 0.5) for i in range(NUM_JOINTS)]
        valid_flags = [True] * (MIN_VALID_JOINTS - 1) + [False] * (NUM_JOINTS - MIN_VALID_JOINTS + 1)
        hand = _make_hand(positions, valid_flags)
        pc = build_point_cloud(hand)
        assert pc is None

    def test_all_invalid_returns_none(self):
        positions = [(0.0, 0.0, 0.5)] * NUM_JOINTS
        valid_flags = [False] * NUM_JOINTS
        hand = _make_hand(positions, valid_flags)
        assert build_point_cloud(hand) is None

    def test_is_valid_method(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.is_valid() is True

    def test_partial_cloud_normalisation_still_correct(self):
        """Normalisation must still be centred + unit sphere with partial joints."""
        positions = [(float(i) * 0.01, float(i) * 0.005, 0.5) for i in range(NUM_JOINTS)]
        valid_flags = [True] * 15 + [False] * 10
        hand = _make_hand(positions, valid_flags)
        pc = build_point_cloud(hand)
        mean = pc.points.mean(axis=0)
        norms = np.linalg.norm(pc.points, axis=1)
        np.testing.assert_allclose(mean, np.zeros(3), atol=1e-5)
        assert norms.max() <= 1.0 + 1e-5


# ──────────────────────────────────────────────────────────────────────────────
# Tests: degenerate edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_joints_at_same_position(self):
        """
        All joints at the same point → scale=0 (degenerate).
        Should not raise; returns a zero array.
        """
        positions = [(0.05, 0.1, 0.3)] * NUM_JOINTS
        hand = _make_hand(positions)
        pc = build_point_cloud(hand)
        assert pc is not None
        # Centred points are all zero; scale=0; points should be all-zero
        np.testing.assert_allclose(pc.points, np.zeros((NUM_JOINTS, 3)), atol=1e-6)

    def test_frame_index_default_minus_one(self):
        hand = _all_valid_hand()
        pc = build_point_cloud(hand)
        assert pc.frame_index == -1

    def test_output_is_not_aliased_to_input(self):
        """Modifying the output array must not affect subsequent calls."""
        hand = _all_valid_hand(seed=10)
        pc1 = build_point_cloud(hand)
        original = pc1.points.copy()
        pc1.points[:] = 999.0   # mutate output
        pc2 = build_point_cloud(hand)
        np.testing.assert_allclose(pc2.points, original, atol=1e-6)
