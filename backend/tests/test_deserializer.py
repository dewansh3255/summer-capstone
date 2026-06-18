"""
test_deserializer.py
--------------------
Unit tests for the binary frame deserializer.

Tests synthesize raw bytes that exactly mirror what C# BinaryWriter produces,
verifying the deserializer without requiring hardware or the Unity runtime.

Run with:
    cd backend
    python -m pytest tests/ -v
"""

import struct
import pytest
import numpy as np

from protocol.frame_deserializer import (
    deserialize_frame,
    Frame,
    HandData,
    JOINT_NAMES,
    NUM_JOINTS,
    BYTES_PER_JOINT,
    BYTES_PER_HAND,
    HANDS_BLOB_SIZE,
    HEADER_SIZE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — build synthetic packets matching C# BinaryWriter output
# ──────────────────────────────────────────────────────────────────────────────

def _make_joint_bytes(valid: bool, x: float, y: float, z: float, depth: float) -> bytes:
    """Encode one joint exactly as C# BinaryWriter does."""
    return struct.pack("<Bffff", int(valid), x, y, z, depth)


def _make_hand_bytes(joints: list[tuple]) -> bytes:
    """
    joints: list of (valid, x, y, z, depth) tuples, length must be NUM_JOINTS.
    """
    assert len(joints) == NUM_JOINTS
    return b"".join(_make_joint_bytes(*j) for j in joints)


def _make_packet(
    frame_index: int = 1,
    timestamp: int = 123456789,
    color_w: int = 0, color_h: int = 0,
    depth_w: int = 0, depth_h: int = 0,
    left_joints: list = None,
    right_joints: list = None,
    color_blob: bytes = b"",
    depth_blob: bytes = b"",
) -> bytes:
    """
    Build a complete binary packet matching FrameSerializer.cs output.
    left_joints / right_joints: list of 25 (valid, x, y, z, depth) tuples.
    """
    if left_joints is None:
        left_joints = [(True, 0.0, 0.0, 0.5, 0.5)] * NUM_JOINTS
    if right_joints is None:
        right_joints = [(True, 0.1, 0.0, 0.5, 0.51)] * NUM_JOINTS

    hands_blob = _make_hand_bytes(left_joints) + _make_hand_bytes(right_joints)

    header = struct.pack(
        "<iqHHHHiii",
        frame_index,
        timestamp,
        color_w, color_h,
        depth_w, depth_h,
        len(color_blob),
        len(depth_blob),
        len(hands_blob),
    )
    # Order matches FrameSerializer.cs: hands → color → depth
    return header + hands_blob + color_blob + depth_blob


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHeaderParsing:
    def test_frame_index_parsed_correctly(self):
        pkt = _make_packet(frame_index=42)
        frame = deserialize_frame(pkt)
        assert frame.frame_index == 42

    def test_timestamp_parsed_correctly(self):
        ts = 9_999_999_999
        pkt = _make_packet(timestamp=ts)
        frame = deserialize_frame(pkt)
        assert frame.timestamp == ts

    def test_color_dimensions_parsed(self):
        pkt = _make_packet(color_w=640, color_h=480)
        frame = deserialize_frame(pkt)
        assert frame.color_width == 640
        assert frame.color_height == 480

    def test_depth_dimensions_parsed(self):
        pkt = _make_packet(depth_w=320, depth_h=240)
        frame = deserialize_frame(pkt)
        assert frame.depth_width == 320
        assert frame.depth_height == 240


class TestHandParsing:
    def test_both_hands_present(self):
        pkt = _make_packet()
        frame = deserialize_frame(pkt)
        assert frame.left_hand is not None
        assert frame.right_hand is not None

    def test_correct_joint_count(self):
        pkt = _make_packet()
        frame = deserialize_frame(pkt)
        assert len(frame.left_hand.joints) == NUM_JOINTS
        assert len(frame.right_hand.joints) == NUM_JOINTS

    def test_joint_names_in_order(self):
        pkt = _make_packet()
        frame = deserialize_frame(pkt)
        for i, joint in enumerate(frame.left_hand.joints):
            assert joint.name == JOINT_NAMES[i]

    def test_valid_flag_true(self):
        joints = [(True, 0.1 * i, 0.0, 0.5, 0.6) for i in range(NUM_JOINTS)]
        pkt = _make_packet(left_joints=joints, right_joints=joints)
        frame = deserialize_frame(pkt)
        assert all(j.valid for j in frame.left_hand.joints)

    def test_valid_flag_false(self):
        joints = [(False, 0.0, 0.0, 0.0, 0.0)] * NUM_JOINTS
        pkt = _make_packet(left_joints=joints, right_joints=joints)
        frame = deserialize_frame(pkt)
        assert not any(j.valid for j in frame.left_hand.joints)

    def test_coordinates_parsed_correctly(self):
        # Set distinct values per joint so we can verify each one
        joints = [(True, float(i), float(i) * 0.1, float(i) * 0.01, float(i) * 0.001)
                  for i in range(NUM_JOINTS)]
        pkt = _make_packet(left_joints=joints)
        frame = deserialize_frame(pkt)
        for i, joint in enumerate(frame.left_hand.joints):
            assert joint.valid is True
            assert abs(joint.x     - float(i))         < 1e-5
            assert abs(joint.y     - float(i) * 0.1)   < 1e-5
            assert abs(joint.z     - float(i) * 0.01)  < 1e-5
            assert abs(joint.depth - float(i) * 0.001) < 1e-5

    def test_right_hand_distinct_from_left(self):
        left  = [(True, 0.0, 0.0, 0.5, 0.5)] * NUM_JOINTS
        right = [(True, 1.0, 1.0, 1.0, 1.0)] * NUM_JOINTS
        pkt = _make_packet(left_joints=left, right_joints=right)
        frame = deserialize_frame(pkt)
        assert frame.left_hand.joints[0].x  == pytest.approx(0.0, abs=1e-5)
        assert frame.right_hand.joints[0].x == pytest.approx(1.0, abs=1e-5)

    def test_mixed_valid_joints(self):
        # First 10 valid, rest invalid
        joints = (
            [(True,  0.1, 0.2, 0.3, 0.4)] * 10 +
            [(False, 0.0, 0.0, 0.0, 0.0)] * 15
        )
        pkt = _make_packet(left_joints=joints)
        frame = deserialize_frame(pkt)
        assert frame.left_hand.valid_count() == 10


class TestHandDataMethods:
    def _build_hand(self, all_valid: bool = True) -> HandData:
        joints_data = [(all_valid, float(i), 0.0, 0.5, 0.5) for i in range(NUM_JOINTS)]
        pkt = _make_packet(left_joints=joints_data)
        return deserialize_frame(pkt).left_hand

    def test_to_point_cloud_shape_valid_only(self):
        hand = self._build_hand(all_valid=True)
        pc = hand.to_point_cloud(valid_only=True)
        assert pc.shape == (NUM_JOINTS, 3)
        assert pc.dtype == np.float32

    def test_to_point_cloud_shape_all(self):
        hand = self._build_hand(all_valid=False)
        pc = hand.to_point_cloud(valid_only=False)
        assert pc.shape == (NUM_JOINTS, 3)

    def test_to_point_cloud_valid_only_excludes_invalid(self):
        joints = (
            [(True,  1.0, 2.0, 3.0, 0.5)] * 5 +
            [(False, 0.0, 0.0, 0.0, 0.0)] * 20
        )
        pkt = _make_packet(left_joints=joints)
        hand = deserialize_frame(pkt).left_hand
        pc = hand.to_point_cloud(valid_only=True)
        assert pc.shape == (5, 3)

    def test_to_point_cloud_empty_when_no_valid(self):
        joints = [(False, 0.0, 0.0, 0.0, 0.0)] * NUM_JOINTS
        pkt = _make_packet(left_joints=joints)
        hand = deserialize_frame(pkt).left_hand
        pc = hand.to_point_cloud(valid_only=True)
        assert pc.shape == (0, 3)

    def test_is_tracked_true(self):
        hand = self._build_hand(all_valid=True)
        assert hand.is_tracked() is True

    def test_is_tracked_false(self):
        hand = self._build_hand(all_valid=False)
        assert hand.is_tracked() is False

    def test_xyz_coordinates_in_point_cloud(self):
        joints = [(True, float(i), float(i)*2, float(i)*3, 0.5) for i in range(NUM_JOINTS)]
        pkt = _make_packet(left_joints=joints)
        hand = deserialize_frame(pkt).left_hand
        pc = hand.to_point_cloud(valid_only=True)
        for i in range(NUM_JOINTS):
            assert pc[i, 0] == pytest.approx(float(i),     abs=1e-5)
            assert pc[i, 1] == pytest.approx(float(i) * 2, abs=1e-5)
            assert pc[i, 2] == pytest.approx(float(i) * 3, abs=1e-5)


class TestEdgeCases:
    def test_empty_packet_raises(self):
        with pytest.raises(ValueError, match="too short"):
            deserialize_frame(b"")

    def test_truncated_header_raises(self):
        with pytest.raises(ValueError):
            deserialize_frame(b"\x00" * 10)

    def test_hands_blob_absent_gives_none(self):
        """If handsLen=0, both hands should be None."""
        header = struct.pack(
            "<iqHHHHiii",
            0, 0,
            0, 0, 0, 0,
            0, 0, 0,   # colorLen=0, depthLen=0, handsLen=0
        )
        frame = deserialize_frame(header)
        assert frame.left_hand  is None
        assert frame.right_hand is None

    def test_color_blob_preserved(self):
        color_data = b"\xFF\x00\x00" * 10   # some fake RGB bytes
        pkt = _make_packet(color_blob=color_data)
        frame = deserialize_frame(pkt)
        assert frame.color_blob == color_data

    def test_depth_blob_preserved(self):
        depth_data = struct.pack("<10f", *[0.5 * i for i in range(10)])
        pkt = _make_packet(depth_blob=depth_data)
        frame = deserialize_frame(pkt)
        assert frame.depth_blob == depth_data

    def test_frame_repr(self):
        pkt = _make_packet(frame_index=7)
        frame = deserialize_frame(pkt)
        assert "7" in repr(frame)

    def test_packet_size_constants(self):
        """Sanity check the protocol constants match C# layout."""
        assert BYTES_PER_JOINT == 17        # 1 bool + 4×4 floats
        assert BYTES_PER_HAND  == 425       # 25 × 17
        assert HANDS_BLOB_SIZE == 850       # 2 × 425
        assert HEADER_SIZE     == 32
