"""
frame_deserializer.py
---------------------
Parses the binary WebSocket frames sent by the Unity DepthVR frontend.

Binary layout is documented in full in ../PROTOCOL.md.
All values are little-endian, matching C# BinaryWriter defaults.

Usage
-----
    from protocol.frame_deserializer import deserialize_frame

    frame = deserialize_frame(raw_bytes)
    left  = frame.left_hand   # HandData or None
    right = frame.right_hand  # HandData or None
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUM_JOINTS = 25          # joints per hand (matches HandLandmarkLogger joint list)
BYTES_PER_JOINT = 17     # 1 bool + 4 floats × 4 bytes
BYTES_PER_HAND = NUM_JOINTS * BYTES_PER_JOINT   # 425
HANDS_BLOB_SIZE = BYTES_PER_HAND * 2            # 850  (left + right)
HEADER_SIZE = 32         # bytes before the hands blob

# Joint names in the exact order written by HandLandmarkLogger.cs
JOINT_NAMES = [
    "Wrist",
    "ThumbMetacarpal",  "ThumbProximal",     "ThumbDistal",       "ThumbTip",
    "IndexMetacarpal",  "IndexProximal",      "IndexIntermediate", "IndexDistal",  "IndexTip",
    "MiddleMetacarpal", "MiddleProximal",     "MiddleIntermediate","MiddleDistal", "MiddleTip",
    "RingMetacarpal",   "RingProximal",       "RingIntermediate",  "RingDistal",  "RingTip",
    "LittleMetacarpal", "LittleProximal",     "LittleIntermediate","LittleDistal","LittleTip",
]

assert len(JOINT_NAMES) == NUM_JOINTS, "Joint name list length mismatch"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JointData:
    """Single hand joint — position in camera-local space (metres)."""
    name:  str
    valid: bool
    x:     float
    y:     float
    z:     float
    depth: float   # Euclidean distance from camera origin

    def as_xyz(self) -> np.ndarray:
        """Returns (3,) float32 array. Zeros if not valid."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class HandData:
    """
    All 25 joints for one hand.
    joints: list of JointData in the canonical order (see JOINT_NAMES).
    """
    joints: list[JointData] = field(default_factory=list)

    def to_point_cloud(self, valid_only: bool = True) -> np.ndarray:
        """
        Returns an (N, 3) float32 array of [x, y, z] positions.
        If valid_only=True (default), only tracked joints are included.
        If valid_only=False, all 25 joints are returned (zeros for untracked).
        """
        if valid_only:
            pts = [j.as_xyz() for j in self.joints if j.valid]
        else:
            pts = [j.as_xyz() for j in self.joints]
        return np.stack(pts, axis=0) if pts else np.zeros((0, 3), dtype=np.float32)

    def valid_count(self) -> int:
        return sum(1 for j in self.joints if j.valid)

    def is_tracked(self) -> bool:
        """True if at least one joint is valid."""
        return self.valid_count() > 0

    def __repr__(self) -> str:
        return f"HandData(valid_joints={self.valid_count()}/{NUM_JOINTS})"


@dataclass
class Frame:
    """
    One complete frame received from the Quest.

    Attributes
    ----------
    frame_index : int
        Monotonically increasing counter from Unity.
    timestamp : int
        High-resolution Stopwatch ticks from C# (System.Diagnostics.Stopwatch).
    color_width, color_height : int
        Dimensions of the colour image (may be 0 if not sent).
    depth_width, depth_height : int
        Dimensions of the depth image (may be 0 if not sent).
    left_hand : HandData or None
        Left hand joints. None if the hand blob was empty / missing.
    right_hand : HandData or None
        Right hand joints. None if the hand blob was empty / missing.
    color_blob : bytes
        Raw colour image bytes (may be empty).
    depth_blob : bytes
        Raw depth image bytes (may be empty).
    """
    frame_index:   int
    timestamp:     int
    color_width:   int
    color_height:  int
    depth_width:   int
    depth_height:  int
    left_hand:     Optional[HandData]
    right_hand:    Optional[HandData]
    color_blob:    bytes
    depth_blob:    bytes

    def __repr__(self) -> str:
        return (
            f"Frame(idx={self.frame_index}, "
            f"left={self.left_hand}, right={self.right_hand})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_hand(data: bytes, offset: int) -> HandData:
    """
    Parse one hand (425 bytes) starting at `offset` in `data`.
    Returns a HandData object.
    """
    joints = []
    for name in JOINT_NAMES:
        valid_byte = data[offset]                       # 1 byte bool
        x, y, z, depth = struct.unpack_from("<ffff", data, offset + 1)
        joints.append(JointData(
            name=name,
            valid=bool(valid_byte),
            x=x, y=y, z=z,
            depth=depth,
        ))
        offset += BYTES_PER_JOINT
    return HandData(joints=joints)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def deserialize_frame(raw: bytes) -> Frame:
    """
    Parse a raw WebSocket binary message into a Frame object.

    Parameters
    ----------
    raw : bytes
        The full binary payload received from the WebSocket connection.

    Returns
    -------
    Frame
        Fully parsed frame. left_hand / right_hand may be None if the
        hands blob was absent (handsLen == 0).

    Raises
    ------
    ValueError
        If the packet is too short or malformed.
    """
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"Packet too short: got {len(raw)} bytes, need at least {HEADER_SIZE}."
        )

    # ── Header ────────────────────────────────────────────────────────────────
    # Offsets match FrameSerializer.cs exactly:
    #   int32  FrameIndex
    #   int64  Timestamp
    #   uint16 ColorWidth, ColorHeight, DepthWidth, DepthHeight
    #   int32  colorLen, depthLen, handsLen
    (
        frame_index,
        timestamp,
        color_width, color_height,
        depth_width, depth_height,
        color_len, depth_len, hands_len,
    ) = struct.unpack_from("<iqHHHHiii", raw, 0)

    expected = HEADER_SIZE + hands_len + color_len + depth_len
    if len(raw) < expected:
        raise ValueError(
            f"Packet body too short: got {len(raw)} bytes, "
            f"header says {expected} bytes total."
        )

    # ── Blobs (order: hands → color → depth) ──────────────────────────────────
    cursor = HEADER_SIZE
    hands_blob = raw[cursor : cursor + hands_len];   cursor += hands_len
    color_blob = raw[cursor : cursor + color_len];   cursor += color_len
    depth_blob = raw[cursor : cursor + depth_len]

    # ── Parse hand joints ─────────────────────────────────────────────────────
    left_hand  = None
    right_hand = None

    if hands_len >= HANDS_BLOB_SIZE:
        left_hand  = _parse_hand(hands_blob, 0)
        right_hand = _parse_hand(hands_blob, BYTES_PER_HAND)
    elif hands_len >= BYTES_PER_HAND:
        # Only one hand's worth of data — treat as left hand
        left_hand = _parse_hand(hands_blob, 0)

    return Frame(
        frame_index=frame_index,
        timestamp=timestamp,
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        left_hand=left_hand,
        right_hand=right_hand,
        color_blob=bytes(color_blob),
        depth_blob=bytes(depth_blob),
    )
