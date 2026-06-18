"""
point_cloud.py
--------------
Step 2 of the pipeline: convert HandData (25 skeletal joints) into a
normalised 3D point cloud ready for Chamfer Distance computation.

Design choice — Option A (sparse skeleton):
    We use the 25 XR joint positions directly as the point cloud.
    Each joint is a real measurement from the Quest's hand-tracking
    subsystem — no synthetic points are introduced.

    Rationale:
    - 25 joints are already a clean, labelled, noise-free representation.
    - Chamfer Distance is valid on any point set regardless of density.
    - The relative ordering (same-gesture CD < different-gesture CD) is
      preserved even with 25 points, which is all classification needs.

Option B (dense interpolation) — documented for future reference:
    If experiments show the 25-point clouds are not discriminative enough
    (i.e., the bad-ratio elbow curve is flat), bone-level interpolation
    can be added.  The approach would be:
      1. For each bone segment (parent_joint → child_joint), sample K
         equally-spaced points along the line.
      2. Concatenate all interpolated points with the original 25.
      3. Typical K=10 gives ~25 + 20*10 = 225 points (20 bone segments).
    Caution: K is a free parameter with no principled default; interpolated
    points carry no new information and bias the centroid towards longer
    bones.  Implement only if Option A results are insufficient.

Normalisation:
    Raw joint coordinates are in camera-local space (metres).
    Before comparison, each point cloud is:
      1. Centred  — subtract the centroid (mean of all points).
      2. Scaled   — divide by the max L2 norm, fitting the cloud inside
                    a unit sphere.  This makes Chamfer Distance invariant
                    to the distance between the hand and the headset camera.

    Orientation (rotation) is deliberately NOT normalised here.
    The C4GT report validated that Chamfer Distance is already robust to
    moderate viewpoint changes without alignment; forced rotation alignment
    (ICP) is 200× slower and unnecessary for classification thresholding.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from protocol.frame_deserializer import HandData, NUM_JOINTS

# Minimum valid joints required to produce a usable point cloud.
# Below this, the hand is too occluded to classify reliably.
MIN_VALID_JOINTS = 10


# ──────────────────────────────────────────────────────────────────────────────
# Output data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PointCloud:
    """
    A normalised 3D point cloud derived from one HandData frame.

    Attributes
    ----------
    points : np.ndarray, shape (N, 3), dtype float32
        Normalised joint positions. N <= 25 (valid joints only).
    centroid : np.ndarray, shape (3,), dtype float32
        Mean of the raw (un-normalised) valid joint positions (metres).
        Stored so the transform can be inverted if needed.
    scale : float
        Max L2 norm of the centred points (metres).
        Stored so the transform can be inverted if needed.
    valid_count : int
        Number of joints that were tracked (= N).
    frame_index : int
        Unity frame index this cloud came from (-1 if not from a live frame).
    """
    points:      np.ndarray          # (N, 3) float32, normalised
    centroid:    np.ndarray          # (3,)   float32, raw-space centroid
    scale:       float               # max L2 norm before scaling
    valid_count: int
    frame_index: int = -1

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return (
            f"PointCloud(n={self.valid_count}, "
            f"scale={self.scale:.4f}m, frame={self.frame_index})"
        )

    def is_valid(self) -> bool:
        """True if the cloud has enough points to be used for classification."""
        return self.valid_count >= MIN_VALID_JOINTS


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_point_cloud(
    hand: HandData,
    frame_index: int = -1,
) -> Optional[PointCloud]:
    """
    Convert a HandData object into a normalised PointCloud.

    Parameters
    ----------
    hand : HandData
        Parsed joint data from one frame (from frame_deserializer).
    frame_index : int, optional
        Unity frame counter — stored in the output for traceability.

    Returns
    -------
    PointCloud
        Normalised point cloud ready for Chamfer Distance computation.
    None
        If the hand has fewer than MIN_VALID_JOINTS tracked joints,
        the cloud is too sparse to be useful — None is returned so the
        caller can skip this frame cleanly.

    Notes
    -----
    Normalisation steps (in order):
        1. Extract (N, 3) raw positions of valid joints only.
        2. Compute centroid = mean over N points.
        3. Centre: subtract centroid from every point.
        4. Compute scale = max(||p_i||_2) over centred points.
        5. Scale: divide every point by scale (if scale > 0).
    Result: all points fit inside a unit sphere, centred at origin.
    """
    # ── 1. Extract raw valid-joint positions ──────────────────────────────────
    raw = hand.to_point_cloud(valid_only=True)   # (N, 3) float32

    if len(raw) < MIN_VALID_JOINTS:
        return None

    # ── 2. Centroid ───────────────────────────────────────────────────────────
    centroid = raw.mean(axis=0)                  # (3,) float32

    # ── 3. Centre ─────────────────────────────────────────────────────────────
    centred = raw - centroid                     # (N, 3)

    # ── 4. Scale ──────────────────────────────────────────────────────────────
    norms = np.linalg.norm(centred, axis=1)      # (N,)
    scale = float(norms.max())

    # ── 5. Normalise ──────────────────────────────────────────────────────────
    if scale > 1e-6:
        normalised = (centred / scale).astype(np.float32)
    else:
        # Degenerate case: all joints at (nearly) the same point.
        # float32 precision means "identical" inputs still produce tiny
        # non-zero centred values (order 1e-8).  Dividing by a scale that
        # is also ~1e-8 amplifies that noise to O(1).  Zero out explicitly.
        normalised = np.zeros_like(centred, dtype=np.float32)

    return PointCloud(
        points=normalised,
        centroid=centroid.astype(np.float32),
        scale=scale,
        valid_count=len(raw),
        frame_index=frame_index,
    )
