"""
chamfer.py
----------
Step 3 of the pipeline: Chamfer Distance computation between point clouds,
pairwise distance matrix construction, and statistical thresholding to derive
a classification decision boundary.

This is a direct implementation of the methodology from the C4GT report
(Sections 4.3, 4.5) adapted for the 25-point skeletal clouds produced by
the Quest hand tracking pipeline.

Mathematical definition
-----------------------
Given two point clouds X (shape M×3) and Y (shape N×3), the symmetric
Chamfer Distance is:

    CD(X, Y) = (1/M) * Σ_{x∈X}  min_{y∈Y}  ‖x − y‖²
             + (1/N) * Σ_{y∈Y}  min_{x∈X}  ‖x − y‖²

Each term is the mean squared nearest-neighbour distance from one cloud
to the other.  The result is symmetric and alignment-free.

Implementation notes
--------------------
- Pure NumPy / SciPy implementation (no PyTorch3D dependency at this stage).
  The C4GT report used PyTorch3D for GPU acceleration when computing 7.3 M
  pairwise comparisons.  With 25-point clouds the per-pair cost is negligible;
  CPU NumPy is fast enough for the dataset sizes expected here.
  PyTorch3D can be added later if needed for very large datasets.

- scipy.spatial.distance.cdist computes the full M×N squared-distance matrix
  in one vectorised call — no Python loops over individual points.

Statistical thresholding
------------------------
The C4GT report derived a classification threshold using the "bad ratio"
elbow method (Section 4.5).  We implement the same approach:

  1. Compute the full pairwise CD matrix for a labelled dataset.
  2. Sweep a threshold T from 0 to max(CD).
  3. At each T, compute Accuracy, Precision, Recall, F1 treating
     CD <= T as "same gesture" and CD > T as "different gesture".
  4. The optimal threshold is the T that maximises F1 (the elbow point).

This threshold is then used at inference time:
    if chamfer_distance(query, reference) <= threshold: same gesture
    else: different gesture
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Optional

from pipeline.point_cloud import PointCloud


# ──────────────────────────────────────────────────────────────────────────────
# Core distance function
# ──────────────────────────────────────────────────────────────────────────────

def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the symmetric Chamfer Distance between two point clouds.

    Parameters
    ----------
    a : np.ndarray, shape (M, 3)
        First point cloud (normalised, float32 or float64).
    b : np.ndarray, shape (N, 3)
        Second point cloud (normalised, float32 or float64).

    Returns
    -------
    float
        The symmetric Chamfer Distance (non-negative).
        Returns 0.0 if either cloud is empty.

    Notes
    -----
    Uses squared Euclidean distances internally (metric='sqeuclidean').
    This matches the C4GT report formula and PyTorch3D's knn_points
    implementation, so thresholds are directly comparable.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0

    # Full M×N squared-distance matrix — one vectorised call
    dist_matrix = cdist(a, b, metric="sqeuclidean")   # (M, N)

    # Each point in a → nearest point in b
    a_to_b = dist_matrix.min(axis=1).mean()           # scalar

    # Each point in b → nearest point in a
    b_to_a = dist_matrix.min(axis=0).mean()           # scalar

    return float(a_to_b + b_to_a)


def chamfer_distance_pc(pc_a: PointCloud, pc_b: PointCloud) -> float:
    """
    Convenience wrapper: compute Chamfer Distance between two PointCloud objects.
    """
    return chamfer_distance(pc_a.points, pc_b.points)


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise distance matrix
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_chamfer_matrix(clouds: list[np.ndarray]) -> np.ndarray:
    """
    Compute the full symmetric pairwise Chamfer Distance matrix for a list
    of point clouds.

    Parameters
    ----------
    clouds : list of np.ndarray, each shape (N_i, 3)
        List of normalised point clouds (e.g., from load_session).

    Returns
    -------
    np.ndarray, shape (K, K), dtype float64
        Symmetric distance matrix D where D[i, j] = CD(clouds[i], clouds[j]).
        Diagonal is 0.

    Notes
    -----
    Only the upper triangle is computed (K*(K-1)/2 pairs); the lower
    triangle is filled by symmetry.  This halves computation time.
    """
    K = len(clouds)
    D = np.zeros((K, K), dtype=np.float64)

    for i in range(K):
        for j in range(i + 1, K):
            d = chamfer_distance(clouds[i], clouds[j])
            D[i, j] = d
            D[j, i] = d

    return D


# ──────────────────────────────────────────────────────────────────────────────
# Statistical thresholding (C4GT Section 4.5 methodology)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdMetrics:
    """Classification metrics at a single Chamfer Distance threshold."""
    threshold:  float
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class ThresholdResult:
    """
    Full output of find_optimal_threshold().

    Attributes
    ----------
    optimal_threshold : float
        The CD value that maximises F1.
    optimal_f1 : float
        F1 score at the optimal threshold.
    metrics_curve : list[ThresholdMetrics]
        Full sweep results — one entry per threshold candidate.
        Use this to plot Precision/Recall/F1 vs threshold curves.
    """
    optimal_threshold: float
    optimal_f1:        float
    metrics_curve:     list[ThresholdMetrics]


def find_optimal_threshold(
    distances: np.ndarray,
    labels:    np.ndarray,
    n_thresholds: int = 500,
) -> ThresholdResult:
    """
    Find the Chamfer Distance threshold that maximises F1 for binary
    same/different gesture classification.

    This implements the elbow-point thresholding from the C4GT report
    (Section 4.5).  The C4GT report used 2000 partitions; 500 is sufficient
    for 25-point clouds where the CD range is narrow.

    Parameters
    ----------
    distances : np.ndarray, shape (K,)
        Flat array of pairwise Chamfer Distances (upper triangle of the
        distance matrix, flattened).
    labels : np.ndarray, shape (K,), dtype int (0 or 1)
        Ground-truth labels: 1 = same gesture pair, 0 = different gesture pair.
        Must be the same length and ordering as `distances`.
    n_thresholds : int
        Number of threshold candidates to evaluate.
        Candidates are linearly spaced from 0 to max(distances).

    Returns
    -------
    ThresholdResult
        Contains the optimal threshold, optimal F1, and the full metrics
        curve for plotting.

    Notes
    -----
    "Same gesture" is the positive class (label=1).
    A pair is predicted "same" if CD <= threshold.
    """
    distances = np.asarray(distances, dtype=np.float64)
    labels    = np.asarray(labels,    dtype=np.int32)

    assert len(distances) == len(labels), (
        f"distances length {len(distances)} != labels length {len(labels)}"
    )

    thresholds = np.linspace(0.0, distances.max(), n_thresholds)
    curve: list[ThresholdMetrics] = []
    best_f1 = -1.0
    best_t  = 0.0

    for t in thresholds:
        preds = (distances <= t).astype(np.int32)   # 1 = "same", 0 = "different"

        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        total     = len(labels)
        accuracy  = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        m = ThresholdMetrics(
            threshold=float(t),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp, tn=tn, fp=fp, fn=fn,
        )
        curve.append(m)

        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)

    return ThresholdResult(
        optimal_threshold=best_t,
        optimal_f1=best_f1,
        metrics_curve=curve,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for building the labelled pairs array from a multi-gesture dataset
# ──────────────────────────────────────────────────────────────────────────────

def extract_pairs(
    all_clouds:  list[np.ndarray],
    all_labels:  list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a flat list of point clouds and their gesture labels, compute all
    pairwise Chamfer Distances and binary same/different labels.

    Parameters
    ----------
    all_clouds : list of np.ndarray
        One point cloud per recorded frame across all gesture classes.
    all_labels : list of str
        Gesture label for each cloud (e.g. "ThumbsUp", "ASL_L").
        Must be the same length as all_clouds.

    Returns
    -------
    distances : np.ndarray, shape (K*(K-1)//2,)
        Upper-triangle pairwise Chamfer Distances.
    pair_labels : np.ndarray, shape (K*(K-1)//2,), dtype int
        1 if the pair shares the same gesture label, 0 otherwise.

    Notes
    -----
    For K clouds this produces K*(K-1)/2 pairs.
    With 100 frames per gesture × 10 gestures = 1000 clouds → ~500k pairs,
    which runs in seconds on CPU with NumPy.
    """
    assert len(all_clouds) == len(all_labels)
    K = len(all_clouds)

    distances   = []
    pair_labels = []

    for i in range(K):
        for j in range(i + 1, K):
            d = chamfer_distance(all_clouds[i], all_clouds[j])
            same = 1 if all_labels[i] == all_labels[j] else 0
            distances.append(d)
            pair_labels.append(same)

    return np.array(distances, dtype=np.float64), np.array(pair_labels, dtype=np.int32)
