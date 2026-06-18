"""
dataset.py
----------
Step 4 (support): builds a labelled dataset from recorded sessions for
classifier training and LOSO-CV evaluation.

Responsibilities
----------------
- Loads all saved recording sessions from a recordings/ directory.
- Assembles a flat dataset of (point_cloud, label) pairs.
- Provides a leave-one-subject-out (LOSO) split utility, matching the
  OOD evaluation methodology from the EMGBench paper (Section 4.3).
- Provides a train/test split along the time axis (TSTS — train-test
  split for time series), which avoids data leakage by ensuring test
  data is always recorded after training data.

Gesture vocabulary
------------------
The 10 gestures in the DepthVR frontend (from Assets/Gesture Videos/):
    ASL_L, ASL_Y, Five, Four, One, Spiderman, Spock, Three, Thumbs_Up, Two

Each gesture is recorded at 3 orientations (0°, 45°, 90°).
The classifier treats all orientations of the same gesture as the same
class — orientation robustness is provided by the normalisation in
point_cloud.py (centring + unit-sphere scaling).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.recorder import load_session, list_sessions


# ──────────────────────────────────────────────────────────────────────────────
# Gesture vocabulary — matches DepthVR frontend gesture video names
# ──────────────────────────────────────────────────────────────────────────────

GESTURE_LABELS = [
    "ASL_L",
    "ASL_Y",
    "Five",
    "Four",
    "One",
    "Spiderman",
    "Spock",
    "Three",
    "Thumbs_Up",
    "Two",
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(GESTURE_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES   = len(GESTURE_LABELS)
FEATURE_DIM   = 25 * 3   # 25 joints × (x, y, z) = 75


# ──────────────────────────────────────────────────────────────────────────────
# Dataset container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GestureDataset:
    """
    A flat, labelled dataset of gesture point clouds.

    Attributes
    ----------
    features : np.ndarray, shape (N, 75), dtype float32
        Each row is one frame's 25 joints flattened: [x0,y0,z0, x1,y1,z1, …].
        Frames with fewer than 25 valid joints are zero-padded to 75 dims.
    labels : np.ndarray, shape (N,), dtype int32
        Integer class index for each frame (see LABEL_TO_IDX).
    label_names : list[str]
        String gesture label for each frame (same ordering as labels).
    session_ids : list[str]
        Which recording session each frame came from.
        Used for LOSO-CV splits.
    """
    features:    np.ndarray       # (N, 75) float32
    labels:      np.ndarray       # (N,)    int32
    label_names: list[str]
    session_ids: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        unique = sorted(set(self.label_names))
        return (
            f"GestureDataset(n={len(self)}, "
            f"classes={len(unique)}, "
            f"sessions={len(set(self.session_ids))})"
        )

    def class_counts(self) -> dict[str, int]:
        """Returns {gesture_label: frame_count} sorted alphabetically."""
        counts: dict[str, int] = {}
        for name in self.label_names:
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def cloud_to_feature(cloud: np.ndarray) -> np.ndarray:
    """
    Flatten a (N, 3) point cloud into a fixed-length (75,) feature vector.

    If N < 25, the remaining slots are zero-padded.
    If N > 25, only the first 25 points are used (shouldn't happen with
    our pipeline, but handled defensively).

    Parameters
    ----------
    cloud : np.ndarray, shape (N, 3), dtype float32
        Normalised point cloud from build_point_cloud().

    Returns
    -------
    np.ndarray, shape (75,), dtype float32
    """
    feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    n = min(len(cloud), 25)
    feature[: n * 3] = cloud[:n].flatten()
    return feature


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(recordings_dir: str | Path) -> GestureDataset:
    """
    Load all recorded sessions from recordings_dir into a GestureDataset.

    Only sessions whose gesture_label appears in GESTURE_LABELS are included.
    Sessions with unknown labels are skipped with a warning.

    Parameters
    ----------
    recordings_dir : str | Path
        Root directory containing session subdirectories
        (produced by Recorder).

    Returns
    -------
    GestureDataset
        Flat labelled dataset ready for training or evaluation.

    Raises
    ------
    ValueError
        If no valid sessions are found.
    """
    sessions = list_sessions(recordings_dir)
    if not sessions:
        raise ValueError(f"No sessions found in: {recordings_dir}")

    all_features:    list[np.ndarray] = []
    all_labels:      list[int]        = []
    all_label_names: list[str]        = []
    all_session_ids: list[str]        = []

    skipped = 0
    for meta in sessions:
        gesture = meta.get("gesture_label", "")
        if gesture not in LABEL_TO_IDX:
            print(f"  [dataset] Skipping unknown gesture label: '{gesture}'")
            skipped += 1
            continue

        session_path = Path(meta["path"])
        try:
            clouds, _ = load_session(session_path)
        except FileNotFoundError:
            print(f"  [dataset] Skipping empty/broken session: {session_path.name}")
            skipped += 1
            continue

        class_idx   = LABEL_TO_IDX[gesture]
        session_id  = session_path.name

        for cloud in clouds:
            all_features.append(cloud_to_feature(cloud))
            all_labels.append(class_idx)
            all_label_names.append(gesture)
            all_session_ids.append(session_id)

    if not all_features:
        raise ValueError(
            f"No valid frames loaded from {recordings_dir}. "
            f"Skipped {skipped} sessions."
        )

    return GestureDataset(
        features=np.stack(all_features).astype(np.float32),
        labels=np.array(all_labels, dtype=np.int32),
        label_names=all_label_names,
        session_ids=all_session_ids,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Train / test splits
# ──────────────────────────────────────────────────────────────────────────────

def tsts_split(
    dataset: GestureDataset,
    train_ratio: float = 0.8,
) -> tuple[GestureDataset, GestureDataset]:
    """
    Train-Test Split for Time Series (TSTS).

    Splits each session's frames at `train_ratio` along the time axis:
    the first train_ratio of frames go to train, the rest to test.
    This respects temporal ordering — test data is always "later" than
    training data, preventing data leakage.

    Matches the TSTS evaluation methodology from EMGBench (Section 4.3).

    Parameters
    ----------
    dataset : GestureDataset
    train_ratio : float
        Fraction of each session's frames to use for training (default 0.8).

    Returns
    -------
    train_dataset, test_dataset : GestureDataset
    """
    train_idx = []
    test_idx  = []

    unique_sessions = list(dict.fromkeys(dataset.session_ids))   # preserve order

    for session_id in unique_sessions:
        idxs = [i for i, s in enumerate(dataset.session_ids) if s == session_id]
        split = max(1, int(len(idxs) * train_ratio))
        train_idx.extend(idxs[:split])
        test_idx.extend(idxs[split:])

    def _subset(indices: list[int]) -> GestureDataset:
        idx = np.array(indices)
        return GestureDataset(
            features=dataset.features[idx],
            labels=dataset.labels[idx],
            label_names=[dataset.label_names[i] for i in indices],
            session_ids=[dataset.session_ids[i] for i in indices],
        )

    return _subset(train_idx), _subset(test_idx)


def loso_splits(
    dataset: GestureDataset,
) -> list[tuple[GestureDataset, GestureDataset, str]]:
    """
    Leave-One-Session-Out (LOSO) cross-validation splits.

    For each session, returns (train_dataset, test_dataset, held_out_session_id).
    The test set is always the held-out session; train is all other sessions.

    Matches the LOSO-CV methodology from EMGBench (Section 4.3).

    Parameters
    ----------
    dataset : GestureDataset

    Returns
    -------
    list of (train, test, session_id) tuples — one per unique session.
    """
    unique_sessions = list(dict.fromkeys(dataset.session_ids))
    splits = []

    for held_out in unique_sessions:
        train_idx = [i for i, s in enumerate(dataset.session_ids) if s != held_out]
        test_idx  = [i for i, s in enumerate(dataset.session_ids) if s == held_out]

        def _subset(indices: list[int]) -> GestureDataset:
            idx = np.array(indices)
            return GestureDataset(
                features=dataset.features[idx],
                labels=dataset.labels[idx],
                label_names=[dataset.label_names[i] for i in indices],
                session_ids=[dataset.session_ids[i] for i in indices],
            )

        splits.append((_subset(train_idx), _subset(test_idx), held_out))

    return splits
