"""
classifier.py
-------------
Step 4 of the pipeline: gesture classifier.

Architecture
------------
A two-layer MLP that maps the 75-dim flattened point cloud feature vector
(25 joints × xyz) to one of NUM_CLASSES gesture labels.

    Input  (75)  →  FC(128) → ReLU → Dropout(0.3)
                 →  FC(64)  → ReLU → Dropout(0.3)
                 →  FC(NUM_CLASSES) → LogSoftmax

This matches the prototype MLP described in C4GT report Section 4.7 and
the BroadGestureClassifier in their code listing, with two improvements:
  1. Dropout layers for regularisation (important with small datasets).
  2. A persist/load interface so the trained model can be reused at
     inference time without retraining.

Design rationale
----------------
The classifier serves as the "broad pre-clustering" stage proposed in
the C4GT report (Section 4.7): instead of computing Chamfer Distance
between a query and every reference cloud in the dataset (O(K) cost),
the classifier first predicts the gesture class, then Chamfer Distance
is only computed against clouds of that predicted class (O(K/C) cost,
where C is the number of gesture classes).

The C4GT report estimated this reduces comparisons by 50–70%.  With
10 gesture classes in DepthVR the reduction is approximately 10×.

Training
--------
Uses standard cross-entropy loss with Adam optimiser.
Supports few-shot fine-tuning: pre-train on all sessions, then
fine-tune on a small amount of data from a new session — the same
adaptation strategy that showed the largest accuracy gains in EMGBench.

Pure NumPy / stdlib implementation — no PyTorch dependency required.
Uses only numpy for the forward pass and training loop, keeping the
deployment footprint minimal for the Quest → laptop pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.dataset import (
    GestureDataset,
    FEATURE_DIM,
    NUM_CLASSES,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
)


# ──────────────────────────────────────────────────────────────────────────────
# Activation / loss helpers (pure NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy loss."""
    n = len(targets)
    log_p = np.log(probs[np.arange(n), targets] + 1e-12)
    return float(-log_p.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Training result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Summary of one training run."""
    epochs:           int
    final_train_loss: float
    final_train_acc:  float
    loss_history:     list[float] = field(default_factory=list)
    duration_s:       float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MLP Classifier
# ──────────────────────────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Two-layer MLP gesture classifier.

    Architecture:
        FC(75→128) → ReLU → FC(128→64) → ReLU → FC(64→NUM_CLASSES)

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector (default 75 = 25 joints × 3).
    hidden_dim : int
        Width of both hidden layers (default 128).
    num_classes : int
        Number of gesture classes (default NUM_CLASSES = 10).
    dropout_rate : float
        Probability of zeroing a unit during training (default 0.3).
    seed : int
        Random seed for weight initialisation (default 42).
    """

    def __init__(
        self,
        input_dim:    int = FEATURE_DIM,
        hidden_dim:   int = 128,
        num_classes:  int = NUM_CLASSES,
        dropout_rate: float = 0.3,
        seed:         int = 42,
    ):
        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        rng = np.random.default_rng(seed)

        # He initialisation (good default for ReLU networks)
        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)

        # Weights and biases
        self.W1 = _he(input_dim,   hidden_dim)          # (75, 128)
        self.b1 = np.zeros(hidden_dim,  dtype=np.float32)
        self.W2 = _he(hidden_dim,  hidden_dim // 2)     # (128, 64)
        self.b2 = np.zeros(hidden_dim // 2, dtype=np.float32)
        self.W3 = _he(hidden_dim // 2, num_classes)     # (64, 10)
        self.b3 = np.zeros(num_classes, dtype=np.float32)

        self._trained = False

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _forward(
        self,
        X: np.ndarray,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Forward pass.  Returns (probabilities, cache) where cache holds
        intermediate activations needed for backprop.
        """
        # Layer 1
        z1 = X @ self.W1 + self.b1          # (N, 128)
        a1 = _relu(z1)

        # Dropout on layer 1 (training only)
        mask1 = np.ones_like(a1)
        if training and self.dropout_rate > 0:
            mask1 = (rng.random(a1.shape) >= self.dropout_rate).astype(np.float32)
            mask1 /= (1.0 - self.dropout_rate + 1e-8)   # inverted dropout scaling
        a1_drop = a1 * mask1

        # Layer 2
        z2 = a1_drop @ self.W2 + self.b2    # (N, 64)
        a2 = _relu(z2)

        # Dropout on layer 2
        mask2 = np.ones_like(a2)
        if training and self.dropout_rate > 0:
            mask2 = (rng.random(a2.shape) >= self.dropout_rate).astype(np.float32)
            mask2 /= (1.0 - self.dropout_rate + 1e-8)
        a2_drop = a2 * mask2

        # Output layer (no activation — softmax applied in loss)
        z3 = a2_drop @ self.W3 + self.b3    # (N, num_classes)
        probs = _softmax(z3)

        cache = dict(X=X, z1=z1, a1=a1, mask1=mask1, a1_drop=a1_drop,
                     z2=z2, a2=a2, mask2=mask2, a2_drop=a2_drop,
                     z3=z3, probs=probs)
        return probs, cache

    # ── Backward pass ─────────────────────────────────────────────────────────

    def _backward(
        self,
        cache: dict,
        targets: np.ndarray,
        lr: float,
    ) -> None:
        """Compute gradients and update weights in-place (SGD/Adam step)."""
        N = len(targets)
        probs = cache["probs"]

        # Output layer gradient (softmax + cross-entropy combined)
        dz3 = probs.copy()
        dz3[np.arange(N), targets] -= 1.0
        dz3 /= N                              # (N, num_classes)

        dW3 = cache["a2_drop"].T @ dz3        # (64, num_classes)
        db3 = dz3.sum(axis=0)

        # Layer 2 gradient
        da2_drop = dz3 @ self.W3.T            # (N, 64)
        da2      = da2_drop * cache["mask2"]
        dz2      = da2 * _relu_grad(cache["z2"])

        dW2 = cache["a1_drop"].T @ dz2        # (128, 64)
        db2 = dz2.sum(axis=0)

        # Layer 1 gradient
        da1_drop = dz2 @ self.W2.T            # (N, 128)
        da1      = da1_drop * cache["mask1"]
        dz1      = da1 * _relu_grad(cache["z1"])

        dW1 = cache["X"].T @ dz1              # (75, 128)
        db1 = dz1.sum(axis=0)

        # SGD update
        self.W3 -= lr * dW3;  self.b3 -= lr * db3
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.W1 -= lr * dW1;  self.b1 -= lr * db1

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        dataset:    GestureDataset,
        epochs:     int   = 100,
        batch_size: int   = 32,
        lr:         float = 1e-3,
        verbose:    bool  = True,
        seed:       int   = 0,
    ) -> TrainingResult:
        """
        Train the classifier on a GestureDataset.

        Parameters
        ----------
        dataset : GestureDataset
        epochs : int
            Number of full passes over the training data.
        batch_size : int
            Mini-batch size.
        lr : float
            Learning rate (SGD).
        verbose : bool
            Print loss/accuracy every 10 epochs.
        seed : int
            RNG seed for mini-batch shuffling and dropout.

        Returns
        -------
        TrainingResult
        """
        X = dataset.features   # (N, 75)
        y = dataset.labels     # (N,)
        N = len(X)

        rng = np.random.default_rng(seed)
        loss_history = []
        t0 = time.monotonic()

        for epoch in range(1, epochs + 1):
            # Shuffle
            perm = rng.permutation(N)
            X_s, y_s = X[perm], y[perm]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                Xb = X_s[start : start + batch_size]
                yb = y_s[start : start + batch_size]

                probs, cache = self._forward(Xb, training=True, rng=rng)
                loss = _cross_entropy(probs, yb)
                self._backward(cache, yb, lr)

                epoch_loss += loss
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                train_acc = self.score(dataset)
                print(
                    f"  Epoch {epoch:4d}/{epochs}  "
                    f"loss={avg_loss:.4f}  acc={train_acc:.3f}"
                )

        self._trained = True
        duration = time.monotonic() - t0
        final_acc = self.score(dataset)

        return TrainingResult(
            epochs=epochs,
            final_train_loss=loss_history[-1],
            final_train_acc=final_acc,
            loss_history=loss_history,
            duration_s=round(duration, 2),
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class indices for a batch of feature vectors.

        Parameters
        ----------
        X : np.ndarray, shape (N, 75) or (75,)

        Returns
        -------
        np.ndarray, shape (N,), dtype int32
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        probs, _ = self._forward(X, training=False)
        return probs.argmax(axis=1).astype(np.int32)

    def predict_label(self, cloud: np.ndarray) -> str:
        """
        Predict the gesture label for a single (N,3) point cloud.

        Parameters
        ----------
        cloud : np.ndarray, shape (N, 3), dtype float32
            Normalised point cloud from build_point_cloud().

        Returns
        -------
        str  — gesture label, e.g. "ThumbsUp"
        """
        from pipeline.dataset import cloud_to_feature
        feature = cloud_to_feature(cloud)[np.newaxis, :]
        idx = int(self.predict(feature)[0])
        return IDX_TO_LABEL.get(idx, f"unknown({idx})")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for a batch of feature vectors.

        Returns
        -------
        np.ndarray, shape (N, NUM_CLASSES), dtype float32
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        probs, _ = self._forward(X, training=False)
        return probs.astype(np.float32)

    def score(self, dataset: GestureDataset) -> float:
        """
        Compute classification accuracy on a GestureDataset.
        """
        preds = self.predict(dataset.features)
        return float((preds == dataset.labels).mean())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save model weights and hyperparameters to a .npz file.

        Parameters
        ----------
        path : str | Path
            Output file path.  '.npz' extension added automatically if absent.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            str(path),
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            meta=np.array([
                self.input_dim,
                self.hidden_dim,
                self.num_classes,
            ], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GestureClassifier":
        """
        Load a saved model from a .npz file.

        Parameters
        ----------
        path : str | Path
            Path to a file previously saved with save().

        Returns
        -------
        GestureClassifier with weights restored.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        data = np.load(str(path))
        meta = data["meta"]
        input_dim, hidden_dim, num_classes = int(meta[0]), int(meta[1]), int(meta[2])

        clf = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        clf.W1 = data["W1"].astype(np.float32)
        clf.b1 = data["b1"].astype(np.float32)
        clf.W2 = data["W2"].astype(np.float32)
        clf.b2 = data["b2"].astype(np.float32)
        clf.W3 = data["W3"].astype(np.float32)
        clf.b3 = data["b3"].astype(np.float32)
        clf._trained = True
        return clf

    def __repr__(self) -> str:
        trained = "trained" if self._trained else "untrained"
        return (
            f"GestureClassifier("
            f"{self.input_dim}→{self.hidden_dim}→{self.hidden_dim//2}"
            f"→{self.num_classes}, {trained})"
        )
