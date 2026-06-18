"""
evaluate.py
-----------
Evaluation script.  Loads a trained model and a recordings directory,
then produces a full evaluation report:

  - Per-gesture classification accuracy
  - Confusion matrix (text)
  - LOSO-CV accuracy table (optional)
  - Chamfer Distance threshold analysis (optional)

No Quest headset required.

Usage
-----
    python3 evaluate.py --model models/gesture_clf.npz --recordings-dir recordings/

    # Full report including LOSO-CV and threshold analysis:
    python3 evaluate.py \\
        --model models/gesture_clf.npz \\
        --recordings-dir recordings/ \\
        --loso \\
        --threshold
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from pipeline.dataset import (
    load_dataset, tsts_split, loso_splits,
    GESTURE_LABELS, IDX_TO_LABEL, NUM_CLASSES,
)
from pipeline.classifier import GestureClassifier
from pipeline.chamfer import extract_pairs, find_optimal_threshold


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                       n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:<{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(labels)))
        print(row)


def _per_class_accuracy(cm: np.ndarray) -> dict[int, float]:
    per_class = {}
    for i in range(len(cm)):
        total = cm[i].sum()
        per_class[i] = float(cm[i, i] / total) if total > 0 else 0.0
    return per_class


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained gesture classifier"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained model .npz file"
    )
    parser.add_argument(
        "--recordings-dir", default="recordings",
        help="Directory containing recorded sessions (default: recordings/)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train/test split ratio used during training (default: 0.8)"
    )
    parser.add_argument(
        "--loso", action="store_true",
        help="Run leave-one-session-out CV evaluation"
    )
    parser.add_argument(
        "--threshold", action="store_true",
        help="Run Chamfer Distance threshold analysis"
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from '{args.model}' ...")
    try:
        clf = GestureClassifier.load(args.model)
        print(f"  {clf}")
    except FileNotFoundError:
        print(f"  ERROR: Model file not found: {args.model}")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset from '{args.recordings_dir}' ...")
    try:
        dataset = load_dataset(args.recordings_dir)
    except ValueError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"  {len(dataset)} frames  |  {len(set(dataset.session_ids))} sessions")

    # ── Test set accuracy ─────────────────────────────────────────────────────
    _, test = tsts_split(dataset, train_ratio=args.train_ratio)

    preds     = clf.predict(test.features)
    overall   = float((preds == test.labels).mean())
    cm        = _confusion_matrix(test.labels, preds, NUM_CLASSES)
    per_class = _per_class_accuracy(cm)

    print(f"\n{'='*60}")
    print(f"  OVERALL TEST ACCURACY: {overall:.3f}  ({overall*100:.1f}%)")
    print(f"{'='*60}")

    # ── Per-gesture accuracy ──────────────────────────────────────────────────
    print("\nPer-gesture accuracy (test set):")
    print(f"  {'Gesture':<16} {'Accuracy':>10}  {'Correct/Total':>14}")
    print(f"  {'-'*16} {'-'*10}  {'-'*14}")

    present_classes = sorted(set(test.labels.tolist()))
    for idx in present_classes:
        label   = IDX_TO_LABEL.get(idx, f"class_{idx}")
        acc     = per_class[idx]
        total   = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        print(f"  {label:<16} {acc:>10.3f}  {correct:>6}/{total:<6}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    present_labels = [IDX_TO_LABEL.get(i, f"c{i}") for i in present_classes]
    sub_cm = cm[np.ix_(present_classes, present_classes)]

    print("\nConfusion matrix (rows=true, cols=predicted):")
    _print_confusion_matrix(sub_cm, present_labels)

    # ── LOSO-CV ───────────────────────────────────────────────────────────────
    if args.loso:
        print("\n\nLOSO-CV (leave-one-session-out cross-validation):")
        print(f"  {'Session':<45} {'Accuracy':>10}")
        print(f"  {'-'*45} {'-'*10}")

        splits    = loso_splits(dataset)
        loso_accs = []

        for train_fold, test_fold, held_out in splits:
            clf_fold = GestureClassifier()
            clf_fold.fit(train_fold, epochs=150, verbose=False)
            acc = clf_fold.score(test_fold)
            loso_accs.append(acc)
            print(f"  {held_out:<45} {acc:>10.3f}")

        mean_acc = float(np.mean(loso_accs))
        std_acc  = float(np.std(loso_accs))
        print(f"\n  LOSO mean ± std:  {mean_acc:.3f} ± {std_acc:.3f}  "
              f"({mean_acc*100:.1f}% ± {std_acc*100:.1f}%)")

    # ── Chamfer Distance threshold analysis ───────────────────────────────────
    if args.threshold:
        print("\n\nChamfer Distance threshold analysis:")
        train, _ = tsts_split(dataset, train_ratio=args.train_ratio)
        all_clouds = [train.features[i].reshape(25, 3) for i in range(len(train))]
        all_labels = train.label_names

        n_pairs = len(all_clouds) * (len(all_clouds) - 1) // 2
        print(f"  Computing {n_pairs:,} pairwise distances ...")
        distances, pair_labels = extract_pairs(all_clouds, all_labels)

        result_t = find_optimal_threshold(distances, pair_labels)
        best     = result_t.metrics_curve[
            max(range(len(result_t.metrics_curve)),
                key=lambda i: result_t.metrics_curve[i].f1)
        ]

        print(f"\n  Optimal threshold : {result_t.optimal_threshold:.6f}")
        print(f"  At optimal threshold:")
        print(f"    F1        : {best.f1:.3f}")
        print(f"    Accuracy  : {best.accuracy:.3f}")
        print(f"    Precision : {best.precision:.3f}")
        print(f"    Recall    : {best.recall:.3f}")
        print(f"    TP={best.tp}  TN={best.tn}  FP={best.fp}  FN={best.fn}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
