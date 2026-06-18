"""
train.py
--------
Offline training script.  Run this after recording gesture sessions with
server.py to train and save the gesture classifier.

No Quest headset required.  All work is done on the saved recordings.

Usage
-----
    python3 train.py --recordings-dir recordings/ --model-out models/gesture_clf.npz

    # With custom training hyperparameters:
    python3 train.py \\
        --recordings-dir recordings/ \\
        --model-out models/gesture_clf.npz \\
        --epochs 200 \\
        --lr 5e-3 \\
        --train-ratio 0.8

    # Also run LOSO-CV evaluation after training:
    python3 train.py --recordings-dir recordings/ --model-out models/clf.npz --loso

Output
------
- Prints per-epoch loss and final train/test accuracy.
- Saves trained model to --model-out path (a .npz file).
- If --loso is set, also prints per-session LOSO-CV accuracy table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pipeline.dataset import load_dataset, tsts_split, loso_splits, GESTURE_LABELS
from pipeline.classifier import GestureClassifier
from pipeline.chamfer import extract_pairs, find_optimal_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Train gesture classifier from recorded sessions"
    )
    parser.add_argument(
        "--recordings-dir", default="recordings",
        help="Directory containing recorded sessions (default: recordings/)"
    )
    parser.add_argument(
        "--model-out", default="models/gesture_clf.npz",
        help="Output path for trained model (default: models/gesture_clf.npz)"
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Training epochs (default: 150)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of each session for training; rest is test (default: 0.8)"
    )
    parser.add_argument(
        "--loso", action="store_true",
        help="Run leave-one-session-out CV after training"
    )
    parser.add_argument(
        "--threshold", action="store_true",
        help="Also compute optimal Chamfer Distance threshold on training data"
    )
    args = parser.parse_args()

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset from '{args.recordings_dir}' ...")
    try:
        dataset = load_dataset(args.recordings_dir)
    except ValueError as e:
        print(f"  ERROR: {e}")
        print("  Have you recorded any sessions yet?")
        print("  Run: python3 server.py --record --gesture <label>")
        sys.exit(1)

    print(f"  Loaded {len(dataset)} frames")
    print("  Class distribution:")
    for label, count in dataset.class_counts().items():
        print(f"    {label:<14} {count:>4} frames")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    print(f"\n[2/4] Splitting dataset (train={args.train_ratio:.0%}, test={1-args.train_ratio:.0%}) ...")
    train, test = tsts_split(dataset, train_ratio=args.train_ratio)
    print(f"  Train: {len(train)} frames   Test: {len(test)} frames")

    # ── 3. Train classifier ───────────────────────────────────────────────────
    print(f"\n[3/4] Training MLP classifier ({args.epochs} epochs, lr={args.lr}) ...")
    clf = GestureClassifier()
    result = clf.fit(
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )

    train_acc = clf.score(train)
    test_acc  = clf.score(test)
    print(f"\n  Final train accuracy : {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"  Final test  accuracy : {test_acc:.3f}  ({test_acc*100:.1f}%)")
    print(f"  Training time        : {result.duration_s:.1f}s")

    # ── 4. Save model ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving model to '{args.model_out}' ...")
    clf.save(args.model_out)
    print(f"  Saved.")

    # ── Optional: LOSO-CV ─────────────────────────────────────────────────────
    if args.loso:
        print("\n[LOSO-CV] Leave-one-session-out cross-validation ...")
        splits   = loso_splits(dataset)
        loso_accs = []

        for train_fold, test_fold, held_out in splits:
            clf_fold = GestureClassifier()
            clf_fold.fit(
                train_fold,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                verbose=False,
            )
            acc = clf_fold.score(test_fold)
            loso_accs.append(acc)
            print(f"  Held-out: {held_out:<40} acc={acc:.3f} ({acc*100:.1f}%)")

        mean_acc = np.mean(loso_accs)
        print(f"\n  LOSO-CV mean accuracy: {mean_acc:.3f} ({mean_acc*100:.1f}%)")

    # ── Optional: Chamfer Distance threshold ──────────────────────────────────
    if args.threshold:
        print("\n[Threshold] Computing optimal Chamfer Distance threshold ...")
        all_clouds = [train.features[i].reshape(25, 3) for i in range(len(train))]
        all_labels = train.label_names

        print(f"  Computing {len(all_clouds)*(len(all_clouds)-1)//2:,} pairwise distances ...")
        distances, pair_labels = extract_pairs(all_clouds, all_labels)
        result_t = find_optimal_threshold(distances, pair_labels)

        print(f"  Optimal threshold : {result_t.optimal_threshold:.6f}")
        print(f"  F1 at threshold   : {result_t.optimal_f1:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
