"""
report.py
---------
End-to-end EMG fatigue analysis CLI.

Runs the full pipeline on a recording and prints a fatigue/activation report:

    parse -> preprocess -> (MVC reference) -> features -> summary -> interpret

By default it uses the deterministic rule-based interpreter (no API key needed).
Pass --llm to use OpenAI (requires OPENAI_API_KEY and the openai package).
Optionally writes plots with --plots.

Usage
-----
    # Rule-based report (offline, no key):
    python3 report.py --task data/Avnish_push1_01.csv \\
        --mvc data/Avnish_push1_leftdynamo_01.csv data/Avnish_push1_righydynamo_01.csv

    # With OpenAI interpretation + plots:
    OPENAI_API_KEY=sk-... python3 report.py \\
        --task data/Avnish_push1_01.csv \\
        --mvc data/Avnish_push1_leftdynamo_01.csv data/Avnish_push1_righydynamo_01.csv \\
        --llm --plots output/

    # With a sensor->muscle mapping (JSON: {"1": "Biceps", ...}):
    python3 report.py --task data/Avnish_push1_01.csv --labels labels.json
"""

import argparse
import json
import sys
from pathlib import Path

from emg_pipeline import (
    parse_delsys_csv, preprocess_recording,
    compute_mvc_reference, combine_mvc_references,
    extract_recording_features, fatigue_summary_frame,
    build_recording_summary, interpret_fatigue,
    plot_rms_trend, plot_mdf_trend, plot_fatigue_ranking,
)


def main():
    ap = argparse.ArgumentParser(description="End-to-end EMG fatigue analysis")
    ap.add_argument("--task", required=True,
                    help="Path to the main task recording CSV")
    ap.add_argument("--mvc", nargs="*", default=[],
                    help="One or more max-effort (dynamometer) CSVs for %%MVC normalisation")
    ap.add_argument("--labels", default=None,
                    help="Optional JSON file mapping sensor_id -> muscle name")
    ap.add_argument("--task-name", default=None,
                    help="Human-readable task name for the report")
    ap.add_argument("--window-ms", type=float, default=250.0,
                    help="Feature window length in ms (default 250)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Window overlap fraction (default 0.5)")
    ap.add_argument("--notch-hz", type=float, default=50.0,
                    help="Power-line notch frequency: 50 (India) or 60 (Americas)")
    ap.add_argument("--llm", action="store_true",
                    help="Use OpenAI for interpretation (needs OPENAI_API_KEY)")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model (default gpt-4o-mini)")
    ap.add_argument("--plots", default=None,
                    help="Directory to write plots into (optional)")
    args = ap.parse_args()

    task_path = Path(args.task)
    if not task_path.exists():
        print(f"ERROR: task file not found: {task_path}", file=sys.stderr)
        sys.exit(1)

    labels = None
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)

    # ── MVC reference ─────────────────────────────────────────────────────────
    mvc = None
    if args.mvc:
        print(f"[1/4] Computing MVC reference from {len(args.mvc)} recording(s)...")
        refs = []
        for m in args.mvc:
            rec = preprocess_recording(parse_delsys_csv(m), notch_hz=args.notch_hz)
            refs.append(compute_mvc_reference(rec, window_ms=args.window_ms,
                                              overlap=args.overlap))
        mvc = combine_mvc_references(*refs)
    else:
        print("[1/4] No MVC recordings given; skipping %MVC normalisation.")

    # ── Features ──────────────────────────────────────────────────────────────
    print(f"[2/4] Parsing + preprocessing task: {task_path.name}")
    task = preprocess_recording(parse_delsys_csv(task_path), notch_hz=args.notch_hz)
    print(f"      {len(task)} sensors, {task.collection_length_s:.1f}s")

    print("[3/4] Extracting features + fatigue metrics...")
    feats = extract_recording_features(
        task, window_ms=args.window_ms, overlap=args.overlap, mvc_references=mvc,
    )

    summary = build_recording_summary(
        feats, labels=labels, task_name=args.task_name or task_path.stem,
    )

    # Optional plots
    if args.plots:
        out = Path(args.plots)
        plot_rms_trend(feats, out / "rms_trend.png", use_pct_mvc=(mvc is not None))
        plot_mdf_trend(feats, out / "mdf_trend.png")
        plot_fatigue_ranking(fatigue_summary_frame(feats), out / "fatigue_ranking.png")
        print(f"      plots written to {out}/")

    # ── Interpretation ─────────────────────────────────────────────────────────
    print("[4/4] Generating report...\n")
    client = None
    if args.llm:
        from emg_pipeline import OpenAIClient
        try:
            client = OpenAIClient(model=args.model)
        except (ImportError, ValueError) as e:
            print(f"WARNING: {e}\nFalling back to rule-based report.\n", file=sys.stderr)
            client = None

    report = interpret_fatigue(summary, client)
    print(report)


if __name__ == "__main__":
    main()
