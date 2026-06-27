"""
Build per-user JSON of segmented keyboard/mouse time series (pre-5s-window).

Run once, then main.py / loss_compare.py can use preprocessed JSON by default
when a mapped preprocessed file exists under the configured preprocessed directory.
"""

import argparse
import glob
import json
import os

from visualize import (
    extract_keyboard_features,
    extract_mouse_features,
    load_raw_kmt_user_events,
    split_events_by_gap,
)

SCHEMA_VERSION = 1


def build_preprocessed_payload(
    raw_json_path,
    data_group="true_data",
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
):
    """
    Return a dict suitable for json.dump: segmented dwell/flight/velocity series + bounds.
    """
    try:
        events = load_raw_kmt_user_events(raw_json_path, data_group=data_group)
    except ValueError as exc:
        raise ValueError(f"Cannot load events from {raw_json_path}") from exc

    segments_out = []
    gap_stats_total = {
        "normal_interval": 0,
        "pause_feature": 0,
        "idle_or_sequence_break": 0,
        "new_session_break": 0,
        "invalid": 0,
    }

    if events:
        segments, gap_stats = split_events_by_gap(
            events,
            sequence_break_seconds=sequence_break_seconds,
            session_break_seconds=session_break_seconds,
        )
        for k, v in gap_stats.items():
            gap_stats_total[k] = gap_stats_total.get(k, 0) + v

        for segment_events in segments:
            if not segment_events:
                continue
            kb = extract_keyboard_features(segment_events)
            md = extract_mouse_features(segment_events)
            t_first = float(segment_events[0]["t"])
            t_last = float(segment_events[-1]["t"])
            segments_out.append(
                {
                    "t_first": t_first,
                    "t_last": t_last,
                    "keyboard_data": kb,
                    "mouse_data": md,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": os.path.basename(raw_json_path),
        "data_group": data_group,
        "sequence_break_seconds": float(sequence_break_seconds),
        "session_break_seconds": float(session_break_seconds),
        "gap_stats": gap_stats_total,
        "segments": segments_out,
    }


def preprocess_user_file(
    raw_json_path,
    output_dir,
    data_group="true_data",
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
):
    """Write one preprocessed JSON with compatible mapped naming."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(raw_json_path)
    if not base.endswith(".json"):
        raise ValueError(f"Expected a .json file: {base}")
    if base.startswith("raw_kmt_user_"):
        out_name = base.replace("raw_kmt_", "preprocessed_kmt_", 1)
    else:
        out_name = f"preprocessed_{base}"
    out_path = os.path.join(output_dir, out_name)

    payload = build_preprocessed_payload(
        raw_json_path,
        data_group=data_group,
        sequence_break_seconds=sequence_break_seconds,
        session_break_seconds=session_break_seconds,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return out_path


def preprocess_dataset(
    dataset_dir="./logs",
    dataset_pattern="*.json",
    output_dir="results/preprocessed_logs",
    data_group="true_data",
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
):
    pattern = os.path.join(dataset_dir, dataset_pattern)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    written = []
    for p in paths:
        written.append(
            preprocess_user_file(
                p,
                output_dir=output_dir,
                data_group=data_group,
                sequence_break_seconds=sequence_break_seconds,
                session_break_seconds=session_break_seconds,
            )
        )
    return written


def parse_args():
    p = argparse.ArgumentParser(description="Write preprocessed time-series JSON per user.")
    p.add_argument("--dataset-dir", default="./logs", help="Folder with session JSON logs")
    p.add_argument(
        "--dataset-pattern",
        default="*.json",
        help="Glob pattern under dataset-dir (e.g. *.json, user_*.json)",
    )
    p.add_argument(
        "--output-dir",
        default="results/preprocessed_logs",
        help="Output folder for preprocessed JSON files",
    )
    p.add_argument(
        "--data-group",
        default="true_data",
        help="Key under legacy raw JSON (ignored for flat logs with top-level key_events/mouse_events)",
    )
    p.add_argument("--sequence-break-seconds", type=float, default=10.0)
    p.add_argument("--session-break-seconds", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    paths = preprocess_dataset(
        dataset_dir=args.dataset_dir,
        dataset_pattern=args.dataset_pattern,
        output_dir=args.output_dir,
        data_group=args.data_group,
        sequence_break_seconds=args.sequence_break_seconds,
        session_break_seconds=args.session_break_seconds,
    )
    print(f"Wrote {len(paths)} file(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
