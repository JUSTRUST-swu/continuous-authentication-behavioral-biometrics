"""
Build per-user JSON of segmented keyboard/mouse time series (pre-5s-window).

Run once, then main.py / loss_compare.py can use preprocessed JSON by default
when a mapped preprocessed file exists under the configured preprocessed directory.

Schema v2 adds stable session_id / segment_id (KMT test_N = session; see DEVELOPMENT.md).
"""

import argparse
import glob
import json
import os
import re

from visualize import (
    extract_keyboard_features,
    extract_mouse_features,
    load_raw_kmt_user_events,
    load_raw_kmt_user_sessions,
    split_events_by_gap,
)

SCHEMA_VERSION = 2

_USER_ID_RE = re.compile(r"user_(\d+)", re.IGNORECASE)


def _infer_user_id(raw_json_path):
    base = os.path.basename(raw_json_path)
    m = _USER_ID_RE.search(base)
    if m:
        return int(m.group(1))
    return None


def _segment_payload(segment_events, session_id, segment_id, segment_index):
    kb = extract_keyboard_features(segment_events)
    md = extract_mouse_features(segment_events)
    t_first = float(segment_events[0]["t"])
    t_last = float(segment_events[-1]["t"])
    return {
        "session_id": str(session_id),
        "segment_id": str(segment_id),
        "segment_index": int(segment_index),
        "t_first": t_first,
        "t_last": t_last,
        "keyboard_data": kb,
        "mouse_data": md,
    }


def build_preprocessed_payload(
    raw_json_path,
    data_group="true_data",
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
):
    """
    Return a dict suitable for json.dump: segmented dwell/flight/velocity series + bounds.

    KMT: each ``test_N`` is a session; gap-split within the session yields segments.
    Flat logs: one session, then gap-split.
    """
    gap_stats_total = {
        "normal_interval": 0,
        "pause_feature": 0,
        "idle_or_sequence_break": 0,
        "new_session_break": 0,
        "invalid": 0,
    }
    segments_out = []
    user_id = _infer_user_id(raw_json_path)

    try:
        sessions = load_raw_kmt_user_sessions(raw_json_path, data_group=data_group)
    except ValueError:
        # Fallback: concatenated stream as a single synthetic session
        try:
            events = load_raw_kmt_user_events(raw_json_path, data_group=data_group)
        except ValueError as exc:
            raise ValueError(f"Cannot load events from {raw_json_path}") from exc
        sessions = [("session_0000", events)] if events else []

    for session_id, events in sessions:
        if not events:
            continue
        segments, gap_stats = split_events_by_gap(
            events,
            sequence_break_seconds=sequence_break_seconds,
            session_break_seconds=session_break_seconds,
        )
        for k, v in gap_stats.items():
            gap_stats_total[k] = gap_stats_total.get(k, 0) + v

        for seg_i, segment_events in enumerate(segments):
            if not segment_events:
                continue
            segment_id = f"{session_id}_seg{seg_i:03d}"
            segments_out.append(
                _segment_payload(segment_events, session_id, segment_id, seg_i)
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": os.path.basename(raw_json_path),
        "user_id": user_id,
        "data_group": data_group,
        "sequence_break_seconds": float(sequence_break_seconds),
        "session_break_seconds": float(session_break_seconds),
        "gap_stats": gap_stats_total,
        "n_sessions": len(sessions),
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
        help="raw_kmt group key (true_data / false_data); ignored for flat logs",
    )
    p.add_argument("--sequence-break-seconds", type=float, default=10.0)
    p.add_argument("--session-break-seconds", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    written = preprocess_dataset(
        dataset_dir=args.dataset_dir,
        dataset_pattern=args.dataset_pattern,
        output_dir=args.output_dir,
        data_group=args.data_group,
        sequence_break_seconds=args.sequence_break_seconds,
        session_break_seconds=args.session_break_seconds,
    )
    print(f"Wrote {len(written)} preprocessed file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
