import json
import math
import os

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dwell_mean",
    "dwell_std",
    "flight_mean",
    "flight_std",
    "velocity_mean",
    "velocity_std",
]


def _mean_or_nan(values):
    """Return mean when values exist, otherwise NaN."""
    if not values:
        return np.nan
    return float(np.mean(values))


def _std_or_nan(values):
    """Return std(ddof=1) when sample count >= 2, otherwise NaN."""
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _clip_1_99_percentile(values):
    """1~99 percentile clipping on a 1d array; 2+ samples required to alter bounds."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return arr
    lo, hi = np.percentile(arr, [1.0, 99.0])
    return np.clip(arr, lo, hi)


def apply_clip_log_transform(df, feature_columns):
    """
    Apply 1~99 percentile clipping and log1p transform to each feature column.
    - Percentile bounds are computed from finite values in each column.
    - Values <= -1 become NaN before log1p.
    """
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            continue
        arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(arr)
        if not np.any(valid):
            out[col] = np.nan
            continue

        vals = arr[valid]
        if vals.size >= 2:
            lo, hi = np.percentile(vals, [1.0, 99.0])
            vals = np.clip(vals, lo, hi)

        vals = vals.astype(float)
        vals[vals <= -1] = np.nan
        pos_mask = np.isfinite(vals)
        vals[pos_mask] = np.log1p(vals[pos_mask])

        transformed = np.full_like(arr, np.nan, dtype=float)
        transformed[valid] = vals
        out[col] = transformed
    return out


def load_events(path):
    """
    Load JSONL events from file.
    - Skip malformed lines and invalid event objects.
    - Keep only events with valid numeric timestamp `t`.
    - Return events sorted by timestamp.
    """
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore malformed JSON lines
                    continue

                if not isinstance(obj, dict):
                    continue

                t_val = obj.get("t")
                try:
                    t = float(t_val)
                except (TypeError, ValueError):
                    continue

                obj["t"] = t
                events.append(obj)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read input file: {path}") from exc

    events.sort(key=lambda e: e["t"])
    return events


def _parse_epoch_seconds(value):
    """
    Parse epoch to seconds.
    - Legacy dataset uses epoch seconds (float-like string).
    - New frontend logs may send epoch milliseconds (13-digit int).
    """
    t = float(value)
    if t > 1.0e11:
        # Millisecond epoch -> seconds
        t = t / 1000.0
    return t


def _normalize_mouse_coordinates(x_val, y_val, monitor_width, monitor_height):
    """
    Normalize mouse coordinates by monitor size when valid dimensions exist.
    Returns (x_norm, y_norm) in [~0, ~1] scale; falls back to original values.
    """
    if monitor_width is None or monitor_height is None:
        return x_val, y_val
    try:
        w = float(monitor_width)
        h = float(monitor_height)
    except (TypeError, ValueError):
        return x_val, y_val
    if w <= 0 or h <= 0:
        return x_val, y_val
    return float(x_val) / w, float(y_val) / h


def _append_events_from_key_mouse_lists(
    events,
    key_events,
    mouse_events,
    monitor_width=None,
    monitor_height=None,
):
    if not isinstance(key_events, list):
        key_events = []
    if not isinstance(mouse_events, list):
        mouse_events = []

    for ev in key_events:
        if not isinstance(ev, dict):
            continue
        try:
            t = _parse_epoch_seconds(ev.get("Epoch"))
        except (TypeError, ValueError):
            continue

        event_name = str(ev.get("Event", "")).strip().lower()
        key_name = ev.get("Key")
        key_value = str(key_name) if key_name is not None else None

        if event_name == "pressed":
            events.append({"type": "keydown", "key": key_value, "t": t})
        elif event_name == "released":
            events.append({"type": "keyup", "key": key_value, "t": t})

    for ev in mouse_events:
        if not isinstance(ev, dict):
            continue
        try:
            t = _parse_epoch_seconds(ev.get("Epoch"))
        except (TypeError, ValueError):
            continue

        event_name = str(ev.get("Event", "")).strip().lower()
        coords = ev.get("Coordinates")
        x_val, y_val = None, None
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                x_val = float(coords[0])
                y_val = float(coords[1])
                x_val, y_val = _normalize_mouse_coordinates(
                    x_val, y_val, monitor_width, monitor_height
                )
            except (TypeError, ValueError):
                x_val, y_val = None, None

        if event_name == "movement":
            if x_val is None or y_val is None:
                continue
            events.append({"type": "mousemove", "x": x_val, "y": y_val, "t": t})
        elif event_name.endswith("press"):
            button = event_name.replace(" press", "")
            events.append({"type": "mousedown", "button": button, "x": x_val, "y": y_val, "t": t})
        elif event_name.endswith("release"):
            button = event_name.replace(" release", "")
            events.append({"type": "mouseup", "button": button, "x": x_val, "y": y_val, "t": t})


def load_raw_kmt_user_events(path, data_group="true_data"):
    """
    Load keyboard/mouse logs and convert to internal event schema.

    Supported input schemas:
    1) Legacy raw_kmt:
       - root[data_group][test_n]["key_events"]
       - root[data_group][test_n]["mouse_events"]
    2) New flat session log:
       - root["key_events"]
       - root["mouse_events"]
       - optional root["session"]["monitor_width"], root["session"]["monitor_height"]
         -> coordinates are normalized by monitor size.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format: {path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read input file: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root object: {path}")

    events = []
    group = raw.get(data_group)
    if isinstance(group, dict):
        for _, test_obj in sorted(group.items()):
            if not isinstance(test_obj, dict):
                continue
            session_obj = test_obj.get("session") if isinstance(test_obj.get("session"), dict) else {}
            mw = session_obj.get("monitor_width")
            mh = session_obj.get("monitor_height")
            _append_events_from_key_mouse_lists(
                events,
                test_obj.get("key_events", []),
                test_obj.get("mouse_events", []),
                monitor_width=mw,
                monitor_height=mh,
            )
    elif isinstance(raw.get("key_events"), list) or isinstance(raw.get("mouse_events"), list):
        session_obj = raw.get("session") if isinstance(raw.get("session"), dict) else {}
        mw = session_obj.get("monitor_width")
        mh = session_obj.get("monitor_height")
        _append_events_from_key_mouse_lists(
            events,
            raw.get("key_events", []),
            raw.get("mouse_events", []),
            monitor_width=mw,
            monitor_height=mh,
        )
    else:
        raise ValueError(
            f"Unsupported log schema: expected data_group '{data_group}' or top-level key_events/mouse_events in {path}"
        )

    events.sort(key=lambda e: e["t"])
    return events


def classify_interval_gap(gap_seconds):
    """
    Classify event interval by policy (used for session/sequence segmentation).
    """
    if gap_seconds < 0:
        return "invalid"
    if gap_seconds <= 1.0:
        return "normal_interval"
    if gap_seconds < 10.0:
        return "pause_feature"
    if gap_seconds < 30.0:
        return "idle_or_sequence_break"
    return "new_session_break"


def split_events_by_gap(events, sequence_break_seconds=10.0, session_break_seconds=30.0):
    """
    Split sorted events into contiguous segments by interval policy.
    - gap >= sequence_break_seconds: sequence break
    - gap >= session_break_seconds: new session break

    Returns:
    - segments: list of event segments
    - gap_stats: category counts by interval policy
    """
    if not events:
        return [], {
            "normal_interval": 0,
            "pause_feature": 0,
            "idle_or_sequence_break": 0,
            "new_session_break": 0,
            "invalid": 0,
        }

    segments = []
    gap_stats = {
        "normal_interval": 0,
        "pause_feature": 0,
        "idle_or_sequence_break": 0,
        "new_session_break": 0,
        "invalid": 0,
    }
    current = [events[0]]
    for i in range(1, len(events)):
        prev_t = events[i - 1]["t"]
        cur = events[i]
        gap = cur["t"] - prev_t
        gap_type = classify_interval_gap(gap)
        gap_stats[gap_type] += 1

        if gap >= session_break_seconds or gap >= sequence_break_seconds:
            segments.append(current)
            current = [cur]
        else:
            current.append(cur)
    segments.append(current)
    return segments, gap_stats


def extract_keyboard_features(events):
    """
    Build keyboard-derived time series used by each sliding window.
    Anchors:
    - dwell: keydown time
    - flight: next keydown time
    - digraph: current keydown time
    - pause: current keydown time
    - typing_rate source: keydown time
    """
    keydowns = []
    keyups = []
    for e in events:
        etype = e.get("type")
        if etype == "keydown":
            keydowns.append({"key": e.get("key"), "t": e["t"]})
        elif etype == "keyup":
            keyups.append({"key": e.get("key"), "t": e["t"]})

    # 1) Dwell time: pair keydown with next keyup of same key.
    waiting_down = {}
    dwell_times = []
    dwell_anchor_times = []

    keyboard_events = [e for e in events if e.get("type") in ("keydown", "keyup")]
    for e in keyboard_events:
        etype = e.get("type")
        key = e.get("key")
        t = e["t"]
        if etype == "keydown":
            waiting_down.setdefault(key, []).append(t)
        elif etype == "keyup":
            q = waiting_down.get(key)
            if q:
                down_t = q.pop(0)
                dwell = t - down_t
                if 0 < dwell <= 2.0:
                    dwell_times.append(dwell)
                    dwell_anchor_times.append(down_t)

    # 2) Flight time: previous keyup -> next keydown
    # Build press pairs from dwell matching output, then connect consecutive presses.
    presses = sorted(zip(dwell_anchor_times, dwell_times), key=lambda x: x[0])
    # Reconstruct press-up from dwell definition.
    press_down = [p[0] for p in presses]
    press_up = [p[0] + p[1] for p in presses]

    flight_times = []
    flight_anchor_times = []
    for i in range(1, len(press_down)):
        flight = press_down[i] - press_up[i - 1]
        if flight < 0:
            continue
        if flight <= 1.0:
            flight_times.append(flight)
            flight_anchor_times.append(press_down[i])  # next keydown anchor

    # 3) Digraph latency: consecutive keydown intervals
    keydown_times = [k["t"] for k in keydowns]
    digraph_latencies = []
    digraph_anchor_times = []
    pauses = []
    pause_anchor_times = []

    for i in range(len(keydown_times) - 1):
        cur_t = keydown_times[i]
        nxt_t = keydown_times[i + 1]
        delta = nxt_t - cur_t
        if 0 < delta <= 1.0:
            digraph_latencies.append(delta)
            digraph_anchor_times.append(cur_t)
        elif delta > 1.0:
            pauses.append(delta)
            pause_anchor_times.append(cur_t)

    return {
        "keydown_times": keydown_times,
        "dwell_times": dwell_times,
        "dwell_anchor_times": dwell_anchor_times,
        "flight_times": flight_times,
        "flight_anchor_times": flight_anchor_times,
        "digraph_latencies": digraph_latencies,
        "digraph_anchor_times": digraph_anchor_times,
        "pauses": pauses,
        "pause_anchor_times": pause_anchor_times,
    }


def extract_mouse_features(events):
    """
    Build mouse-derived time series used by each sliding window.
    Raw segment velocities (px/s) are stored; each window applies 1~99% clipping on positive
    velocities in that window, then velocity_mean / velocity_std use ln(velocity).
    Anchors:
    - velocity/movement_distance: second mousemove time
    - acceleration: second velocity sample time (v2 time)
    - angle_change: current vector time
    - click_interval: current mousedown time
    """
    mousemoves = [e for e in events if e.get("type") == "mousemove"]
    mousedown_times = [e["t"] for e in events if e.get("type") == "mousedown"]

    velocities = []
    velocity_anchor_times = []
    distances = []
    angles = []
    angle_anchor_times = []

    # Mouse move segments
    for i in range(1, len(mousemoves)):
        prev = mousemoves[i - 1]
        cur = mousemoves[i]

        x1, y1, t1 = prev.get("x"), prev.get("y"), prev["t"]
        x2, y2, t2 = cur.get("x"), cur.get("y"), cur["t"]

        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        dt = t2 - t1

        if dt <= 0 or dt < 0.005:
            continue

        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 2.0:
            continue

        velocity = distance / dt
        velocities.append(velocity)
        velocity_anchor_times.append(t2)
        distances.append(distance)
        angles.append(math.atan2(dy, dx))
        angle_anchor_times.append(t2)

    # Mouse acceleration from consecutive velocity samples
    accelerations = []
    acceleration_anchor_times = []
    for i in range(1, len(velocities)):
        v1, v2 = velocities[i - 1], velocities[i]
        t1, t2 = velocity_anchor_times[i - 1], velocity_anchor_times[i]
        dt = t2 - t1
        if dt <= 0:
            continue
        accelerations.append((v2 - v1) / dt)
        acceleration_anchor_times.append(t2)  # v2 time anchor

    # Angle change between consecutive valid vectors
    angle_changes = []
    angle_change_anchor_times = []
    for i in range(1, len(angles)):
        a1, a2 = angles[i - 1], angles[i]
        diff = a2 - a1
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        angle_changes.append(abs(diff))
        angle_change_anchor_times.append(angle_anchor_times[i])

    # Click intervals from mousedown events
    click_intervals = []
    click_interval_anchor_times = []
    for i in range(1, len(mousedown_times)):
        cur_t = mousedown_times[i]
        prev_t = mousedown_times[i - 1]
        interval = cur_t - prev_t
        if 0 < interval <= 5.0:
            click_intervals.append(interval)
            click_interval_anchor_times.append(cur_t)

    return {
        "velocity_values": velocities,
        "velocity_anchor_times": velocity_anchor_times,
        "distance_values": distances,
        "acceleration_values": accelerations,
        "acceleration_anchor_times": acceleration_anchor_times,
        "angle_change_values": angle_changes,
        "angle_change_anchor_times": angle_change_anchor_times,
        "mousedown_times": mousedown_times,
        "click_interval_values": click_intervals,
        "click_interval_anchor_times": click_interval_anchor_times,
    }


def build_windows(events, window_size=5.0, stride=1.0):
    """
    Build [start, end) windows from first event time to last event time.
    """
    if not events:
        return []

    start_time = events[0]["t"]
    last_time = events[-1]["t"]
    windows = []

    cur = start_time
    # Include windows whose start is <= last event time.
    while cur <= last_time:
        windows.append((cur, cur + window_size))
        cur += stride

    return windows


def _filter_by_window(values, anchors, w_start, w_end):
    """Pick values whose anchor is inside [w_start, w_end)."""
    selected = []
    for val, t in zip(values, anchors):
        if w_start <= t < w_end:
            selected.append(val)
    return selected


def compute_window_features(events, keyboard_data, mouse_data, window_start, window_end, window_size):
    """
    Compute all requested features for one window.
    Clip/log transformation is applied later in DataFrame stage.
    """
    row = {
        "window_start": window_start,
        "window_end": window_end,
    }

    # Keyboard features
    dwell_w = _filter_by_window(
        keyboard_data["dwell_times"], keyboard_data["dwell_anchor_times"], window_start, window_end
    )
    flight_w = _filter_by_window(
        keyboard_data["flight_times"], keyboard_data["flight_anchor_times"], window_start, window_end
    )
    row["dwell_mean"] = _mean_or_nan(dwell_w)
    row["dwell_std"] = _std_or_nan(dwell_w)
    row["flight_mean"] = _mean_or_nan(flight_w)
    row["flight_std"] = _std_or_nan(flight_w)

    # Mouse features
    velocity_w = _filter_by_window(
        mouse_data["velocity_values"], mouse_data["velocity_anchor_times"], window_start, window_end
    )
    row["velocity_mean"] = _mean_or_nan(velocity_w)
    row["velocity_std"] = _std_or_nan(velocity_w)

    return row


def main(input_path, output_csv_path):
    """
    End-to-end pipeline:
    1) Load events
    2) Extract keyboard/mouse derived series
    3) Build sliding windows
    4) Compute window-level features
    5) Save CSV and return DataFrame
    """
    events = load_events(input_path)

    # If there is no event, return empty DataFrame with fixed columns.
    feature_columns = [
        "window_start",
        "window_end",
        "dwell_mean",
        "dwell_std",
        "flight_mean",
        "flight_std",
        "velocity_mean",
        "velocity_std",
    ]

    if not events:
        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_empty = pd.DataFrame(columns=feature_columns)
        df_empty.to_csv(output_csv_path, index=False)
        return df_empty

    keyboard_data = extract_keyboard_features(events)
    mouse_data = extract_mouse_features(events)
    # Save features in 5-second windows with 1-second sliding stride.
    windows = build_windows(events, window_size=5.0, stride=1.0)

    rows = []
    for w_start, w_end in windows:
        rows.append(
            compute_window_features(
                events=events,
                keyboard_data=keyboard_data,
                mouse_data=mouse_data,
                window_start=w_start,
                window_end=w_end,
                window_size=5.0,
            )
        )

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows, columns=feature_columns)
    df = apply_clip_log_transform(df, FEATURE_COLUMNS)
    df.to_csv(output_csv_path, index=False)
    return df


def main_from_raw_kmt_user(raw_json_path, output_csv_path, data_group="true_data"):
    """
    Build feature DataFrame from raw_kmt_user_N.json for a given data group.
    """
    events = load_raw_kmt_user_events(raw_json_path, data_group=data_group)

    feature_columns = [
        "window_start",
        "window_end",
        "dwell_mean",
        "dwell_std",
        "flight_mean",
        "flight_std",
        "velocity_mean",
        "velocity_std",
    ]

    if not events:
        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_empty = pd.DataFrame(columns=feature_columns)
        df_empty.to_csv(output_csv_path, index=False)
        return df_empty

    keyboard_data = extract_keyboard_features(events)
    mouse_data = extract_mouse_features(events)
    windows = build_windows(events, window_size=5.0, stride=1.0)

    rows = []
    for w_start, w_end in windows:
        rows.append(
            compute_window_features(
                events=events,
                keyboard_data=keyboard_data,
                mouse_data=mouse_data,
                window_start=w_start,
                window_end=w_end,
                window_size=5.0,
            )
        )

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows, columns=feature_columns)
    df = apply_clip_log_transform(df, FEATURE_COLUMNS)
    df.to_csv(output_csv_path, index=False)
    return df


def plot_feature_histograms(df, output_dir="results/histograms", time_bin_ms=1.0):
    """
    Plot histogram for each column in DataFrame and save as PNG.
    - FEATURE_COLUMNS are transformed with 1~99 percentile clipping + log1p first.
    - Time-based features are converted sec -> ms and binned by `time_bin_ms`.
    - Non-time features are plotted in their transformed unit with 30 bins.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for histogram plotting.") from exc

    os.makedirs(output_dir, exist_ok=True)
    plot_df = apply_clip_log_transform(df, FEATURE_COLUMNS)

    time_feature_cols = {
        "dwell_mean",
        "dwell_std",
        "flight_mean",
        "flight_std",
        "digraph_latency_mean",
        "digraph_latency_std",
        "pause_duration_mean",
        "click_interval_mean",
    }

    numeric_columns = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
    for col in numeric_columns:
        values = pd.to_numeric(plot_df[col], errors="coerce").dropna()
        if values.empty:
            continue

        plt.figure(figsize=(8, 4))

        if col in time_feature_cols:
            values_ms = values * 1000.0
            min_v = float(values_ms.min())
            max_v = float(values_ms.max())
            step = float(time_bin_ms)

            # Build stable 10ms bins, even when all values are equal.
            start = math.floor(min_v / step) * step
            end = math.ceil(max_v / step) * step
            if end <= start:
                end = start + step
            bins = np.arange(start, end + step, step)
            if len(bins) < 2:
                bins = np.array([start, start + step])

            plt.hist(values_ms, bins=bins, edgecolor="black", alpha=0.8)
            plt.xlabel(f"{col} (ms)")
            plt.title(f"Histogram: {col} ({time_bin_ms:g}ms bins)")
        else:
            plt.hist(values, bins=30, edgecolor="black", alpha=0.8)
            if col in ("velocity_mean", "velocity_std"):
                plt.xlabel(f"{col} (log1p px/s)")
                plt.title(f"Histogram: {col} (log1p, 1-99 pct clip)")
            else:
                plt.xlabel(col)
                plt.title(f"Histogram: {col}")

        plt.ylabel("count")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{col}_hist.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    # Example usage for raw_kmt_user_0001 true_data
    INPUT_PATH = "raw_kmt_dataset/raw_kmt_user_0002.json"
    OUTPUT_PATH = "results/histograms_user_0002_true/features_user_0002_true.csv"
    HIST_DIR = "results/histograms_user_0002_true"

    try:
        features_df = main_from_raw_kmt_user(INPUT_PATH, OUTPUT_PATH, data_group="true_data")
        print(f"Saved features to: {OUTPUT_PATH}")
        plot_feature_histograms(features_df, output_dir=HIST_DIR, time_bin_ms=1.0)
        print(f"Saved histogram images to: {HIST_DIR}/")
        print(features_df.head())
    except Exception as exc:
        print(f"Failed to build features: {exc}")
