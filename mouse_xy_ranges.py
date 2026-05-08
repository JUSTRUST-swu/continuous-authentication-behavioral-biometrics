"""
Per-user mouse pointer x, y extent from raw_kmt_user_*.json mouse_events.

Uses every mouse event that carries valid Coordinates (movement, clicks, etc.).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterable, Mapping, Sequence


def _iter_mouse_xy_from_user_root(
    root: Mapping[str, Any],
    data_groups: Sequence[str] = ("true_data", "false_data"),
) -> Iterable[tuple[float, float]]:
    """Yield (x, y) for each valid Coordinates entry under the given data groups."""
    for group_name in data_groups:
        group = root.get(group_name)
        if not isinstance(group, dict):
            continue
        for _test_key, test_obj in sorted(group.items()):
            if not isinstance(test_obj, dict):
                continue
            mouse_events = test_obj.get("mouse_events", [])
            if not isinstance(mouse_events, list):
                continue
            for ev in mouse_events:
                if not isinstance(ev, dict):
                    continue
                coords = ev.get("Coordinates")
                if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                    continue
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                except (TypeError, ValueError):
                    continue
                yield x, y


def user_mouse_xy_range(
    raw_user_json_path: str,
    data_groups: Sequence[str] = ("true_data", "false_data"),
) -> dict[str, Any]:
    """
    Return min/max x,y and point count for one user's JSON file.

    Keys: user_file, x_min, x_max, y_min, y_max, count,
          x_span, y_span (max - min, 0 if count < 2 along that axis conceptually;
          span is still max-min when count>=1 single point gives 0).
    """
    user_file = os.path.basename(raw_user_json_path)
    empty = {
        "user_file": user_file,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "x_span": None,
        "y_span": None,
        "count": 0,
    }

    try:
        with open(raw_user_json_path, encoding="utf-8") as f:
            root = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty

    if not isinstance(root, dict):
        return empty

    xs: list[float] = []
    ys: list[float] = []
    for x, y in _iter_mouse_xy_from_user_root(root, data_groups=data_groups):
        xs.append(x)
        ys.append(y)

    n = len(xs)
    if n == 0:
        return {**empty, "user_file": user_file}

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "user_file": user_file,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_span": x_max - x_min,
        "y_span": y_max - y_min,
        "count": n,
    }


def print_all_users_mouse_xy_ranges(
    dataset_dir: str = "./raw_kmt_dataset",
    data_groups: Sequence[str] = ("true_data", "false_data"),
    pattern: str = "raw_kmt_user_*.json",
) -> None:
    """Scan all user JSON files and print a table of x/y ranges to stdout."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
    if not paths:
        print(f"No files matched: {os.path.join(dataset_dir, pattern)}")
        return

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(user_mouse_xy_range(path, data_groups=data_groups))

    header = (
        f"{'user':<24} {'count':>8} "
        f"{'x_min':>10} {'x_max':>10} {'x_span':>10} "
        f"{'y_min':>10} {'y_max':>10} {'y_span':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["count"] == 0:
            print(f"{r['user_file']:<24} {'0':>8} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        print(
            f"{r['user_file']:<24} {r['count']:>8} "
            f"{r['x_min']:>10.2f} {r['x_max']:>10.2f} {r['x_span']:>10.2f} "
            f"{r['y_min']:>10.2f} {r['y_max']:>10.2f} {r['y_span']:>10.2f}"
        )


if __name__ == "__main__":
    print_all_users_mouse_xy_ranges()
