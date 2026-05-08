import glob
import logging
import math
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from visualize import (
    build_windows,
    compute_window_features,
    extract_keyboard_features,
    extract_mouse_features,
    load_raw_kmt_user_events,
)


FEATURE_COLUMNS = [
    "dwell_mean",
    "dwell_std",
    "flight_mean",
    "flight_std",
    "velocity_mean",
    "velocity_std",
]


def classify_interval_gap(gap_seconds):
    """
    Classify event interval by policy.
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


def setup_logger(log_path="evaluate.log"):
    """Create console + file logger for long-running evaluation."""
    logger = logging.getLogger("evaluate")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_feature_df_from_events(
    events,
    window_size=5.0,
    stride=1.0,
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
):
    """Create window-level feature DataFrame, excluding long idle gaps via segmentation."""
    columns = ["window_start", "window_end"] + FEATURE_COLUMNS
    if not events:
        return pd.DataFrame(columns=columns)

    rows = []
    segments, _ = split_events_by_gap(
        events,
        sequence_break_seconds=sequence_break_seconds,
        session_break_seconds=session_break_seconds,
    )
    for segment_events in segments:
        keyboard_data = extract_keyboard_features(segment_events)
        mouse_data = extract_mouse_features(segment_events)
        windows = build_windows(segment_events, window_size=window_size, stride=stride)
        for window_start, window_end in windows:
            rows.append(
                compute_window_features(
                    events=segment_events,
                    keyboard_data=keyboard_data,
                    mouse_data=mouse_data,
                    window_start=window_start,
                    window_end=window_end,
                    window_size=window_size,
                )
            )

    return pd.DataFrame(rows, columns=columns)


def build_feature_df_for_user(
    raw_user_json_path,
    window_size=5.0,
    stride=1.0,
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
):
    """
    Build feature rows for one user using true_data only.
    """
    try:
        events = load_raw_kmt_user_events(raw_user_json_path, data_group="true_data")
    except ValueError:
        return pd.DataFrame(columns=["window_start", "window_end", *FEATURE_COLUMNS, "data_group"])

    frame = build_feature_df_from_events(
        events,
        window_size=window_size,
        stride=stride,
        sequence_break_seconds=sequence_break_seconds,
        session_break_seconds=session_break_seconds,
    )
    if frame.empty:
        return pd.DataFrame(columns=["window_start", "window_end", *FEATURE_COLUMNS, "data_group"])

    frame["data_group"] = "true_data"
    return frame


def _calc_aic_bic(log_likelihood, num_params, n_samples):
    """Return (AIC, BIC) from log-likelihood and parameter/sample counts."""
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * math.log(n_samples) - 2 * log_likelihood
    return float(aic), float(bic)


def _fit_gaussian(values):
    x = values[np.isfinite(values)]
    n = len(x)
    if n < 2:
        return None

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    if sigma <= 0:
        return None

    ll = float(np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma)))
    aic, bic = _calc_aic_bic(ll, num_params=2, n_samples=n)
    return {
        "model": "Gaussian",
        "n_used": n,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "params": {"mu": mu, "sigma": sigma},
    }


def _fit_lognormal(values):
    x = values[np.isfinite(values)]
    x = x[x > 0]
    n = len(x)
    if n < 2:
        return None

    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    ll = float(np.sum(stats.lognorm.logpdf(x, shape, loc=loc, scale=scale)))
    if not np.isfinite(ll):
        return None
    aic, bic = _calc_aic_bic(ll, num_params=2, n_samples=n)
    return {
        "model": "Log-normal",
        "n_used": n,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "params": {"shape": float(shape), "loc": float(loc), "scale": float(scale)},
    }


def _fit_gamma(values):
    x = values[np.isfinite(values)]
    x = x[x > 0]
    n = len(x)
    if n < 2:
        return None

    shape, loc, scale = stats.gamma.fit(x, floc=0)
    ll = float(np.sum(stats.gamma.logpdf(x, shape, loc=loc, scale=scale)))
    if not np.isfinite(ll):
        return None
    aic, bic = _calc_aic_bic(ll, num_params=2, n_samples=n)
    return {
        "model": "Gamma",
        "n_used": n,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "params": {"shape": float(shape), "loc": float(loc), "scale": float(scale)},
    }


def _fit_weibull(values):
    x = values[np.isfinite(values)]
    x = x[x > 0]
    n = len(x)
    if n < 2:
        return None

    c, loc, scale = stats.weibull_min.fit(x, floc=0)
    ll = float(np.sum(stats.weibull_min.logpdf(x, c, loc=loc, scale=scale)))
    if not np.isfinite(ll):
        return None
    aic, bic = _calc_aic_bic(ll, num_params=2, n_samples=n)
    return {
        "model": "Weibull",
        "n_used": n,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "params": {"shape": float(c), "loc": float(loc), "scale": float(scale)},
    }


def evaluate_feature_models(feature_name, values):
    """Fit four candidate distributions and return one-row-per-model results."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n_total = int(np.sum(np.isfinite(x)))

    fitters = [_fit_gaussian, _fit_lognormal, _fit_gamma, _fit_weibull]
    rows = []
    for fitter in fitters:
        result = fitter(x)
        if result is None:
            continue
        rows.append(
            {
                "feature": feature_name,
                "model": result["model"],
                "n_total": n_total,
                "n_used": int(result["n_used"]),
                "log_likelihood": result["log_likelihood"],
                "aic": result["aic"],
                "bic": result["bic"],
                "params": str(result["params"]),
            }
        )
    return rows


def evaluate_all_users(
    dataset_dir="./raw_kmt_dataset",
    output_all_models_csv="model_fit_all_users_all_models.csv",
    output_best_csv="model_fit_all_users_best.csv",
    output_feature_csv="features_all_users_merged.csv",
    window_size=5.0,
    stride=1.0,
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
    logger=None,
):
    """End-to-end: collect all user features, fit models, save AIC/BIC results."""
    if logger is None:
        logger = setup_logger()

    started_at = time.perf_counter()
    pattern = os.path.join(dataset_dir, "raw_kmt_user_*.json")
    user_files = sorted(glob.glob(pattern))
    if not user_files:
        raise FileNotFoundError(f"No user files found with pattern: {pattern}")
    logger.info(
        "Found %d user files from %s (true_data only, sequence_break=%.1fs, session_break=%.1fs)",
        len(user_files),
        dataset_dir,
        sequence_break_seconds,
        session_break_seconds,
    )

    all_frames = []
    num_with_rows = 0
    for idx, user_path in enumerate(user_files, start=1):
        user_name = os.path.basename(user_path)
        logger.info("Processing user %d/%d: %s", idx, len(user_files), user_name)
        df_user = build_feature_df_for_user(
            user_path,
            window_size=window_size,
            stride=stride,
            sequence_break_seconds=sequence_break_seconds,
            session_break_seconds=session_break_seconds,
        )
        if df_user.empty:
            logger.warning("No rows produced for %s", user_name)
            continue
        df_user["user_file"] = os.path.basename(user_path)
        all_frames.append(df_user)
        num_with_rows += 1
        logger.info("Rows from %s: %d", user_name, len(df_user))

    if not all_frames:
        raise ValueError("No feature rows produced from dataset.")

    all_features_df = pd.concat(all_frames, ignore_index=True)
    all_features_df.to_csv(output_feature_csv, index=False)
    logger.info(
        "Saved merged feature rows: %s (rows=%d, users_with_rows=%d)",
        output_feature_csv,
        len(all_features_df),
        num_with_rows,
    )

    result_rows = []
    for feature in FEATURE_COLUMNS:
        logger.info("Fitting models for feature: %s", feature)
        result_rows.extend(evaluate_feature_models(feature, all_features_df[feature]))
        logger.info("Completed model fitting for feature: %s", feature)

    if not result_rows:
        raise ValueError("No model fit results were produced.")

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(["feature", "aic", "bic"], ascending=[True, True, True])
    result_df.to_csv(output_all_models_csv, index=False)
    logger.info("Saved all model fits: %s", output_all_models_csv)

    best_rows = []
    for feature in FEATURE_COLUMNS:
        sub = result_df[result_df["feature"] == feature]
        if sub.empty:
            continue
        best_aic_row = sub.loc[sub["aic"].idxmin()]
        best_bic_row = sub.loc[sub["bic"].idxmin()]
        best_rows.append(
            {
                "feature": feature,
                "best_model_by_aic": best_aic_row["model"],
                "best_aic": float(best_aic_row["aic"]),
                "best_model_by_bic": best_bic_row["model"],
                "best_bic": float(best_bic_row["bic"]),
            }
        )

    best_df = pd.DataFrame(best_rows).sort_values("feature")
    best_df.to_csv(output_best_csv, index=False)
    logger.info("Saved best-model summary: %s", output_best_csv)
    logger.info("Evaluation finished in %.2f sec", time.perf_counter() - started_at)
    return all_features_df, result_df, best_df


if __name__ == "__main__":
    logger = setup_logger()
    try:
        _, all_model_scores, best_summary = evaluate_all_users(
            dataset_dir="./raw_kmt_dataset",
            output_all_models_csv="model_fit_all_users_all_models.csv",
            output_best_csv="model_fit_all_users_best.csv",
            output_feature_csv="features_all_users_merged.csv",
            window_size=5.0,
            stride=1.0,
            sequence_break_seconds=10.0,
            session_break_seconds=30.0,
            logger=logger,
        )
        logger.info("Best model by feature (AIC/BIC):\n%s", best_summary.to_string(index=False))
        logger.info("Total fitted rows: %d", len(all_model_scores))
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
