import glob
import json
import logging
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from visualize import (
    apply_clip_log_transform,
    build_windows,
    compute_window_features,
    extract_keyboard_features,
    extract_mouse_features,
    load_raw_kmt_user_events,
    split_events_by_gap,
)


FEATURE_COLUMNS = [
    "dwell_mean",
    "dwell_std",
    "flight_mean",
    "flight_std",
    "velocity_mean",
    "velocity_std",
]


def setup_logger(log_path="results/main/logs/evaluate.log"):
    """Create console + file logger for long-running evaluation."""
    logger = logging.getLogger("evaluate")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _ensure_parent_dir(log_path)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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

    df = pd.DataFrame(rows, columns=columns)
    return apply_clip_log_transform(df, FEATURE_COLUMNS)


def resolve_preprocessed_json_path(raw_user_json_path, preprocessed_dir):
    """
    Resolve preprocessed JSON path under preprocessed_dir.

    Candidate mapping order:
    - raw_kmt_user_XXXX.json -> preprocessed_kmt_user_XXXX.json
    - any_name.json -> preprocessed_any_name.json
    - any_name.json -> any_name.json

    Returns first existing candidate; otherwise None.
    """
    if not preprocessed_dir or not str(preprocessed_dir).strip():
        return None
    base = os.path.basename(raw_user_json_path)
    if not base.endswith(".json"):
        return None

    candidates = []
    if base.startswith("raw_kmt_user_"):
        candidates.append(base.replace("raw_kmt_", "preprocessed_kmt_", 1))
    candidates.append(f"preprocessed_{base}")
    candidates.append(base)

    root = str(preprocessed_dir).strip()
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_feature_df_from_preprocessed_json(
    preprocessed_json_path,
    window_size=5.0,
    stride=1.0,
):
    """
    Build window feature rows from preprocess.py output (segmented keyboard/mouse series).
    Windowing uses each segment's [t_first, t_last] timeline (same as raw segment bounds).
    """
    columns = ["window_start", "window_end"] + FEATURE_COLUMNS
    with open(preprocessed_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if int(data.get("schema_version", 1)) != 1:
        raise ValueError(f"Unsupported preprocess schema_version in {preprocessed_json_path}")

    segments = data.get("segments")
    if not isinstance(segments, list):
        segments = []

    rows = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        kb = seg.get("keyboard_data")
        md = seg.get("mouse_data")
        if not isinstance(kb, dict) or not isinstance(md, dict):
            continue
        try:
            t0 = float(seg["t_first"])
            t1 = float(seg["t_last"])
        except (KeyError, TypeError, ValueError):
            continue

        pseudo_events = [{"t": t0}, {"t": t1}]
        windows = build_windows(pseudo_events, window_size=window_size, stride=stride)
        for window_start, window_end in windows:
            rows.append(
                compute_window_features(
                    events=pseudo_events,
                    keyboard_data=kb,
                    mouse_data=md,
                    window_start=window_start,
                    window_end=window_end,
                    window_size=window_size,
                )
            )

    df = pd.DataFrame(rows, columns=columns)
    return apply_clip_log_transform(df, FEATURE_COLUMNS)


def build_feature_df_for_user(
    raw_user_json_path,
    preprocessed_dir="results/preprocessed_logs",
    window_size=5.0,
    stride=1.0,
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
    logger=None,
):
    """
    Build feature rows for one user using true_data only.

    If ``preprocessed_dir`` is set and a mapped preprocessed file exists, load that
    JSON (from preprocess.py) instead of re-parsing raw events and re-deriving
    keyboard/mouse time series.
    """
    empty_cols = ["window_start", "window_end", *FEATURE_COLUMNS, "data_group"]
    pp_path = resolve_preprocessed_json_path(raw_user_json_path, preprocessed_dir)
    if pp_path:
        if logger is not None:
            logger.info("Using preprocessed JSON: %s", pp_path)
        frame = build_feature_df_from_preprocessed_json(
            pp_path, window_size=window_size, stride=stride
        )
    else:
        if preprocessed_dir and str(preprocessed_dir).strip() and logger is not None:
            logger.debug(
                "No matched preprocessed file under %s; using raw JSON if available",
                str(preprocessed_dir).strip(),
            )

        if not os.path.isfile(raw_user_json_path):
            return pd.DataFrame(columns=empty_cols)

        try:
            events = load_raw_kmt_user_events(raw_user_json_path, data_group="true_data")
        except ValueError:
            return pd.DataFrame(columns=empty_cols)

        frame = build_feature_df_from_events(
            events,
            window_size=window_size,
            stride=stride,
            sequence_break_seconds=sequence_break_seconds,
            session_break_seconds=session_break_seconds,
        )

    if frame.empty:
        return pd.DataFrame(columns=empty_cols)

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


def _fit_student_t(values):
    x = values[np.isfinite(values)]
    n = len(x)
    if n < 2:
        return None

    df, loc, scale = stats.t.fit(x)
    if not np.isfinite(scale) or scale <= 0:
        return None

    ll = float(np.sum(stats.t.logpdf(x, df, loc=loc, scale=scale)))
    if not np.isfinite(ll):
        return None

    aic, bic = _calc_aic_bic(ll, num_params=3, n_samples=n)
    return {
        "model": "Student-t",
        "n_used": n,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "params": {"df": float(df), "loc": float(loc), "scale": float(scale)},
    }


def evaluate_feature_models(feature_name, values, user_file=None):
    """Fit four candidate distributions and return one-row-per-model results."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n_total = int(np.sum(np.isfinite(x)))

    fitters = [_fit_gaussian, _fit_lognormal, _fit_gamma, _fit_weibull, _fit_student_t]
    rows = []
    for fitter in fitters:
        result = fitter(x)
        if result is None:
            continue
        row = {
            "feature": feature_name,
            "model": result["model"],
            "n_total": n_total,
            "n_used": int(result["n_used"]),
            "log_likelihood": result["log_likelihood"],
            "aic": result["aic"],
            "bic": result["bic"],
            "params": str(result["params"]),
        }
        if user_file is not None:
            row["user_file"] = user_file
        rows.append(row)
    return rows


def _prepare_values_for_model(values, model_name):
    """Apply model-specific value filtering before fitting/QQ plotting."""
    x = values[np.isfinite(values)]
    if model_name in ("Log-normal", "Gamma", "Weibull"):
        x = x[x > 0]
    return x


def _theoretical_quantiles_for_model(model_name, probs, params):
    """Return model quantiles at probabilities using fitted params."""
    if model_name == "Gaussian":
        return stats.norm.ppf(probs, loc=params["mu"], scale=params["sigma"])
    if model_name == "Log-normal":
        return stats.lognorm.ppf(
            probs,
            params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "Gamma":
        return stats.gamma.ppf(
            probs,
            params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "Weibull":
        return stats.weibull_min.ppf(
            probs,
            params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "Student-t":
        return stats.t.ppf(
            probs,
            params["df"],
            loc=params["loc"],
            scale=params["scale"],
        )
    raise ValueError(f"Unknown model: {model_name}")


def _qq_r2_rmse(theory_q, sample_q):
    """Compute R2 and RMSE for QQ points (sample as target, theory as predictor)."""
    resid = sample_q - theory_q
    rmse = float(np.sqrt(np.mean(resid * resid)))
    denom = float(np.sum((sample_q - np.mean(sample_q)) ** 2))
    if denom <= 0:
        r2 = np.nan
    else:
        r2 = float(1.0 - np.sum(resid * resid) / denom)
    return r2, rmse


def generate_qq_plots_by_feature(all_features_df, output_dir="qqplots", logger=None):
    """
    Generate QQ plots for each feature x model on merged feature values.
    Also return a metrics DataFrame with R2/RMSE per plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for QQ plot generation.") from exc

    if logger is None:
        logger = setup_logger()

    os.makedirs(output_dir, exist_ok=True)

    model_fitters = {
        "Gaussian": _fit_gaussian,
        "Log-normal": _fit_lognormal,
        "Gamma": _fit_gamma,
        "Weibull": _fit_weibull,
        "Student-t": _fit_student_t,
    }
    metric_rows = []

    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(all_features_df[feature], errors="coerce").to_numpy(dtype=float)
        for model_name, fitter in model_fitters.items():
            x_used = _prepare_values_for_model(values, model_name)
            fit_result = fitter(values)
            if fit_result is None or len(x_used) < 2:
                logger.warning("Skip QQ plot: feature=%s model=%s (insufficient valid data)", feature, model_name)
                continue

            sample_q = np.sort(x_used)
            probs = (np.arange(1, len(sample_q) + 1) - 0.5) / len(sample_q)
            theory_q = _theoretical_quantiles_for_model(model_name, probs, fit_result["params"])

            valid = np.isfinite(sample_q) & np.isfinite(theory_q)
            if np.sum(valid) < 2:
                logger.warning("Skip QQ plot: feature=%s model=%s (non-finite quantiles)", feature, model_name)
                continue

            sample_q = sample_q[valid]
            theory_q = theory_q[valid]
            r2, rmse = _qq_r2_rmse(theory_q, sample_q)

            fig = plt.figure(figsize=(6, 6))
            plt.scatter(theory_q, sample_q, s=10, alpha=0.7)
            lo = min(float(np.min(theory_q)), float(np.min(sample_q)))
            hi = max(float(np.max(theory_q)), float(np.max(sample_q)))
            plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            plt.xlabel("Theoretical quantiles")
            plt.ylabel("Sample quantiles")
            plt.title(f"QQ: {feature} - {model_name}")
            plt.tight_layout()

            model_slug = model_name.lower().replace("-", "_").replace(" ", "_")
            out_path = os.path.join(output_dir, f"{feature}__{model_slug}_qq.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info("Saved QQ plot: %s", out_path)
            metric_rows.append(
                {
                    "feature": feature,
                    "model": model_name,
                    "n_points": int(len(sample_q)),
                    "r2": r2,
                    "rmse": rmse,
                    "qq_plot_path": out_path,
                }
            )

    if not metric_rows:
        return pd.DataFrame(columns=["feature", "model", "n_points", "r2", "rmse", "qq_plot_path"])
    return pd.DataFrame(metric_rows).sort_values(["feature", "model"])


def aggregate_per_user_model_fits(per_user_df):
    """
    Combine per-user fit scores into a final model choice per feature.

    Methods:
    1) majority_vote_aic / majority_vote_bic: each user picks argmin AIC/BIC among
       fitted models for that feature; overall winner is the mode across users.
    2) weighted_mean_aic / weighted_mean_bic: for each (feature, model), compute
       weighted average of AIC/BIC with weights n_used; pick minimum.
    3) sum_log_likelihood: sum LL across users per (feature, model); pick maximum.
    """
    summary_rows = []
    vote_detail_rows = []
    weighted_rows = []

    for feature in FEATURE_COLUMNS:
        sub = per_user_df[per_user_df["feature"] == feature].copy()
        if sub.empty:
            continue

        users = sub["user_file"].dropna().unique()
        votes_aic = []
        votes_bic = []
        for uf in users:
            u = sub[sub["user_file"] == uf]
            if u.empty:
                continue
            votes_aic.append(u.loc[u["aic"].idxmin(), "model"])
            votes_bic.append(u.loc[u["bic"].idxmin(), "model"])

        cnt_aic = Counter(votes_aic)
        cnt_bic = Counter(votes_bic)
        # Tie-break: higher votes first, then model name lexicographic.
        best_vote_aic = sorted(cnt_aic.items(), key=lambda x: (-x[1], x[0]))[0][0] if cnt_aic else None
        best_vote_bic = sorted(cnt_bic.items(), key=lambda x: (-x[1], x[0]))[0][0] if cnt_bic else None

        for model_name, n_votes in sorted(cnt_aic.items()):
            vote_detail_rows.append(
                {
                    "feature": feature,
                    "criterion": "majority_vote_aic",
                    "model": model_name,
                    "n_votes": n_votes,
                    "n_users": len(votes_aic),
                }
            )
        for model_name, n_votes in sorted(cnt_bic.items()):
            vote_detail_rows.append(
                {
                    "feature": feature,
                    "criterion": "majority_vote_bic",
                    "model": model_name,
                    "n_votes": n_votes,
                    "n_users": len(votes_bic),
                }
            )

        # Weighted mean AIC/BIC per model (weight = n_used for that user's fit)
        weighted_for_feature = []
        for model_name, g in sub.groupby("model"):
            w = g["n_used"].to_numpy(dtype=float)
            if len(w) == 0 or np.sum(w) <= 0:
                continue
            weighted_for_feature.append(
                {
                    "feature": feature,
                    "model": model_name,
                    "weighted_mean_aic": float(np.average(g["aic"], weights=w)),
                    "weighted_mean_bic": float(np.average(g["bic"], weights=w)),
                    "total_n_used": int(g["n_used"].sum()),
                }
            )
        weighted_rows.extend(weighted_for_feature)

        if weighted_for_feature:
            wdf = pd.DataFrame(weighted_for_feature)
            best_weighted_aic = wdf.loc[wdf["weighted_mean_aic"].idxmin(), "model"]
            best_weighted_bic = wdf.loc[wdf["weighted_mean_bic"].idxmin(), "model"]
        else:
            best_weighted_aic = None
            best_weighted_bic = None

        sum_ll = sub.groupby("model")["log_likelihood"].sum().sort_values(ascending=False)
        best_sum_ll = sum_ll.index[0] if len(sum_ll) else None

        summary_rows.append(
            {
                "feature": feature,
                "best_majority_vote_aic": best_vote_aic,
                "best_majority_vote_bic": best_vote_bic,
                "n_users_with_votes": len(votes_aic),
                "best_weighted_mean_aic": best_weighted_aic,
                "best_weighted_mean_bic": best_weighted_bic,
                "best_sum_log_likelihood": best_sum_ll,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("feature")
    vote_detail_df = pd.DataFrame(vote_detail_rows)
    weighted_detail_df = pd.DataFrame(weighted_rows).sort_values(["feature", "model"])
    return summary_df, vote_detail_df, weighted_detail_df


def compute_pearson_by_stat_group(all_features_df):
    """
    Compute Pearson correlation matrices by stat suffix group:
    - mean features only
    - std features only
    """
    mean_cols = [c for c in FEATURE_COLUMNS if c.endswith("_mean")]
    std_cols = [c for c in FEATURE_COLUMNS if c.endswith("_std")]

    mean_corr = pd.DataFrame()
    std_corr = pd.DataFrame()

    if mean_cols:
        mean_corr = all_features_df[mean_cols].corr(method="pearson")
    if std_cols:
        std_corr = all_features_df[std_cols].corr(method="pearson")

    return mean_corr, std_corr


def plot_pearson_heatmap(corr_df, title, output_path):
    """Save Pearson correlation matrix as a heatmap image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for Pearson heatmap plotting.") from exc

    if corr_df.empty:
        return

    values = corr_df.to_numpy(dtype=float)
    labels = list(corr_df.columns)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate correlation values for readability.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_all_users(
    dataset_dir="./logs",
    dataset_pattern="*.json",
    preprocessed_dir="results/preprocessed_logs",
    output_per_user_models_csv="",
    output_aggregated_summary_csv="",
    output_vote_detail_csv="",
    output_weighted_detail_csv="",
    output_pearson_mean_csv="",
    output_pearson_std_csv="",
    output_pearson_mean_png="",
    output_pearson_std_png="",
    output_feature_csv="",
    output_qq_dir="",
    output_qq_metrics_csv="",
    window_size=5.0,
    stride=1.0,
    sequence_break_seconds=10.0,
    session_break_seconds=30.0,
    logger=None,
):
    """
    End-to-end: collect all user features, fit models per user, aggregate scores.

    For each (user, feature) the four candidate distributions are fitted. Results are
    saved in long form. Final model choice per feature uses:
    - majority vote on per-user argmin AIC / argmin BIC
    - minimum weighted-mean AIC/BIC (weights = n_used)
    - maximum sum of log-likelihoods across users
    """
    if logger is None:
        logger = setup_logger()

    def _enabled(path):
        return isinstance(path, str) and path.strip() != ""

    started_at = time.perf_counter()
    pattern = os.path.join(dataset_dir, dataset_pattern)
    user_files = sorted(glob.glob(pattern))
    if not user_files:
        raise FileNotFoundError(f"No user files found with pattern: {pattern}")
    logger.info(
        "Found %d input files from %s/%s (sequence_break=%.1fs, session_break=%.1fs)",
        len(user_files),
        dataset_dir,
        dataset_pattern,
        sequence_break_seconds,
        session_break_seconds,
    )
    if preprocessed_dir and str(preprocessed_dir).strip():
        logger.info("Preprocessed JSON dir (used when file exists): %s", preprocessed_dir)

    all_frames = []
    result_rows = []
    num_with_rows = 0
    for idx, user_path in enumerate(user_files, start=1):
        user_name = os.path.basename(user_path)
        logger.info("Processing user %d/%d: %s", idx, len(user_files), user_name)
        df_user = build_feature_df_for_user(
            user_path,
            preprocessed_dir=preprocessed_dir,
            window_size=window_size,
            stride=stride,
            sequence_break_seconds=sequence_break_seconds,
            session_break_seconds=session_break_seconds,
            logger=logger,
        )
        if df_user.empty:
            logger.warning("No rows produced for %s", user_name)
            continue
        df_user["user_file"] = user_name
        all_frames.append(df_user)
        num_with_rows += 1
        logger.info("Rows from %s: %d; fitting 4 models x %d features", user_name, len(df_user), len(FEATURE_COLUMNS))
        for feature in FEATURE_COLUMNS:
            result_rows.extend(
                evaluate_feature_models(feature, df_user[feature], user_file=user_name)
            )

    if not all_frames:
        raise ValueError("No feature rows produced from dataset.")

    if _enabled(output_feature_csv):
        _ensure_parent_dir(output_feature_csv)
    if _enabled(output_per_user_models_csv):
        _ensure_parent_dir(output_per_user_models_csv)
    if _enabled(output_aggregated_summary_csv):
        _ensure_parent_dir(output_aggregated_summary_csv)
    if _enabled(output_vote_detail_csv):
        _ensure_parent_dir(output_vote_detail_csv)
    if _enabled(output_weighted_detail_csv):
        _ensure_parent_dir(output_weighted_detail_csv)
    if _enabled(output_pearson_mean_csv):
        _ensure_parent_dir(output_pearson_mean_csv)
    if _enabled(output_pearson_std_csv):
        _ensure_parent_dir(output_pearson_std_csv)
    if _enabled(output_pearson_mean_png):
        _ensure_parent_dir(output_pearson_mean_png)
    if _enabled(output_pearson_std_png):
        _ensure_parent_dir(output_pearson_std_png)
    if _enabled(output_qq_metrics_csv):
        _ensure_parent_dir(output_qq_metrics_csv)
    if _enabled(output_qq_dir):
        os.makedirs(output_qq_dir, exist_ok=True)

    all_features_df = pd.concat(all_frames, ignore_index=True)
    if _enabled(output_feature_csv):
        all_features_df.to_csv(output_feature_csv, index=False)
        logger.info(
            "Saved merged feature rows: %s (rows=%d, users_with_rows=%d)",
            output_feature_csv,
            len(all_features_df),
            num_with_rows,
        )

    if not result_rows:
        raise ValueError("No model fit results were produced.")

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(["user_file", "feature", "aic"], ascending=[True, True, True])
    if _enabled(output_per_user_models_csv):
        result_df.to_csv(output_per_user_models_csv, index=False)
        logger.info("Saved per-user model fits: %s (%d rows)", output_per_user_models_csv, len(result_df))

    aggregated_summary, vote_detail, weighted_detail = aggregate_per_user_model_fits(result_df)
    if _enabled(output_aggregated_summary_csv):
        aggregated_summary.to_csv(output_aggregated_summary_csv, index=False)
        logger.info("Saved aggregated summary: %s", output_aggregated_summary_csv)
    if _enabled(output_vote_detail_csv):
        vote_detail.to_csv(output_vote_detail_csv, index=False)
        logger.info("Saved vote detail: %s", output_vote_detail_csv)
    if _enabled(output_weighted_detail_csv):
        weighted_detail.to_csv(output_weighted_detail_csv, index=False)
        logger.info("Saved weighted means by model: %s", output_weighted_detail_csv)

    mean_corr = None
    std_corr = None
    pearson_enabled = any(
        [
            _enabled(output_pearson_mean_csv),
            _enabled(output_pearson_std_csv),
            _enabled(output_pearson_mean_png),
            _enabled(output_pearson_std_png),
        ]
    )
    if pearson_enabled:
        mean_corr, std_corr = compute_pearson_by_stat_group(all_features_df)
        if _enabled(output_pearson_mean_csv):
            mean_corr.to_csv(output_pearson_mean_csv, index=True)
            logger.info("Saved Pearson(mean-only): %s", output_pearson_mean_csv)
        if _enabled(output_pearson_std_csv):
            std_corr.to_csv(output_pearson_std_csv, index=True)
            logger.info("Saved Pearson(std-only): %s", output_pearson_std_csv)
        if _enabled(output_pearson_mean_png):
            plot_pearson_heatmap(mean_corr, "Pearson Correlation (Mean Features)", output_pearson_mean_png)
            logger.info("Saved Pearson(mean heatmap): %s", output_pearson_mean_png)
        if _enabled(output_pearson_std_png):
            plot_pearson_heatmap(std_corr, "Pearson Correlation (Std Features)", output_pearson_std_png)
            logger.info("Saved Pearson(std heatmap): %s", output_pearson_std_png)
    else:
        logger.info("Skip Pearson computation (all Pearson outputs disabled).")

    qq_metrics_df = None
    qq_enabled = _enabled(output_qq_dir) or _enabled(output_qq_metrics_csv)
    if qq_enabled and _enabled(output_qq_dir):
        qq_metrics_df = generate_qq_plots_by_feature(all_features_df, output_dir=output_qq_dir, logger=logger)
        logger.info("Saved QQ plots to: %s", output_qq_dir)
        if _enabled(output_qq_metrics_csv):
            qq_metrics_df.to_csv(output_qq_metrics_csv, index=False)
            logger.info("Saved QQ metrics: %s", output_qq_metrics_csv)
    elif qq_enabled:
        logger.info("Skip QQ computation (output_qq_dir is disabled).")
    else:
        logger.info("Skip QQ computation (QQ outputs disabled).")

    logger.info("Evaluation finished in %.2f sec", time.perf_counter() - started_at)
    return (
        all_features_df,
        result_df,
        aggregated_summary,
        vote_detail,
        weighted_detail,
        qq_metrics_df,
        mean_corr,
        std_corr,
    )


if __name__ == "__main__":
    logger = setup_logger()
    try:
        (
            _,
            per_user_fits,
            agg_summary,
            vote_detail,
            weighted_detail,
            qq_metrics,
            mean_corr,
            std_corr,
        ) = evaluate_all_users(
            dataset_dir="./logs",
            dataset_pattern="*.json",
            preprocessed_dir="results/preprocessed_logs",
            output_per_user_models_csv="results/main/tables/model_fit_per_user_all_models.csv",
            output_aggregated_summary_csv="results/main/tables/model_fit_aggregated_summary.csv",
            output_vote_detail_csv="results/main/tables/model_fit_aggregated_vote_counts.csv",
            output_weighted_detail_csv="results/main/tables/model_fit_aggregated_weighted_by_model.csv",
            # output_pearson_mean_csv="results/main/tables/pearson_mean_features.csv",
            # output_pearson_std_csv="results/main/tables/pearson_std_features.csv",
            # output_pearson_mean_png="results/main/plots/pearson/pearson_mean_features.png",
            # output_pearson_std_png="results/main/plots/pearson/pearson_std_features.png",
            output_feature_csv="results/main/tables/features_all_users_merged.csv",
            # output_qq_dir="results/main/plots/qq",
            # output_qq_metrics_csv="results/main/tables/model_fit_qq_metrics.csv",
            window_size=5.0,
            stride=1.0,
            sequence_break_seconds=10.0,
            session_break_seconds=30.0,
            logger=logger,
        )
        # logger.info("Aggregated model choice (per-user fits combined):\n%s", agg_summary.to_string(index=False))
        # logger.info("Per-user fit rows: %d", len(per_user_fits))
        # logger.info("QQ metric rows: %d", len(qq_metrics))
        # logger.info("Pearson(mean) shape: %s", mean_corr.shape)
        # logger.info("Pearson(std) shape: %s", std_corr.shape)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
