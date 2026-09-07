import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from main import (
    FEATURE_COLUMNS,
    _prepare_values_for_model,
    build_feature_df_for_user,
    evaluate_feature_models,
    get_model_fitters,
    resolve_preprocessed_json_path,
)


# Default registry without GMM (opt-in via get_model_fitters(include_gmm=True))
MODEL_FITTERS = get_model_fitters(include_gmm=False)


def user_json_path(user_id, dataset_dir="./raw_kmt_dataset"):
    return os.path.join(dataset_dir, f"raw_kmt_user_{int(user_id):04d}.json")


def discover_user_ids(dataset_dir="./raw_kmt_dataset"):
    """Auto-discover user ids from raw_kmt_user_*.json filenames."""
    paths = _list_input_files(dataset_dir, "raw_kmt_user_*.json")
    user_ids = []
    for path in paths:
        base = os.path.basename(path)
        try:
            uid = int(base.replace("raw_kmt_user_", "").replace(".json", ""))
        except ValueError:
            continue
        user_ids.append(uid)
    return sorted(set(user_ids))


def select_models_by_aic_on_user(
    train_df,
    include_gmm=False,
    gmm_n_components=2,
    gmm_random_state=0,
):
    """
    For each FEATURE_COLUMN, fit candidate models on the train user and
    pick the AIC-minimizing model. Returns feature -> model_name.

    GMM is considered only when include_gmm=True.
    """
    model_map = {}
    for feature in FEATURE_COLUMNS:
        rows = evaluate_feature_models(
            feature,
            train_df[feature],
            include_gmm=include_gmm,
            gmm_n_components=gmm_n_components,
            gmm_random_state=gmm_random_state,
        )
        if not rows:
            continue
        best = min(rows, key=lambda r: r["aic"])
        model_map[feature] = best["model"]
    return model_map


def _load_feature_df_cached(user_id, dataset_dir, preprocessed_dir, cache=None):
    uid = int(user_id)
    if cache is not None and uid in cache:
        return cache[uid]
    path = user_json_path(uid, dataset_dir=dataset_dir)
    frame = build_feature_df_for_user(path, preprocessed_dir=preprocessed_dir, logger=None)
    if cache is not None:
        cache[uid] = frame
    return frame


def summarize_train_vs_rest(result_df):
    """Summary stats over rest users only (excludes train_user == eval_user rows)."""
    empty = pd.DataFrame(
        [
            {
                "n_eval": 0,
                "mean_ll_diff": np.nan,
                "std_ll_diff": np.nan,
                "mean_abs_ll_diff": np.nan,
                "mean_risk": np.nan,
                "std_risk": np.nan,
            }
        ]
    )
    if result_df is None or result_df.empty:
        return empty

    rest_df = result_df
    if "train_user" in result_df.columns and "eval_user" in result_df.columns:
        rest_df = result_df[
            pd.to_numeric(result_df["eval_user"], errors="coerce")
            != pd.to_numeric(result_df["train_user"], errors="coerce")
        ]
    if rest_df.empty:
        return empty

    ll_diff = pd.to_numeric(rest_df["mean_ll_diff_vs_train"], errors="coerce")
    risk = pd.to_numeric(rest_df["risk_score"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "n_eval": int(len(rest_df)),
                "mean_ll_diff": float(ll_diff.mean(skipna=True)),
                "std_ll_diff": float(ll_diff.std(skipna=True)),
                "mean_abs_ll_diff": float(ll_diff.abs().mean(skipna=True)),
                "mean_risk": float(risk.mean(skipna=True)),
                "std_risk": float(risk.std(skipna=True)),
            }
        ]
    )


def load_selected_models(summary_csv_path, criterion_col="best_weighted_mean_aic"):
    """
    Load feature -> selected_model from aggregated summary CSV.
    """
    summary_df = pd.read_csv(summary_csv_path)
    required = {"feature", criterion_col}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")

    model_map = {}
    for _, row in summary_df.iterrows():
        feature = str(row["feature"])
        model_name = str(row[criterion_col])
        if feature in FEATURE_COLUMNS:
            model_map[feature] = model_name
    return model_map


def fit_feature_models_on_user(
    train_df,
    model_map,
    include_gmm=False,
    gmm_n_components=2,
    gmm_random_state=0,
):
    """
    Fit selected model per feature on training user's feature frame.
    Returns dict: feature -> fit_result from fitter.

    If ``model_map`` names ``GMM`` for any feature, GMM fitting is enabled even when
    ``include_gmm=False`` (legacy compare/API loading a GMM summary must not silently
    drop those features). Unknown model names raise ``ValueError``.
    """
    requested = {
        str(model_map[f])
        for f in FEATURE_COLUMNS
        if f in model_map and model_map[f] is not None and str(model_map[f]).strip()
    }
    needs_gmm = "GMM" in requested
    if needs_gmm and not include_gmm:
        warnings.warn(
            "model_map selects GMM; enabling GMM fitters automatically "
            "(pass include_gmm=True to silence this warning).",
            UserWarning,
            stacklevel=2,
        )
        include_gmm = True

    fitters = get_model_fitters(
        include_gmm=include_gmm,
        gmm_n_components=gmm_n_components,
        gmm_random_state=gmm_random_state,
    )
    unknown = sorted(requested - set(fitters))
    if unknown:
        raise ValueError(
            f"model_map references unknown model(s) {unknown}; "
            f"known={sorted(fitters)}"
        )

    fitted = {}
    for feature in FEATURE_COLUMNS:
        model_name = model_map.get(feature)
        if model_name is None or model_name not in fitters:
            continue
        values = pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
        result = fitters[model_name](values)
        if result is not None:
            fitted[feature] = {"model": model_name, **result}
    return fitted


def _logpdf_by_model(model_name, x, params):
    if model_name == "Gaussian":
        return stats.norm.logpdf(x, loc=params["mu"], scale=params["sigma"])
    if model_name == "Log-normal":
        return stats.lognorm.logpdf(x, params["shape"], loc=params["loc"], scale=params["scale"])
    if model_name == "Gamma":
        return stats.gamma.logpdf(x, params["shape"], loc=params["loc"], scale=params["scale"])
    if model_name == "Weibull":
        return stats.weibull_min.logpdf(x, params["shape"], loc=params["loc"], scale=params["scale"])
    if model_name == "Log-logistic":
        return stats.fisk.logpdf(x, params["shape"], loc=params["loc"], scale=params["scale"])
    if model_name == "GMM":
        weights = np.asarray(params["weights"], dtype=float)
        means = np.asarray(params["means"], dtype=float)
        stds = np.asarray(params["stds"], dtype=float)
        x = np.asarray(x, dtype=float)
        # log p(x) = logsumexp_k [ log w_k + log N(x; mu_k, sigma_k) ]
        comps = []
        for w, mu, sigma in zip(weights, means, stds):
            sigma = max(float(sigma), 1e-12)
            comps.append(np.log(max(float(w), 1e-300)) + stats.norm.logpdf(x, loc=mu, scale=sigma))
        return np.logaddexp.reduce(comps, axis=0)
    if model_name == "Student-t":
        return stats.t.logpdf(x, params["df"], loc=params["loc"], scale=params["scale"])
    raise ValueError(f"Unsupported model: {model_name}")


def score_user_against_fitted_models(eval_df, fitted_models):
    """
    Compute per-feature mean/total log-likelihood, mean NLL (-mean LL), and sample count.
    """
    rows = []
    for feature in FEATURE_COLUMNS:
        fit = fitted_models.get(feature)
        if fit is None:
            rows.append(
                {
                    "feature": feature,
                    "model": None,
                    "n_used": 0,
                    "mean_log_likelihood": np.nan,
                    "total_log_likelihood": np.nan,
                    "mean_nll": np.nan,
                    "total_nll": np.nan,
                }
            )
            continue

        model_name = fit["model"]
        params = fit["params"]
        x_all = pd.to_numeric(eval_df[feature], errors="coerce").to_numpy(dtype=float)
        x = _prepare_values_for_model(x_all, model_name)

        if len(x) == 0:
            rows.append(
                {
                    "feature": feature,
                    "model": model_name,
                    "n_used": 0,
                    "mean_log_likelihood": np.nan,
                    "total_log_likelihood": np.nan,
                    "mean_nll": np.nan,
                    "total_nll": np.nan,
                }
            )
            continue

        logpdf = _logpdf_by_model(model_name, x, params)
        logpdf = logpdf[np.isfinite(logpdf)]
        if len(logpdf) == 0:
            mean_ll = np.nan
            total_ll = np.nan
            mean_nll = np.nan
            total_nll = np.nan
            n_used = 0
        else:
            mean_ll = float(np.mean(logpdf))
            total_ll = float(np.sum(logpdf))
            mean_nll = float(-mean_ll)
            total_nll = float(-total_ll)
            n_used = int(len(logpdf))

        rows.append(
            {
                "feature": feature,
                "model": model_name,
                "n_used": n_used,
                "mean_log_likelihood": mean_ll,
                "total_log_likelihood": total_ll,
                "mean_nll": mean_nll,
                "total_nll": total_nll,
            }
        )
    return pd.DataFrame(rows)


def _default_output_csv_path(train_user_id, eval_user_ids):
    evals = sorted({int(u) for u in eval_user_ids})
    if len(evals) <= 10:
        eval_tag = "_".join(f"{u:04d}" for u in evals)
    else:
        eval_tag = f"{evals[0]:04d}_to_{evals[-1]:04d}_n{len(evals)}"
    return (
        f"results/compare/tables/loss_compare_train_{int(train_user_id):04d}_eval_{eval_tag}.csv"
    )


def _default_validate_output_csv_path():
    return "results/compare/tables/validate_risk_vs_logs.csv"


def _list_input_files(input_dir, pattern):
    search = os.path.join(input_dir, pattern)
    return sorted(glob.glob(search))


def _build_merged_feature_df(file_paths, preprocessed_dir):
    frames = []
    for src_path in file_paths:
        frame = build_feature_df_for_user(
            src_path,
            preprocessed_dir=preprocessed_dir,
            logger=None,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def compare_validate_against_logs(
    logs_dir="./logs",
    logs_pattern="*.json",
    validate_source_dir="./validate/source",
    validate_source_pattern="*.json",
    validate_imposter_dir="./validate/imposter",
    validate_imposter_pattern="*.json",
    logs_preprocessed_dir="results/preprocessed_logs",
    validate_source_preprocessed_dir="results/preprocessed_validate/source",
    validate_imposter_preprocessed_dir="results/preprocessed_validate/imposter",
    summary_csv_path="results/main/tables/model_fit_aggregated_summary.csv",
    criterion_col="best_weighted_mean_aic",
    output_csv_path=None,
):
    """
    Fit selected models on all logs/* files, then score validate source/imposter files.

    Risk score definition:
    - ll_diff_vs_logs = mean_log_likelihood - logs_baseline_mean_ll
    - risk_score = max(0, logs_baseline_mean_ll - mean_log_likelihood)
    """
    model_map = load_selected_models(summary_csv_path, criterion_col=criterion_col)

    log_files = _list_input_files(logs_dir, logs_pattern)
    if not log_files:
        raise FileNotFoundError(f"No logs files found: {os.path.join(logs_dir, logs_pattern)}")
    logs_df = _build_merged_feature_df(log_files, preprocessed_dir=logs_preprocessed_dir)
    if logs_df.empty:
        raise RuntimeError("No usable feature rows from logs files.")

    fitted_models = fit_feature_models_on_user(logs_df, model_map)
    logs_score_df = score_user_against_fitted_models(logs_df, fitted_models)
    logs_baseline_mean_ll = float(logs_score_df["mean_log_likelihood"].mean(skipna=True))

    source_files = _list_input_files(validate_source_dir, validate_source_pattern)
    imposter_files = _list_input_files(validate_imposter_dir, validate_imposter_pattern)
    if not source_files and not imposter_files:
        raise FileNotFoundError(
            "No validate files found in either source or imposter: "
            f"{os.path.join(validate_source_dir, validate_source_pattern)}, "
            f"{os.path.join(validate_imposter_dir, validate_imposter_pattern)}"
        )

    rows = [
        {
            "source": "logs_baseline",
            "validate_group": "baseline",
            "file_name": "__logs_baseline__",
            "n_features_used": int(logs_score_df["mean_log_likelihood"].notna().sum()),
            "mean_log_likelihood": logs_baseline_mean_ll,
            "logs_baseline_mean_ll": logs_baseline_mean_ll,
            "ll_diff_vs_logs": 0.0,
            "risk_score": 0.0,
            "mean_nll": float(logs_score_df["mean_nll"].mean(skipna=True)),
        }
    ]

    validate_specs = [
        ("source", source_files, validate_source_preprocessed_dir),
        ("imposter", imposter_files, validate_imposter_preprocessed_dir),
    ]
    for group_name, paths, pp_dir in validate_specs:
        for src_path in paths:
            eval_df = build_feature_df_for_user(
                src_path,
                preprocessed_dir=pp_dir,
                logger=None,
            )
            if eval_df.empty:
                rows.append(
                    {
                        "source": "validate",
                        "validate_group": group_name,
                        "file_name": os.path.basename(src_path),
                        "n_features_used": 0,
                        "mean_log_likelihood": np.nan,
                        "logs_baseline_mean_ll": logs_baseline_mean_ll,
                        "ll_diff_vs_logs": np.nan,
                        "risk_score": np.nan,
                        "mean_nll": np.nan,
                    }
                )
                continue

            feature_score_df = score_user_against_fitted_models(eval_df, fitted_models)
            mean_ll = float(feature_score_df["mean_log_likelihood"].mean(skipna=True))
            ll_diff = float(mean_ll - logs_baseline_mean_ll)
            rows.append(
                {
                    "source": "validate",
                    "validate_group": group_name,
                    "file_name": os.path.basename(src_path),
                    "n_features_used": int(feature_score_df["mean_log_likelihood"].notna().sum()),
                    "mean_log_likelihood": mean_ll,
                    "logs_baseline_mean_ll": logs_baseline_mean_ll,
                    "ll_diff_vs_logs": ll_diff,
                    "risk_score": float(max(0.0, -ll_diff)),
                    "mean_nll": float(feature_score_df["mean_nll"].mean(skipna=True)),
                }
            )

    result_df = pd.DataFrame(rows)
    baseline_df = result_df[result_df["source"] == "logs_baseline"]
    validate_df = result_df[result_df["source"] == "validate"].sort_values(
        ["validate_group", "risk_score"], ascending=[True, False], na_position="last"
    )
    result_df = pd.concat([baseline_df, validate_df], ignore_index=True)

    if output_csv_path is None:
        output_csv_path = _default_validate_output_csv_path()
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    return result_df, output_csv_path


def compare_users(
    train_user_id=1,
    eval_user_ids=(1, 2),
    dataset_dir="./raw_kmt_dataset",
    preprocessed_dir="results/preprocessed_kmt",
    summary_csv_path="results/main/tables/model_fit_aggregated_summary.csv",
    criterion_col="best_weighted_mean_aic",
    output_csv_path=None,
    skip_missing_eval_files=True,
):
    """
    Fit selected models on one train user, then compute per-user mean log-likelihood.

    Output has one row per eval user (not per feature). The train user row is always
    included as the first row and acts as the in-sample baseline.
    """
    model_map = load_selected_models(summary_csv_path, criterion_col=criterion_col)

    train_path = user_json_path(train_user_id, dataset_dir=dataset_dir)
    if not os.path.isfile(train_path) and resolve_preprocessed_json_path(
        train_path, preprocessed_dir
    ) is None:
        raise FileNotFoundError(f"Train user raw or preprocessed file not found: {train_path}")
    train_df = build_feature_df_for_user(
        train_path, preprocessed_dir=preprocessed_dir, logger=None
    )
    fitted_models = fit_feature_models_on_user(train_df, model_map)

    train_score_df = score_user_against_fitted_models(train_df, fitted_models)
    train_baseline_mean_ll = float(train_score_df["mean_log_likelihood"].mean(skipna=True))

    all_eval_ids = [int(train_user_id)]
    all_eval_ids.extend([int(u) for u in eval_user_ids if int(u) != int(train_user_id)])

    user_rows = []
    for eval_user_id in all_eval_ids:
        eval_path = user_json_path(eval_user_id, dataset_dir=dataset_dir)
        eval_pp = resolve_preprocessed_json_path(eval_path, preprocessed_dir)
        if skip_missing_eval_files and not os.path.isfile(eval_path) and not eval_pp:
            print(f"[skip] missing eval raw and preprocessed: {eval_path}")
            continue
        eval_df = build_feature_df_for_user(
            eval_path, preprocessed_dir=preprocessed_dir, logger=None
        )
        feature_score_df = score_user_against_fitted_models(eval_df, fitted_models)
        mean_ll = float(feature_score_df["mean_log_likelihood"].mean(skipna=True))
        mean_nll = float(feature_score_df["mean_nll"].mean(skipna=True))
        n_features_used = int(feature_score_df["mean_log_likelihood"].notna().sum())
        user_rows.append(
            {
                "train_user": int(train_user_id),
                "eval_user": int(eval_user_id),
                "n_features_used": n_features_used,
                "mean_log_likelihood": mean_ll,
                "mean_ll_train_baseline": train_baseline_mean_ll,
                "mean_ll_diff_vs_train": float(mean_ll - train_baseline_mean_ll),
                "mean_nll": mean_nll,
            }
        )

    if not user_rows:
        raise RuntimeError("No eval users produced scores (missing files or empty eval list).")

    result_df = pd.DataFrame(user_rows)
    result_df = result_df.sort_values(
        by=["eval_user"],
        key=lambda s: s.where(s != int(train_user_id), -1),
    )

    if output_csv_path is None:
        output_csv_path = _default_output_csv_path(train_user_id, eval_user_ids)
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    return result_df, output_csv_path


def _default_train_vs_rest_paths(train_user_id):
    detail = f"results/compare/tables/train_vs_rest_user_{int(train_user_id):04d}.csv"
    summary = f"results/compare/tables/train_vs_rest_user_{int(train_user_id):04d}_summary.csv"
    return detail, summary


def _default_all_vs_rest_paths():
    per_train = "results/compare/tables/all_vs_rest_per_train.csv"
    grand = "results/compare/tables/all_vs_rest_grand_summary.csv"
    return per_train, grand


def compare_train_vs_rest(
    train_user_id=1,
    dataset_dir="./raw_kmt_dataset",
    preprocessed_dir="results/preprocessed_kmt",
    output_csv_path=None,
    feature_df_cache=None,
    write_outputs=True,
):
    """
    Fit AIC-min models on --train-user, score train user + all other discovered users.

    Metrics per eval user (including train self-row):
    - mean_ll_diff_vs_train = mean_ll(eval) - mean_ll(train baseline)
    - risk_score = max(0, -mean_ll_diff_vs_train)

    Summary stats exclude the self-row (train_user == eval_user).
    """
    user_ids = discover_user_ids(dataset_dir)
    if not user_ids:
        raise FileNotFoundError(
            f"No raw_kmt_user_*.json files found under: {dataset_dir}"
        )
    train_uid = int(train_user_id)
    if train_uid not in user_ids:
        raise FileNotFoundError(
            f"Train user {train_uid:04d} not found in {dataset_dir}"
        )

    train_df = _load_feature_df_cached(
        train_uid, dataset_dir, preprocessed_dir, cache=feature_df_cache
    )
    if train_df.empty:
        raise RuntimeError(f"No usable feature rows for train user {train_uid:04d}.")

    model_map = select_models_by_aic_on_user(train_df)
    if not model_map:
        raise RuntimeError(f"No models could be selected by AIC for train user {train_uid:04d}.")
    fitted_models = fit_feature_models_on_user(train_df, model_map)
    train_score_df = score_user_against_fitted_models(train_df, fitted_models)
    train_baseline_mean_ll = float(train_score_df["mean_log_likelihood"].mean(skipna=True))

    # Include train user first (self baseline), then the rest.
    eval_ids = [train_uid] + [uid for uid in user_ids if uid != train_uid]
    user_rows = []
    for eval_user_id in eval_ids:
        eval_df = _load_feature_df_cached(
            eval_user_id, dataset_dir, preprocessed_dir, cache=feature_df_cache
        )
        feature_score_df = score_user_against_fitted_models(eval_df, fitted_models)
        mean_ll = float(feature_score_df["mean_log_likelihood"].mean(skipna=True))
        mean_nll = float(feature_score_df["mean_nll"].mean(skipna=True))
        n_features_used = int(feature_score_df["mean_log_likelihood"].notna().sum())
        ll_diff = float(mean_ll - train_baseline_mean_ll)
        user_rows.append(
            {
                "train_user": train_uid,
                "eval_user": int(eval_user_id),
                "n_features_used": n_features_used,
                "mean_log_likelihood": mean_ll,
                "mean_ll_train_baseline": train_baseline_mean_ll,
                "mean_ll_diff_vs_train": ll_diff,
                "risk_score": float(max(0.0, -ll_diff)),
                "mean_nll": mean_nll,
            }
        )

    if not user_rows:
        raise RuntimeError("No eval users found to score against train user.")

    result_df = pd.DataFrame(user_rows)
    result_df = result_df.sort_values(
        by=["eval_user"],
        key=lambda s: s.where(s != train_uid, -1),
    ).reset_index(drop=True)
    summary_df = summarize_train_vs_rest(result_df)

    detail_path = output_csv_path
    summary_path = None
    if write_outputs:
        if detail_path is None:
            detail_path, summary_path = _default_train_vs_rest_paths(train_uid)
        else:
            stem, ext = os.path.splitext(detail_path)
            summary_path = f"{stem}_summary{ext or '.csv'}"
        for path in (detail_path, summary_path):
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        result_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

    return result_df, summary_df, detail_path, summary_path


def compare_all_vs_rest(
    dataset_dir="./raw_kmt_dataset",
    preprocessed_dir="results/preprocessed_kmt",
    output_csv_path=None,
):
    """
    For each discovered user n, run train_vs_rest with train_user=n.
    Feature DataFrames are loaded once and reused via cache.
    """
    user_ids = discover_user_ids(dataset_dir)
    if not user_ids:
        raise FileNotFoundError(
            f"No raw_kmt_user_*.json files found under: {dataset_dir}"
        )

    feature_df_cache = {}
    for uid in user_ids:
        _load_feature_df_cached(uid, dataset_dir, preprocessed_dir, cache=feature_df_cache)

    per_train_rows = []
    for i, train_uid in enumerate(user_ids, start=1):
        print(f"[all_vs_rest] train_user={train_uid:04d} ({i}/{len(user_ids)})")
        _, summary_df, _, _ = compare_train_vs_rest(
            train_user_id=train_uid,
            dataset_dir=dataset_dir,
            preprocessed_dir=preprocessed_dir,
            feature_df_cache=feature_df_cache,
            write_outputs=False,
        )
        row = {"train_user": int(train_uid)}
        row.update(summary_df.iloc[0].to_dict())
        per_train_rows.append(row)

    per_train_df = pd.DataFrame(per_train_rows).sort_values(by=["train_user"]).reset_index(drop=True)

    metric_cols = [
        "mean_ll_diff",
        "std_ll_diff",
        "mean_abs_ll_diff",
        "mean_risk",
        "std_risk",
    ]
    grand_row = {"n_train_users": int(len(per_train_df))}
    for col in metric_cols:
        series = pd.to_numeric(per_train_df[col], errors="coerce")
        grand_row[f"{col}_mean"] = float(series.mean(skipna=True))
        grand_row[f"{col}_std"] = float(series.std(skipna=True))
    grand_summary_df = pd.DataFrame([grand_row])

    if output_csv_path is None:
        per_train_path, grand_path = _default_all_vs_rest_paths()
    else:
        per_train_path = output_csv_path
        stem, ext = os.path.splitext(per_train_path)
        grand_path = f"{stem}_grand_summary{ext or '.csv'}"

    for path in (per_train_path, grand_path):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    per_train_df.to_csv(per_train_path, index=False)
    grand_summary_df.to_csv(grand_path, index=False)
    return per_train_df, grand_summary_df, per_train_path, grand_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fitted distributions by user or validate risk mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="validate_risk",
        choices=[
            "validate_risk",
            "user_compare",
            "train_vs_rest",
            "all_vs_rest",
            "authentication_eval",
        ],
        help=(
            "authentication_eval: leakage-free paper evaluation; "
            "validate_risk / user_compare / train_vs_rest / all_vs_rest: legacy/exploratory."
        ),
    )
    parser.add_argument("--train-user", type=int, default=1, help="User id used for fitting.")
    parser.add_argument(
        "--eval-users",
        type=int,
        nargs="+",
        default=None,
        help="One or more user ids to evaluate. Ignored if --eval-user-range is set.",
    )
    parser.add_argument(
        "--eval-user-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Inclusive user id range (e.g. 2 88). Overrides --eval-users when set.",
    )
    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        default=None,
        help="(authentication_eval) subset of enrolled/eval user ids.",
    )
    parser.add_argument(
        "--user-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="(authentication_eval) inclusive user id range for smoke/full runs.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./raw_kmt_dataset",
        help="Directory containing raw_kmt_user_*.json files.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="results/preprocessed_kmt",
        help="Directory with preprocessed_kmt_user_*.json (from preprocess.py). Empty string disables.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/main/tables/model_fit_aggregated_summary.csv",
        help="Aggregated summary CSV path for selected model column.",
    )
    parser.add_argument(
        "--criterion-col",
        type=str,
        default="best_weighted_mean_aic",
        choices=[
            "best_majority_vote_aic",
            "best_majority_vote_bic",
            "best_weighted_mean_aic",
            "best_weighted_mean_bic",
            "best_sum_log_likelihood",
        ],
        help="Column in summary CSV used to choose model per feature.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output CSV path. If omitted, mode-specific default under results/compare/tables is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="(authentication_eval) output directory for metrics and assignments.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6, help="(authentication_eval)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="(authentication_eval)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="(authentication_eval)")
    parser.add_argument("--split-seed", type=int, default=42, help="(authentication_eval)")
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="validation_eer",
        choices=["genuine_quantile", "validation_eer"],
        help="(authentication_eval) threshold calibration mode (default: validation_eer).",
    )
    parser.add_argument(
        "--genuine-quantile",
        type=float,
        default=0.05,
        help="(authentication_eval) quantile for genuine_quantile threshold.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=["all", "dwell", "flight", "velocity", "mouse", "keyboard"],
        help="(authentication_eval) feature ablation group. mouse=velocity; keyboard=dwell+flight.",
    )
    parser.add_argument(
        "--distribution-selection",
        type=str,
        default="local_aic",
        choices=["local_aic", "global_weighted_aic"],
        help="(authentication_eval) distribution family selection policy.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="(authentication_eval) sliding window length in seconds.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=1.0,
        help="(authentication_eval) sliding window stride in seconds.",
    )
    parser.add_argument(
        "--include-gmm",
        action="store_true",
        help="(authentication_eval) opt-in: include univariate GMM in AIC candidates.",
    )
    parser.add_argument(
        "--gmm-n-components",
        type=int,
        default=2,
        help="(authentication_eval) GMM components when --include-gmm is set (default: 2).",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="./logs",
        help="(validate_risk) reference logs directory.",
    )
    parser.add_argument(
        "--logs-pattern",
        type=str,
        default="*.json",
        help="(validate_risk) glob pattern for logs-dir.",
    )
    parser.add_argument(
        "--validate-source-dir",
        type=str,
        default="./validate/source",
        help="(validate_risk) genuine-source sessions directory.",
    )
    parser.add_argument(
        "--validate-source-pattern",
        type=str,
        default="*.json",
        help="(validate_risk) glob pattern for validate-source-dir.",
    )
    parser.add_argument(
        "--validate-imposter-dir",
        type=str,
        default="./validate/imposter",
        help="(validate_risk) imposter sessions directory.",
    )
    parser.add_argument(
        "--validate-imposter-pattern",
        type=str,
        default="*.json",
        help="(validate_risk) glob pattern for validate-imposter-dir.",
    )
    parser.add_argument(
        "--logs-preprocessed-dir",
        type=str,
        default="results/preprocessed_logs",
        help="(validate_risk) preprocessed directory for logs files.",
    )
    parser.add_argument(
        "--validate-source-preprocessed-dir",
        type=str,
        default="results/preprocessed_validate/source",
        help="(validate_risk) preprocessed directory for validate source files.",
    )
    parser.add_argument(
        "--validate-imposter-preprocessed-dir",
        type=str,
        default="results/preprocessed_validate/imposter",
        help="(validate_risk) preprocessed directory for validate imposter files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_csv if args.output_csv else None
    pp_dir = args.preprocessed_dir if args.preprocessed_dir.strip() else ""

    if args.mode == "authentication_eval":
        from authentication_eval import run_authentication_eval

        if args.user_range is not None:
            lo, hi = args.user_range
            user_ids = list(range(int(lo), int(hi) + 1))
        elif args.users is not None:
            user_ids = list(args.users)
        else:
            user_ids = None

        result = run_authentication_eval(
            dataset_dir=args.dataset_dir,
            preprocessed_dir=pp_dir,
            output_dir=args.output_dir,
            user_ids=user_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
            threshold_mode=args.threshold_mode,
            genuine_quantile=args.genuine_quantile,
            feature_set=args.feature_set,
            distribution_selection=args.distribution_selection,
            window_size=args.window_size,
            stride=args.stride,
            include_gmm=bool(args.include_gmm),
            gmm_n_components=int(args.gmm_n_components),
        )
        print("authentication_eval summary:")
        print(result["summary"].to_string(index=False))
        print("\nOutputs:")
        for name, path in result["paths"].items():
            print(f"  {name}: {path}")
        return

    if args.mode == "validate_risk":
        logs_pp = args.logs_preprocessed_dir if args.logs_preprocessed_dir.strip() else ""
        source_pp = args.validate_source_preprocessed_dir if args.validate_source_preprocessed_dir.strip() else ""
        imposter_pp = args.validate_imposter_preprocessed_dir if args.validate_imposter_preprocessed_dir.strip() else ""
        df, saved_path = compare_validate_against_logs(
            logs_dir=args.logs_dir,
            logs_pattern=args.logs_pattern,
            validate_source_dir=args.validate_source_dir,
            validate_source_pattern=args.validate_source_pattern,
            validate_imposter_dir=args.validate_imposter_dir,
            validate_imposter_pattern=args.validate_imposter_pattern,
            logs_preprocessed_dir=logs_pp,
            validate_source_preprocessed_dir=source_pp,
            validate_imposter_preprocessed_dir=imposter_pp,
            summary_csv_path=args.summary_csv,
            criterion_col=args.criterion_col,
            output_csv_path=output_path,
        )
        print(df.to_string(index=False))
        print(f"\nSaved: {saved_path}")
        return

    if args.mode == "train_vs_rest":
        result_df, summary_df, detail_path, summary_path = compare_train_vs_rest(
            train_user_id=args.train_user,
            dataset_dir=args.dataset_dir,
            preprocessed_dir=pp_dir,
            output_csv_path=output_path,
        )
        # Legacy alias for paper docs: risk_score == legacy_risk
        if "risk_score" in result_df.columns and "legacy_risk" not in result_df.columns:
            result_df = result_df.copy()
            result_df["legacy_risk"] = result_df["risk_score"]
            result_df.to_csv(detail_path, index=False)
        print(summary_df.to_string(index=False))
        print(f"\nSaved detail: {detail_path}")
        print(f"Saved summary: {summary_path}")
        return

    if args.mode == "all_vs_rest":
        per_train_df, grand_df, per_train_path, grand_path = compare_all_vs_rest(
            dataset_dir=args.dataset_dir,
            preprocessed_dir=pp_dir,
            output_csv_path=output_path,
        )
        print("Grand summary:")
        print(grand_df.to_string(index=False))
        print("\nPer-train (brief):")
        brief_cols = [
            c
            for c in ("train_user", "n_eval", "mean_ll_diff", "mean_abs_ll_diff", "mean_risk")
            if c in per_train_df.columns
        ]
        print(per_train_df[brief_cols].to_string(index=False))
        print(f"\nSaved per-train: {per_train_path}")
        print(f"Saved grand summary: {grand_path}")
        return

    # user_compare (legacy)
    if args.eval_user_range is not None:
        lo, hi = args.eval_user_range
        eval_ids = tuple(range(int(lo), int(hi) + 1))
    elif args.eval_users is not None:
        eval_ids = tuple(args.eval_users)
    else:
        eval_ids = (1, 2)

    df, saved_path = compare_users(
        train_user_id=args.train_user,
        eval_user_ids=eval_ids,
        dataset_dir=args.dataset_dir,
        preprocessed_dir=pp_dir,
        summary_csv_path=args.summary_csv,
        criterion_col=args.criterion_col,
        output_csv_path=output_path,
    )
    print(df.to_string(index=False))
    print(f"\nSaved: {saved_path}")


if __name__ == "__main__":
    main()
