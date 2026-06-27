import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from main import (
    FEATURE_COLUMNS,
    _fit_gamma,
    _fit_gaussian,
    _fit_lognormal,
    _fit_student_t,
    _fit_weibull,
    _prepare_values_for_model,
    build_feature_df_for_user,
    resolve_preprocessed_json_path,
)


MODEL_FITTERS = {
    "Gaussian": _fit_gaussian,
    "Log-normal": _fit_lognormal,
    "Gamma": _fit_gamma,
    "Weibull": _fit_weibull,
    "Student-t": _fit_student_t,
}


def user_json_path(user_id, dataset_dir="./raw_kmt_dataset"):
    return os.path.join(dataset_dir, f"raw_kmt_user_{int(user_id):04d}.json")


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


def fit_feature_models_on_user(train_df, model_map):
    """
    Fit selected model per feature on training user's feature frame.
    Returns dict: feature -> fit_result from fitter.
    """
    fitted = {}
    for feature in FEATURE_COLUMNS:
        model_name = model_map.get(feature)
        if model_name not in MODEL_FITTERS:
            continue
        values = pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
        result = MODEL_FITTERS[model_name](values)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fitted distributions by user or validate risk mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="validate_risk",
        choices=["validate_risk", "user_compare"],
        help="validate_risk: logs->validate risk scoring, user_compare: legacy user-id compare.",
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
    else:
        if args.eval_user_range is not None:
            lo, hi = args.eval_user_range
            eval_ids = tuple(range(int(lo), int(hi) + 1))
        elif args.eval_users is not None:
            eval_ids = tuple(args.eval_users)
        else:
            eval_ids = (1, 2)

        pp_dir = args.preprocessed_dir if args.preprocessed_dir.strip() else ""
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
