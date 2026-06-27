import argparse
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
        description=(
            "Fit selected models on one user and score target users: "
            "log-likelihood, optional NLL, and mean_ll_diff_vs_train vs in-sample train LL."
        )
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
        help="Optional output CSV path. If omitted, auto path under results/compare/tables is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_user_range is not None:
        lo, hi = args.eval_user_range
        eval_ids = tuple(range(int(lo), int(hi) + 1))
    elif args.eval_users is not None:
        eval_ids = tuple(args.eval_users)
    else:
        eval_ids = (1, 2)

    output_path = args.output_csv if args.output_csv else None
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
