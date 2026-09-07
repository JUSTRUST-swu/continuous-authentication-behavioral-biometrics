"""
Leakage-free authentication evaluation (paper primary path).

Split at session/segment (not overlapping windows), fit clip+models on train only,
calibrate threshold on validation genuines, score held-out test genuines vs impostors.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from auth_metrics import (
    calibrate_genuine_quantile_threshold,
    compute_binary_metrics,
    compute_eer,
    roc_auc_score,
)
from evaluation_split import (
    assign_time_block_labels,
    build_split_assignments,
)
from feature_transform import (
    FeatureTransformParams,
    fit_transform_params,
    transform_features,
    transform_params_to_rows,
)
from main import (
    FEATURE_COLUMNS,
    _prepare_values_for_model,
    build_feature_df_for_user,
    evaluate_feature_models,
)


FEATURE_SETS = {
    "all": list(FEATURE_COLUMNS),
    "dwell": ["dwell_mean", "dwell_std"],
    "flight": ["flight_mean", "flight_std"],
    "velocity": ["velocity_mean", "velocity_std"],
}


@dataclass
class ScoreResult:
    mean_log_likelihood: float
    n_features_used: int
    per_feature_ll: Dict[str, float] = field(default_factory=dict)
    per_feature_n: Dict[str, int] = field(default_factory=dict)

    @property
    def mean_nll(self) -> float:
        if not np.isfinite(self.mean_log_likelihood):
            return float("nan")
        return float(-self.mean_log_likelihood)


def resolve_feature_columns(feature_set: str) -> List[str]:
    key = str(feature_set).strip().lower()
    if key not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[key])


def score_user(
    feature_df: pd.DataFrame,
    fitted_models: dict,
    transform_params: Optional[FeatureTransformParams] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> ScoreResult:
    """
    Unified scoring: optional transform, then mean LL over available features.
    Does not use train-LL baseline.
    """
    # Local import avoids circular dependency with loss_compare.
    from loss_compare import _logpdf_by_model

    cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    df = feature_df
    if transform_params is not None:
        df = transform_features(df, transform_params, feature_columns=cols)

    per_ll: Dict[str, float] = {}
    per_n: Dict[str, int] = {}
    for feature in cols:
        fit = fitted_models.get(feature)
        if fit is None or feature not in df.columns:
            continue
        model_name = fit["model"]
        params = fit["params"]
        x_all = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        x = _prepare_values_for_model(x_all, model_name)
        if len(x) == 0:
            continue
        logpdf = _logpdf_by_model(model_name, x, params)
        logpdf = logpdf[np.isfinite(logpdf)]
        if len(logpdf) == 0:
            continue
        per_ll[feature] = float(np.mean(logpdf))
        per_n[feature] = int(len(logpdf))

    if not per_ll:
        return ScoreResult(
            mean_log_likelihood=float("nan"),
            n_features_used=0,
            per_feature_ll={},
            per_feature_n={},
        )
    return ScoreResult(
        mean_log_likelihood=float(np.mean(list(per_ll.values()))),
        n_features_used=int(len(per_ll)),
        per_feature_ll=per_ll,
        per_feature_n=per_n,
    )


def _group_key_column(split_unit: str) -> str:
    if split_unit == "session":
        return "session_id"
    if split_unit == "segment":
        return "segment_id"
    if split_unit == "time_block":
        return "group_id"
    return "session_id"


def _load_raw_feature_frame(
    user_id: int,
    dataset_dir: str,
    preprocessed_dir: str,
    cache: Optional[dict],
    window_size: float = 5.0,
    stride: float = 1.0,
) -> pd.DataFrame:
    from loss_compare import user_json_path

    uid = int(user_id)
    if cache is not None and uid in cache:
        return cache[uid]
    path = user_json_path(uid, dataset_dir=dataset_dir)
    # Prefer session-aware raw/schema-v2 path; no transform (fit later on train)
    frame = build_feature_df_for_user(
        path,
        preprocessed_dir=preprocessed_dir,
        logger=None,
        window_size=window_size,
        stride=stride,
        apply_transform=False,
        prefer_sessions=True,
    )
    if cache is not None:
        cache[uid] = frame
    return frame


def _units_from_frame(df: pd.DataFrame) -> Dict[str, List[str]]:
    sessions = []
    segments = []
    if "session_id" in df.columns:
        sessions = sorted({str(x) for x in df["session_id"].dropna().unique()})
    if "segment_id" in df.columns:
        segments = sorted({str(x) for x in df["segment_id"].dropna().unique()})
    return {"session_ids": sessions, "segment_ids": segments}


def _apply_split_labels(df: pd.DataFrame, assignment) -> pd.DataFrame:
    out = df.copy()
    out["split"] = None
    out["group_id"] = None

    if assignment.split_unit == "time_block":
        ordered = out.sort_values(["window_start", "window_end"]).reset_index(drop=True)
        labels = assign_time_block_labels(len(ordered))
        ordered["group_id"] = labels
        ordered["split"] = ordered["group_id"].map(
            {
                "time_block_train": "train",
                "time_block_validation": "validation",
                "time_block_test": "test",
            }
        )
        return ordered

    key_col = _group_key_column(assignment.split_unit)
    if key_col not in out.columns:
        raise ValueError(f"Missing {key_col} for split_unit={assignment.split_unit}")

    mapping = {}
    for split_name, groups in (
        ("train", assignment.train),
        ("validation", assignment.validation),
        ("test", assignment.test),
    ):
        for gid in groups:
            mapping[str(gid)] = split_name
    out["group_id"] = out[key_col].astype(str)
    out["split"] = out["group_id"].map(mapping)
    return out


def _attempt_ids(df: pd.DataFrame, split_unit: str) -> List[str]:
    if df.empty:
        return []
    col = _group_key_column(split_unit)
    if col == "group_id" and "group_id" in df.columns:
        return sorted({str(x) for x in df["group_id"].dropna().unique()})
    if col in df.columns:
        return sorted({str(x) for x in df[col].dropna().unique()})
    return ["__all__"]


def _slice_attempt(df: pd.DataFrame, split_unit: str, attempt_id: str) -> pd.DataFrame:
    col = _group_key_column(split_unit)
    if attempt_id == "__all__":
        return df
    if col not in df.columns and "group_id" in df.columns:
        col = "group_id"
    return df[df[col].astype(str) == str(attempt_id)]


def _fit_models_for_enrollment(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    distribution_selection: str,
    global_model_map: Optional[Dict[str, str]] = None,
    include_gmm: bool = True,
    gmm_n_components: int = 2,
    gmm_random_state: int = 0,
) -> Tuple[dict, Dict[str, str], List[dict]]:
    """Return fitted_models, model_map, fitted_models_rows."""
    from loss_compare import (
        get_model_fitters,
        resolve_include_gmm_for_model_map,
        select_models_by_aic_on_user,
    )

    mode = str(distribution_selection).strip().lower()
    if mode == "local_aic":
        model_map = select_models_by_aic_on_user(
            train_df,
            include_gmm=include_gmm,
            gmm_n_components=gmm_n_components,
            gmm_random_state=gmm_random_state,
        )
        model_map = {f: m for f, m in model_map.items() if f in feature_columns}
    elif mode == "global_weighted_aic":
        if not global_model_map:
            raise ValueError("global_weighted_aic requires a global_model_map from train partitions")
        model_map = {f: global_model_map[f] for f in feature_columns if f in global_model_map}
    else:
        raise ValueError(f"Unknown distribution_selection={distribution_selection!r}")

    include_gmm, requested = resolve_include_gmm_for_model_map(
        model_map.values(), include_gmm=include_gmm
    )

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
    rows = []
    for feature in feature_columns:
        model_name = model_map.get(feature)
        if model_name is None or model_name not in fitters:
            continue
        eval_rows = evaluate_feature_models(
            feature,
            train_df[feature],
            include_gmm=include_gmm,
            gmm_n_components=gmm_n_components,
            gmm_random_state=gmm_random_state,
        )
        chosen = next((r for r in eval_rows if r.get("model") == model_name), None)
        result = fitters[model_name](
            pd.to_numeric(train_df[feature], errors="coerce").to_numpy(dtype=float)
        )
        if result is None:
            continue
        fitted[feature] = {"model": model_name, **result}
        params = result.get("params") or {}
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple, dict)):
                flat_params[f"param_{k}"] = json.dumps(v)
            else:
                flat_params[f"param_{k}"] = v
        rows.append(
            {
                "feature": feature,
                "distribution": model_name,
                "aic": None if chosen is None else chosen.get("aic"),
                "bic": None if chosen is None else chosen.get("bic"),
                "n_train": int(result.get("n_used", 0)),
                **flat_params,
            }
        )
    return fitted, model_map, rows


def _build_global_model_map_from_train(
    labeled_frames: Dict[int, pd.DataFrame],
    feature_columns: Sequence[str],
    include_gmm: bool = True,
    gmm_n_components: int = 2,
    gmm_random_state: int = 0,
    weight_by_n_used: bool = False,
) -> Dict[str, str]:
    """
    Cohort mean AIC over users' train partitions only (no test leakage).

    Default: equal weight per user with a valid fit.
    ``weight_by_n_used=True``: weight each user's AIC by ``n_used``.

    For each user, fit clip+log1p on that user's train, transform train, then
    evaluate candidate distributions on the transformed scale (same scale as
    enrollment parameter fitting).

    Note: the chosen distribution *family* is shared across the cohort (each
    enrollee's train contributes to the vote). That is intentional population
    pooling for the ablation — not test-set leakage. Per-enrollee parameters
    are still estimated only on that enrollee's train after selection.
    """
    from collections import defaultdict

    from loss_compare import get_model_fitters

    fitters = get_model_fitters(
        include_gmm=include_gmm,
        gmm_n_components=gmm_n_components,
        gmm_random_state=gmm_random_state,
    )

    sum_w = defaultdict(float)
    sum_waic = defaultdict(float)
    for uid, df in labeled_frames.items():
        train_df = df[df["split"] == "train"]
        if train_df.empty:
            continue
        params = fit_transform_params(train_df, feature_columns)
        train_tx = transform_features(train_df, params, feature_columns=feature_columns)
        for feature in feature_columns:
            if feature not in train_tx.columns:
                continue
            for row in evaluate_feature_models(
                feature,
                train_tx[feature],
                include_gmm=include_gmm,
                gmm_n_components=gmm_n_components,
                gmm_random_state=gmm_random_state,
            ):
                key = (feature, row["model"])
                n_used = float(row.get("n_used") or 0)
                if n_used <= 0 or not np.isfinite(row.get("aic", np.nan)):
                    continue
                w = n_used if weight_by_n_used else 1.0
                sum_w[key] += w
                sum_waic[key] += w * float(row["aic"])

    model_map = {}
    for feature in feature_columns:
        best_model = None
        best_val = float("inf")
        for model_name in fitters:
            key = (feature, model_name)
            if sum_w[key] <= 0:
                continue
            waic = sum_waic[key] / sum_w[key]
            if waic < best_val:
                best_val = waic
                best_model = model_name
        if best_model is not None:
            model_map[feature] = best_model
    return model_map


def run_authentication_eval(
    dataset_dir: str = "./raw_kmt_dataset",
    preprocessed_dir: str = "results/preprocessed_kmt",
    output_dir: str = "results/evaluation",
    user_ids: Optional[Sequence[int]] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    split_seed: int = 42,
    threshold_mode: str = "validation_eer",
    genuine_quantile: float = 0.05,
    feature_set: str = "all",
    distribution_selection: str = "local_aic",
    window_size: float = 5.0,
    stride: float = 1.0,
    include_gmm: bool = True,
    gmm_n_components: int = 2,
    gmm_random_state: int = 0,
    weight_global_aic: bool = False,
) -> dict:
    """
    End-to-end leakage-free authentication evaluation.

    Score unit = session (or segment / time_block). Impostors = other users' TEST
    partitions under the enrolled user's train-fitted transform + models.
    """
    feature_columns = resolve_feature_columns(feature_set)
    if threshold_mode not in ("genuine_quantile", "validation_eer"):
        raise ValueError(f"Unsupported threshold_mode={threshold_mode!r}")

    from loss_compare import discover_user_ids

    all_ids = discover_user_ids(dataset_dir)
    if user_ids is None:
        selected = list(all_ids)
    else:
        selected = sorted({int(u) for u in user_ids})
        missing = [u for u in selected if u not in all_ids]
        if missing:
            raise FileNotFoundError(f"Users not found in dataset: {missing}")

    if not selected:
        raise FileNotFoundError(f"No users to evaluate under {dataset_dir}")

    os.makedirs(output_dir, exist_ok=True)
    feature_cache: Dict[int, pd.DataFrame] = {}

    # Load raw (untransformed) frames with the configured window/stride
    user_units = {}
    for uid in selected:
        df = _load_raw_feature_frame(
            uid,
            dataset_dir,
            preprocessed_dir,
            feature_cache,
            window_size=window_size,
            stride=stride,
        )
        if df.empty:
            warnings.warn(f"user {uid:04d}: empty feature frame; skipping")
            continue
        user_units[uid] = _units_from_frame(df)

    if not user_units:
        raise RuntimeError("No usable user feature frames")

    assignments = build_split_assignments(
        user_units,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    assignments.assert_all_disjoint()

    labeled: Dict[int, pd.DataFrame] = {}
    fallback_counts = {"session": 0, "segment": 0, "time_block": 0}
    for uid, asg in assignments.by_user.items():
        fallback_counts[asg.split_unit] = fallback_counts.get(asg.split_unit, 0) + 1
        raw = feature_cache[uid]
        labeled[uid] = _apply_split_labels(raw, asg)

    global_model_map = None
    if distribution_selection == "global_weighted_aic":
        global_model_map = _build_global_model_map_from_train(
            labeled,
            feature_columns,
            include_gmm=include_gmm,
            gmm_n_components=gmm_n_components,
            gmm_random_state=gmm_random_state,
            weight_by_n_used=bool(weight_global_aic),
        )

    score_rows: List[dict] = []
    per_feature_rows: List[dict] = []
    val_score_rows: List[dict] = []
    transform_rows: List[dict] = []
    fitted_rows: List[dict] = []
    per_user_metric_rows: List[dict] = []

    pooled_genuine: List[float] = []
    pooled_impostor: List[float] = []
    pooled_threshold_decisions_g: List[Tuple[float, float]] = []  # (score, threshold)
    pooled_threshold_decisions_i: List[Tuple[float, float]] = []

    for enrolled in selected:
        if enrolled not in labeled:
            continue
        asg = assignments.by_user[enrolled]
        edf = labeled[enrolled]
        train_df = edf[edf["split"] == "train"]
        val_df = edf[edf["split"] == "validation"]
        test_df = edf[edf["split"] == "test"]
        if train_df.empty or val_df.empty or test_df.empty:
            warnings.warn(f"user {enrolled:04d}: empty train/val/test after split; skip")
            continue

        params = fit_transform_params(train_df, feature_columns)
        transform_rows.extend(transform_params_to_rows(params, enrolled))

        train_tx = transform_features(train_df, params, feature_columns=feature_columns)
        fitted, model_map, fit_meta = _fit_models_for_enrollment(
            train_tx,
            feature_columns,
            distribution_selection=distribution_selection,
            global_model_map=global_model_map,
            include_gmm=include_gmm,
            gmm_n_components=gmm_n_components,
            gmm_random_state=gmm_random_state,
        )
        if not fitted:
            warnings.warn(f"user {enrolled:04d}: no fitted models; skip")
            continue
        for row in fit_meta:
            row = dict(row)
            row["enrolled_user"] = int(enrolled)
            fitted_rows.append(row)

        # Validation genuine attempt scores
        val_scores = []
        for attempt_id in _attempt_ids(val_df, asg.split_unit):
            part = _slice_attempt(val_df, asg.split_unit, attempt_id)
            if part.empty:
                continue
            sr = score_user(part, fitted, transform_params=params, feature_columns=feature_columns)
            if not np.isfinite(sr.mean_log_likelihood):
                continue
            val_scores.append(sr.mean_log_likelihood)
            val_score_rows.append(
                {
                    "enrolled_user": int(enrolled),
                    "eval_user": int(enrolled),
                    "eval_split": "validation",
                    "label": "genuine",
                    "group_id": attempt_id,
                    "score_ll": sr.mean_log_likelihood,
                    "n_features_used": sr.n_features_used,
                }
            )

        if not val_scores:
            warnings.warn(f"user {enrolled:04d}: no validation scores; skip")
            continue
        if len(val_scores) < 10:
            warnings.warn(
                f"user {enrolled:04d}: n_validation_scores={len(val_scores)} < 10; "
                f"{threshold_mode} threshold may be unstable"
            )

        val_median = float(np.median(val_scores))
        val_mean = float(np.mean(val_scores))

        if threshold_mode == "genuine_quantile":
            threshold = calibrate_genuine_quantile_threshold(val_scores, quantile=genuine_quantile)
            threshold_supervision = "genuine_validation_only"
        elif threshold_mode == "validation_eer":
            # EER on validation genuines vs other users' validation impostors
            imp_val = []
            for other in selected:
                if other == enrolled or other not in labeled:
                    continue
                odf = labeled[other]
                oval = odf[odf["split"] == "validation"]
                o_asg = assignments.by_user[other]
                for attempt_id in _attempt_ids(oval, o_asg.split_unit):
                    part = _slice_attempt(oval, o_asg.split_unit, attempt_id)
                    if part.empty:
                        continue
                    sr = score_user(
                        part, fitted, transform_params=params, feature_columns=feature_columns
                    )
                    if np.isfinite(sr.mean_log_likelihood):
                        imp_val.append(sr.mean_log_likelihood)
            if not imp_val:
                threshold = calibrate_genuine_quantile_threshold(
                    val_scores, quantile=genuine_quantile
                )
                threshold_supervision = "genuine_validation_only_fallback"
                warnings.warn(
                    f"user {enrolled:04d}: no validation impostors for EER; "
                    f"falling back to genuine_quantile={genuine_quantile}"
                )
            else:
                _, threshold = compute_eer(val_scores, imp_val)
                threshold_supervision = "genuine_and_impostor_validation"
        else:
            raise ValueError(f"Unsupported threshold_mode={threshold_mode!r}")

        genuine_test_scores: List[float] = []
        impostor_test_scores: List[float] = []

        def _record_attempt(
            eval_user: int,
            split_name: str,
            label: str,
            group_id: str,
            part: pd.DataFrame,
            score_list: List[float],
        ):
            sr = score_user(part, fitted, transform_params=params, feature_columns=feature_columns)
            if not np.isfinite(sr.mean_log_likelihood):
                return
            score = sr.mean_log_likelihood
            score_list.append(score)
            delta_val = float(score - val_median)
            risk_margin = float(threshold - score)
            decision = "accept" if score >= threshold else "reject"
            score_rows.append(
                {
                    "enrolled_user": int(enrolled),
                    "eval_user": int(eval_user),
                    "eval_split": split_name,
                    "label": label,
                    "group_id": group_id,
                    "score_ll": score,
                    "delta_val": delta_val,
                    "validation_median_ll": val_median,
                    "validation_mean_ll": val_mean,
                    "threshold": float(threshold),
                    "decision": decision,
                    "risk_margin": risk_margin,
                    "risk": float(max(0.0, risk_margin)),
                    "legacy_risk": float("nan"),  # not used in paper eval
                    "n_features_used": sr.n_features_used,
                    "threshold_supervision": threshold_supervision,
                }
            )
            for feat, ll in sr.per_feature_ll.items():
                per_feature_rows.append(
                    {
                        "enrolled_user": int(enrolled),
                        "eval_user": int(eval_user),
                        "split": split_name,
                        "label": label,
                        "group_id": group_id,
                        "feature": feat,
                        "mean_ll": ll,
                        "n_used": sr.per_feature_n.get(feat, 0),
                    }
                )

        for attempt_id in _attempt_ids(test_df, asg.split_unit):
            part = _slice_attempt(test_df, asg.split_unit, attempt_id)
            if part.empty:
                continue
            _record_attempt(
                enrolled, "test", "genuine", attempt_id, part, genuine_test_scores
            )

        for other in selected:
            if other == enrolled or other not in labeled:
                continue
            odf = labeled[other]
            otest = odf[odf["split"] == "test"]
            o_asg = assignments.by_user[other]
            for attempt_id in _attempt_ids(otest, o_asg.split_unit):
                part = _slice_attempt(otest, o_asg.split_unit, attempt_id)
                if part.empty:
                    continue
                _record_attempt(
                    other, "test", "impostor", attempt_id, part, impostor_test_scores
                )

        if not genuine_test_scores or not impostor_test_scores:
            warnings.warn(f"user {enrolled:04d}: missing genuine or impostor test scores")
            continue

        metrics = compute_binary_metrics(
            genuine_test_scores, impostor_test_scores, threshold=threshold
        )
        metrics_row = {
            "enrolled_user": int(enrolled),
            "split_unit": asg.split_unit,
            "fallback_reason": asg.fallback_reason or "",
            "n_train_groups": len(asg.train),
            "n_val_groups": len(asg.validation),
            "n_test_groups": len(asg.test),
            "validation_median_ll": val_median,
            "threshold": float(threshold),
            **metrics,
        }
        per_user_metric_rows.append(metrics_row)

        pooled_genuine.extend(genuine_test_scores)
        pooled_impostor.extend(impostor_test_scores)
        for s in genuine_test_scores:
            pooled_threshold_decisions_g.append((s, threshold))
        for s in impostor_test_scores:
            pooled_threshold_decisions_i.append((s, threshold))

    # Macro + pooled summary
    per_user_df = pd.DataFrame(per_user_metric_rows)
    summary_rows = []
    if not per_user_df.empty:
        for metric in ("roc_auc", "far", "frr", "eer"):
            series = pd.to_numeric(per_user_df[metric], errors="coerce")
            summary_rows.append(
                {
                    "metric": metric,
                    "macro": float(series.mean(skipna=True)),
                    "pooled": float("nan"),
                }
            )
        # Pooled FAR/FRR use each attempt's enrolled threshold
        if pooled_threshold_decisions_g and pooled_threshold_decisions_i:
            g_accept = [1.0 if s >= t else 0.0 for s, t in pooled_threshold_decisions_g]
            i_accept = [1.0 if s >= t else 0.0 for s, t in pooled_threshold_decisions_i]
            pooled_frr = 1.0 - float(np.mean(g_accept))
            pooled_far = float(np.mean(i_accept))
            pooled_auc = roc_auc_score(pooled_genuine, pooled_impostor)
            pooled_eer, _ = compute_eer(pooled_genuine, pooled_impostor)
            pooled_map = {
                "roc_auc": pooled_auc,
                "far": pooled_far,
                "frr": pooled_frr,
                "eer": pooled_eer,
            }
            for row in summary_rows:
                row["pooled"] = float(pooled_map[row["metric"]])

    summary_df = pd.DataFrame(summary_rows)
    scores_df = pd.DataFrame(score_rows)
    per_feat_df = pd.DataFrame(per_feature_rows)
    val_df_out = pd.DataFrame(val_score_rows)
    transform_df = pd.DataFrame(transform_rows)
    fitted_df = pd.DataFrame(fitted_rows)
    split_df = pd.DataFrame(assignments.to_rows())

    config = {
        "purpose": "authentication_evaluation",
        "dataset_dir": dataset_dir,
        "preprocessed_dir": preprocessed_dir,
        "n_users": len(selected),
        "user_ids": selected,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "threshold_mode": threshold_mode,
        "genuine_quantile": genuine_quantile,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "distribution_selection": distribution_selection,
        "window_size": window_size,
        "stride": stride,
        "include_gmm": bool(include_gmm),
        "gmm_n_components": int(gmm_n_components),
        "gmm_random_state": int(gmm_random_state),
        "weight_global_aic": bool(weight_global_aic),
        "split_unit_counts": fallback_counts,
        "n_users_time_block_fallback": sum(
            1
            for a in assignments.by_user.values()
            if a.split_unit == "time_block"
        ),
        "n_users_segment_fallback": sum(
            1 for a in assignments.by_user.values() if a.split_unit == "segment"
        ),
        "notes": (
            "KMT primary split unit is test_N as session (no native session UUID). "
            "See DEVELOPMENT.md. Decision threshold from validation_eer (default); "
            "test EER is reporting-only."
            + (
                " global_weighted_aic shares distribution family across cohort "
                "train partitions (not test leakage); params remain per-enrollee. "
                + (
                    "Cohort AIC uses n_used weights (--weight-global-aic)."
                    if weight_global_aic
                    else "Cohort AIC is unweighted mean over users (default)."
                )
                if str(distribution_selection).strip().lower() == "global_weighted_aic"
                else ""
            )
        ),
    }

    paths = {
        "experiment_config": os.path.join(output_dir, "experiment_config.json"),
        "split_assignments": os.path.join(output_dir, "split_assignments.csv"),
        "fitted_models": os.path.join(output_dir, "fitted_models.csv"),
        "transform_params": os.path.join(output_dir, "transform_params.csv"),
        "validation_scores": os.path.join(output_dir, "validation_scores.csv"),
        "authentication_scores": os.path.join(output_dir, "authentication_scores.csv"),
        "authentication_scores_per_feature": os.path.join(
            output_dir, "authentication_scores_per_feature.csv"
        ),
        "authentication_metrics_per_user": os.path.join(
            output_dir, "authentication_metrics_per_user.csv"
        ),
        "authentication_metrics_summary": os.path.join(
            output_dir, "authentication_metrics_summary.csv"
        ),
    }

    with open(paths["experiment_config"], "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    split_df.to_csv(paths["split_assignments"], index=False)
    fitted_df.to_csv(paths["fitted_models"], index=False)
    transform_df.to_csv(paths["transform_params"], index=False)
    val_df_out.to_csv(paths["validation_scores"], index=False)
    scores_df.to_csv(paths["authentication_scores"], index=False)
    per_feat_df.to_csv(paths["authentication_scores_per_feature"], index=False)
    per_user_df.to_csv(paths["authentication_metrics_per_user"], index=False)
    summary_df.to_csv(paths["authentication_metrics_summary"], index=False)

    return {
        "config": config,
        "paths": paths,
        "summary": summary_df,
        "per_user": per_user_df,
        "scores": scores_df,
    }
