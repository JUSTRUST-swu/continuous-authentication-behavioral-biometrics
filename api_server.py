import argparse
import datetime as dt
import glob
import json
import os
import re
import threading
import uuid
from typing import Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, request

import numpy as np

from feature_transform import FeatureTransformParams, fit_transform_params, transform_features
from loss_compare import fit_feature_models_on_user, score_user_against_fitted_models
from main import (
    FEATURE_COLUMNS,
    aggregate_per_user_model_fits,
    build_feature_df_for_user,
    evaluate_feature_models,
)


MODEL_LOCK = threading.Lock()
SESSION_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_CRITERION_COL = "best_weighted_mean_aic"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sanitize_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("sessionId must be a non-empty string")
    cleaned = SESSION_ID_SAFE.sub("_", session_id.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("sessionId became empty after sanitization")
    return cleaned


def _make_log_basename(session_id: str = None) -> str:
    if isinstance(session_id, str) and session_id.strip():
        return _sanitize_session_id(session_id)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _save_log_json(base_dir: str, mode: str, log_data: dict, session_id: str = None) -> str:
    if mode not in ("train", "validate"):
        raise ValueError("mode must be 'train' or 'validate'")
    safe_id = _make_log_basename(session_id=session_id)
    target_dir = os.path.join(base_dir, mode)
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, f"{safe_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False)
    return out_path


def _list_json_files(directory: str) -> List[str]:
    return sorted(glob.glob(os.path.join(directory, "*.json")))


def _clear_training_state(logs_base_dir: str, model_state_path: str) -> dict:
    train_dir = os.path.join(logs_base_dir, "train")
    removed_train_logs = 0
    if os.path.isdir(train_dir):
        for path in _list_json_files(train_dir):
            try:
                os.remove(path)
                removed_train_logs += 1
            except OSError:
                pass

    removed_model_state = False
    if os.path.isfile(model_state_path):
        try:
            os.remove(model_state_path)
            removed_model_state = True
        except OSError:
            removed_model_state = False

    return {
        "removedTrainLogs": removed_train_logs,
        "removedModelState": removed_model_state,
    }


def _build_merged_feature_df_from_files(file_paths: List[str], apply_transform: bool = True) -> pd.DataFrame:
    frames = []
    for path in file_paths:
        frame = build_feature_df_for_user(
            path, preprocessed_dir="", logger=None, apply_transform=apply_transform
        )
        if not frame.empty:
            frame = frame.copy()
            frame["user_file"] = os.path.basename(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["window_start", "window_end", *FEATURE_COLUMNS, "data_group", "user_file"]
        )
    return pd.concat(frames, ignore_index=True)


def _choose_models_from_summary(summary_df: pd.DataFrame, criterion_col: str) -> Dict[str, str]:
    if criterion_col not in summary_df.columns:
        raise ValueError(f"criterion column not found in summary: {criterion_col}")
    model_map: Dict[str, str] = {}
    for _, row in summary_df.iterrows():
        feature = str(row.get("feature", ""))
        model_name = row.get(criterion_col)
        if feature in FEATURE_COLUMNS and isinstance(model_name, str) and model_name.strip():
            model_map[feature] = model_name
    return model_map


def _to_float_or_none(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _serialize_feature_models(fitted_models: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for feature in FEATURE_COLUMNS:
        fit = fitted_models.get(feature)
        if fit is None:
            out[feature] = {"model": None, "params": None, "n_used": None}
            continue
        params = fit.get("params", {})
        out[feature] = {
            "model": fit.get("model"),
            "params": {k: _to_float_or_none(v) for k, v in params.items()},
            "n_used": int(fit.get("n_used", 0)) if fit.get("n_used") is not None else None,
        }
    return out


def _load_persisted_model(model_state_path: str) -> dict:
    if not os.path.isfile(model_state_path):
        raise FileNotFoundError("No trained model state found")
    with open(model_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state


def _reconstruct_fitted_models(feature_models: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for feature in FEATURE_COLUMNS:
        item = feature_models.get(feature, {})
        model = item.get("model")
        params = item.get("params")
        if isinstance(model, str) and isinstance(params, dict):
            out[feature] = {"model": model, "params": params}
    return out


def train_with_cumulative_logs(
    logs_base_dir: str,
    criterion_col: str,
    model_state_path: str,
) -> Tuple[dict, dict]:
    train_dir = os.path.join(logs_base_dir, "train")
    train_files = _list_json_files(train_dir)
    if not train_files:
        raise ValueError("No train logs available. Send train data first.")

    # Raw features → fit clip on train only → transform → fit models (no validate refit later)
    raw_df = _build_merged_feature_df_from_files(train_files, apply_transform=False)
    if raw_df.empty:
        raise ValueError("No usable feature rows from train logs.")

    transform_params = fit_transform_params(raw_df, FEATURE_COLUMNS)
    merged_df = transform_features(raw_df, transform_params, feature_columns=FEATURE_COLUMNS)

    result_rows = []
    for src_name, group_df in merged_df.groupby("user_file"):
        for feature in FEATURE_COLUMNS:
            result_rows.extend(
                evaluate_feature_models(feature, group_df[feature], user_file=src_name)
            )
    if not result_rows:
        raise ValueError("No model fit rows produced from train logs.")

    per_user_df = pd.DataFrame(result_rows)
    summary_df, _, _ = aggregate_per_user_model_fits(per_user_df)
    model_map = _choose_models_from_summary(summary_df, criterion_col=criterion_col)
    fitted_models = fit_feature_models_on_user(merged_df, model_map)

    baseline_feature_df = score_user_against_fitted_models(merged_df, fitted_models)
    baseline_feature_ll = {}
    for _, row in baseline_feature_df.iterrows():
        baseline_feature_ll[str(row["feature"])] = _to_float_or_none(row["mean_log_likelihood"])

    baseline_mean_ll = _to_float_or_none(
        baseline_feature_df["mean_log_likelihood"].mean(skipna=True)
    )
    # Online API has no held-out val split by default: store train median as soft baseline
    # and a genuine_quantile threshold from train attempt-level scores (documented).
    train_scores = []
    for _, gdf in merged_df.groupby("user_file"):
        score_df = score_user_against_fitted_models(gdf, fitted_models)
        train_scores.append(float(score_df["mean_log_likelihood"].mean(skipna=True)))
    validation_baseline = float(np.median(train_scores)) if train_scores else baseline_mean_ll
    try:
        from auth_metrics import calibrate_genuine_quantile_threshold

        threshold = (
            calibrate_genuine_quantile_threshold(train_scores, quantile=0.05)
            if len(train_scores) >= 1
            else None
        )
    except Exception:
        threshold = None

    model_state = {
        "updated_at": _utc_now_iso(),
        "criterion_col": criterion_col,
        "n_training_sessions": len(train_files),
        "n_training_rows": int(len(merged_df)),
        "feature_models": _serialize_feature_models(fitted_models),
        "transform_params": transform_params.to_dict(),
        "baseline_per_feature_mean_ll": baseline_feature_ll,
        "baseline_mean_log_likelihood": baseline_mean_ll,
        "baseline_mean_nll": _to_float_or_none(
            baseline_feature_df["mean_nll"].mean(skipna=True)
        ),
        "validation_baseline": validation_baseline,
        "threshold": threshold,
        "threshold_note": (
            "Online train logs have no separate val split; threshold uses train-session "
            "genuine_quantile=0.05. Prefer offline authentication_eval for paper metrics."
        ),
    }

    os.makedirs(os.path.dirname(model_state_path) or ".", exist_ok=True)
    with open(model_state_path, "w", encoding="utf-8") as f:
        json.dump(model_state, f, ensure_ascii=False, indent=2)

    feature_resp = {}
    for feature in FEATURE_COLUMNS:
        feature_resp[feature] = model_state["feature_models"].get(feature, {})

    response_payload = {
        "nTrainingSessions": model_state["n_training_sessions"],
        "nTrainingRows": model_state["n_training_rows"],
        "criterionCol": criterion_col,
        "features": feature_resp,
        "baseline": {
            "mean_log_likelihood": model_state["baseline_mean_log_likelihood"],
            "mean_nll": model_state["baseline_mean_nll"],
            "validation_baseline": validation_baseline,
            "threshold": threshold,
        },
        "transformParamsStored": True,
    }
    return model_state, response_payload


def validate_with_persisted_model(
    validate_file_path: str,
    model_state: dict,
) -> dict:
    feature_models = model_state.get("feature_models", {})
    fitted_models = _reconstruct_fitted_models(feature_models)
    if not fitted_models:
        raise ValueError("Persisted model has no fitted feature models.")

    # Never refit: apply saved train clip bounds, then score
    eval_raw = build_feature_df_for_user(
        validate_file_path, preprocessed_dir="", logger=None, apply_transform=False
    )
    if eval_raw.empty:
        raise ValueError("No usable feature rows from validate log.")

    tp_raw = model_state.get("transform_params")
    if isinstance(tp_raw, dict) and tp_raw:
        params = FeatureTransformParams.from_dict(tp_raw)
        eval_df = transform_features(eval_raw, params, feature_columns=FEATURE_COLUMNS)
    else:
        # Legacy model state without transform_params
        eval_df = build_feature_df_for_user(
            validate_file_path, preprocessed_dir="", logger=None, apply_transform=True
        )

    feature_df = score_user_against_fitted_models(eval_df, fitted_models)
    baseline_mean_ll = _to_float_or_none(model_state.get("baseline_mean_log_likelihood"))
    mean_ll = _to_float_or_none(feature_df["mean_log_likelihood"].mean(skipna=True))
    mean_nll = _to_float_or_none(feature_df["mean_nll"].mean(skipna=True))
    ll_diff = None if (mean_ll is None or baseline_mean_ll is None) else float(mean_ll - baseline_mean_ll)
    legacy_risk = None if ll_diff is None else float(max(0.0, -ll_diff))

    threshold = _to_float_or_none(model_state.get("threshold"))
    validation_baseline = _to_float_or_none(model_state.get("validation_baseline"))
    decision = None
    risk_margin = None
    risk = None
    delta_val = None
    if mean_ll is not None and threshold is not None:
        decision = "accept" if mean_ll >= threshold else "reject"
        risk_margin = float(threshold - mean_ll)
        risk = float(max(0.0, risk_margin))
    if mean_ll is not None and validation_baseline is not None:
        delta_val = float(mean_ll - validation_baseline)

    per_feature = {}
    for _, row in feature_df.iterrows():
        feature = str(row["feature"])
        per_feature[feature] = {
            "model": row.get("model"),
            "n_used": int(row.get("n_used", 0)),
            "mean_log_likelihood": _to_float_or_none(row.get("mean_log_likelihood")),
            "total_log_likelihood": _to_float_or_none(row.get("total_log_likelihood")),
            "mean_nll": _to_float_or_none(row.get("mean_nll")),
            "total_nll": _to_float_or_none(row.get("total_nll")),
        }

    return {
        "features": per_feature,
        "score": mean_ll,
        "threshold": threshold,
        "margin": risk_margin,
        "risk": risk,
        "decision": decision,
        "loss": {
            "mean_log_likelihood": mean_ll,
            "mean_nll": mean_nll,
            "baseline_mean_log_likelihood": baseline_mean_ll,
            "validation_baseline": validation_baseline,
            "delta_val": delta_val,
            "ll_diff_vs_train": ll_diff,
            "risk_score": legacy_risk,
            "legacy_risk": legacy_risk,
            "risk_margin": risk_margin,
            "decision": decision,
        },
    }


def create_app(
    logs_base_dir: str = "logs",
    model_state_path: str = "results/api_model/fitted_models.json",
    default_criterion_col: str = DEFAULT_CRITERION_COL,
) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        # Allow local-file origin ("null") and localhost frontend usage.
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

    @app.get("/api/health")
    def health():
        with MODEL_LOCK:
            model_exists = os.path.isfile(model_state_path)
            train_logs = _list_json_files(os.path.join(logs_base_dir, "train"))
            model_info = None
            if model_exists:
                try:
                    state = _load_persisted_model(model_state_path)
                    model_info = {
                        "updated_at": state.get("updated_at"),
                        "n_training_sessions": state.get("n_training_sessions"),
                    }
                except Exception:
                    model_info = None
        return jsonify(
            {
                "status": "ok",
                "model_exists": model_exists,
                "model_info": model_info,
                "n_train_logs": len(train_logs),
            }
        )

    @app.post("/api/session")
    def session_api():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be JSON object"}), 400

        session_id = payload.get("sessionId")
        data_type = payload.get("dataType")
        log_data = payload.get("log")
        criterion_col = payload.get("criterionCol", default_criterion_col)

        if data_type not in ("train", "validate", "clear"):
            return jsonify({"error": "dataType must be one of: train, validate, clear"}), 400
        if data_type in ("train", "validate") and not isinstance(log_data, dict):
            return jsonify({"error": "log must be a JSON object"}), 400

        try:
            with MODEL_LOCK:
                if data_type == "clear":
                    clear_result = _clear_training_state(
                        logs_base_dir=logs_base_dir,
                        model_state_path=model_state_path,
                    )
                    return jsonify(
                        {
                            "dataType": data_type,
                            "status": "cleared",
                            **clear_result,
                        }
                    )

                saved_path = _save_log_json(
                    base_dir=logs_base_dir,
                    mode=data_type,
                    log_data=log_data,
                    session_id=session_id,
                )
                if data_type == "train":
                    _, train_resp = train_with_cumulative_logs(
                        logs_base_dir=logs_base_dir,
                        criterion_col=criterion_col,
                        model_state_path=model_state_path,
                    )
                    return jsonify(
                        {
                            "dataType": data_type,
                            "sessionId": _sanitize_session_id(session_id)
                            if isinstance(session_id, str) and session_id.strip()
                            else None,
                            "savedPath": saved_path,
                            **train_resp,
                        }
                    )

                state = _load_persisted_model(model_state_path)
                validate_resp = validate_with_persisted_model(
                    validate_file_path=saved_path,
                    model_state=state,
                )
                return jsonify(
                    {
                        "dataType": data_type,
                        "sessionId": _sanitize_session_id(session_id)
                        if isinstance(session_id, str) and session_id.strip()
                        else None,
                        "savedPath": saved_path,
                        **validate_resp,
                    }
                )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 409
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"Internal error: {exc}"}), 500

    @app.post("/api/clear")
    def clear_api():
        with MODEL_LOCK:
            clear_result = _clear_training_state(
                logs_base_dir=logs_base_dir,
                model_state_path=model_state_path,
            )
        return jsonify(
            {
                "status": "cleared",
                **clear_result,
            }
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Keystroke biometrics train/validate API server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--logs-base-dir", default="logs")
    parser.add_argument("--model-state-path", default="results/api_model/fitted_models.json")
    parser.add_argument("--criterion-col", default=DEFAULT_CRITERION_COL)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(
        logs_base_dir=args.logs_base_dir,
        model_state_path=args.model_state_path,
        default_criterion_col=args.criterion_col,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
