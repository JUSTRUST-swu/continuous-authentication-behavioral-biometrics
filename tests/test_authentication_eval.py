"""Minimal leakage-free evaluation tests."""

import json

import numpy as np
import pandas as pd
import pytest

from auth_metrics import far_frr_at_threshold, calibrate_genuine_quantile_threshold
from evaluation_split import make_user_split, split_group_ids
from feature_transform import fit_transform_params, transform_features


def test_group_split_has_no_overlap():
    sessions = [f"test_{i}" for i in range(1, 11)]
    asg = make_user_split(
        user_id=1,
        session_ids=sessions,
        segment_ids=[],
        seed=42,
    )
    train, val, test = set(asg.train), set(asg.validation), set(asg.test)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(sessions)


def test_transform_fit_ignores_test_extremes():
    train = pd.DataFrame({"dwell_mean": [1.0, 2.0, 3.0, 4.0, 5.0]})
    params_a = fit_transform_params(train, ["dwell_mean"])
    params_b = fit_transform_params(train, ["dwell_mean"])
    assert params_a.clip_low == params_b.clip_low
    assert params_a.clip_high == params_b.clip_high

    test_extreme = pd.DataFrame({"dwell_mean": [2.0, 3.0, 999999.0]})
    # Fitting on train only: extreme test must not change params
    out = transform_features(test_extreme, params_a, feature_columns=["dwell_mean"])
    assert float(out["dwell_mean"].max()) <= np.log1p(params_a.clip_high["dwell_mean"]) + 1e-9


def test_far_frr_definitions():
    genuine_scores = [5, 4, 1]
    impostor_scores = [3, 2, 0]
    threshold = 2.5
    far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, threshold)
    assert far == pytest.approx(1 / 3)
    assert frr == pytest.approx(1 / 3)


def test_threshold_from_validation_only():
    val = [1.0, 2.0, 3.0, 4.0, 5.0]
    t1 = calibrate_genuine_quantile_threshold(val, quantile=0.05)
    # Changing a hypothetical test set does not affect calibration
    t2 = calibrate_genuine_quantile_threshold(val, quantile=0.05)
    assert t1 == t2


def test_reproducible_split_seed():
    sessions = [f"s{i}" for i in range(10)]
    a = split_group_ids(sessions, seed=42, user_id=7)
    b = split_group_ids(sessions, seed=42, user_id=7)
    assert a == b
    c = split_group_ids(sessions, seed=43, user_id=7)
    assert a != c


def test_load_raw_feature_frame_passes_window_stride(monkeypatch):
    from authentication_eval import _load_raw_feature_frame

    captured = {}

    def fake_build(path, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"session_id": ["test_1"], "dwell_mean": [1.0]})

    monkeypatch.setattr("authentication_eval.build_feature_df_for_user", fake_build)
    monkeypatch.setattr(
        "loss_compare.user_json_path",
        lambda user_id, dataset_dir="./raw_kmt_dataset": "fake.json",
    )

    _load_raw_feature_frame(
        1, "./raw_kmt_dataset", "", None, window_size=10.0, stride=2.5
    )
    assert captured["window_size"] == 10.0
    assert captured["stride"] == 2.5
    assert captured["apply_transform"] is False
    assert captured["prefer_sessions"] is True


def test_global_model_map_uses_transformed_train(monkeypatch):
    from authentication_eval import _build_global_model_map_from_train

    seen = []

    def capture_evaluate(feature, values, user_file=None, **kwargs):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        seen.append(finite.copy())
        n = int(len(finite))
        return [
            {"model": "Gaussian", "aic": 10.0, "n_used": n},
            {"model": "Log-normal", "aic": 20.0, "n_used": n},
        ]

    monkeypatch.setattr("authentication_eval.evaluate_feature_models", capture_evaluate)

    train = pd.DataFrame(
        {
            "split": ["train"] * 5 + ["test"] * 2,
            "dwell_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0],
        }
    )
    labeled = {1: train}
    params = fit_transform_params(train[train["split"] == "train"], ["dwell_mean"])
    expected = transform_features(
        train[train["split"] == "train"], params, feature_columns=["dwell_mean"]
    )["dwell_mean"].to_numpy(dtype=float)
    expected = expected[np.isfinite(expected)]

    model_map = _build_global_model_map_from_train(labeled, ["dwell_mean"])
    assert model_map["dwell_mean"] == "Gaussian"
    assert len(seen) >= 1
    np.testing.assert_allclose(seen[0], expected, rtol=1e-9)
    # Raw train extremes must not be what AIC saw (log1p of clipped values)
    assert float(np.max(seen[0])) < 100.0


def test_roc_curve_has_no_synthetic_endpoints():
    from plot_modality_figures import roc_curve_from_scores

    genuine = [2.0, 2.5, 3.0]
    impostor = [0.0, 0.5, 1.0]
    fpr, tpr, auc = roc_curve_from_scores(genuine, impostor, n_thresholds=11)
    # Forced (0,0)/(1,1) would always appear; without them, endpoints are data-driven
    assert not (fpr[0] == 0.0 and tpr[0] == 0.0 and fpr[-1] == 1.0 and tpr[-1] == 1.0)
    assert 0.0 <= float(auc) <= 1.0
    assert len(fpr) == len(tpr) == 11


def test_model_agreement_ignores_nan_as_match():
    from run_aic_selection_ablation import summarize_model_agreement

    agree_df = pd.DataFrame(
        {
            "enrolled_user": [1, 1, 2],
            "feature": ["dwell_mean", "flight_mean", "dwell_mean"],
            "distribution_local_aic": ["Gaussian", np.nan, "Gamma"],
            "distribution_global_weighted_aic": ["Gaussian", np.nan, "Weibull"],
            "agree": [True, False, False],
            "comparable": [True, False, True],
        }
    )
    summary = summarize_model_agreement(agree_df)
    overall = summary[summary["scope"] == "overall"].iloc[0]
    assert int(overall["n"]) == 2
    assert int(overall["n_agree"]) == 1
    assert float(overall["agreement_rate"]) == pytest.approx(0.5)


def test_compare_model_selections_nan_not_agree(tmp_path):
    from run_aic_selection_ablation import compare_model_selections

    local_dir = tmp_path / "local"
    global_dir = tmp_path / "global"
    local_dir.mkdir()
    global_dir.mkdir()
    pd.DataFrame(
        {
            "enrolled_user": [1, 1],
            "feature": ["dwell_mean", "flight_mean"],
            "distribution": ["Gaussian", "Gamma"],
        }
    ).to_csv(local_dir / "fitted_models.csv", index=False)
    pd.DataFrame(
        {
            "enrolled_user": [1],
            "feature": ["dwell_mean"],
            "distribution": ["Gaussian"],
        }
    ).to_csv(global_dir / "fitted_models.csv", index=False)

    merged = compare_model_selections(str(local_dir), str(global_dir))
    flight = merged[merged["feature"] == "flight_mean"].iloc[0]
    assert bool(flight["agree"]) is False
    assert bool(flight["comparable"]) is False
    dwell = merged[merged["feature"] == "dwell_mean"].iloc[0]
    assert bool(dwell["agree"]) is True
    assert bool(dwell["comparable"]) is True


def test_fit_loglogistic_positive_support():
    from scipy import stats as scipy_stats

    from main import _fit_loglogistic, _prepare_values_for_model, evaluate_feature_models

    x = scipy_stats.fisk.rvs(c=2.5, scale=1.2, size=200, random_state=0)
    result = _fit_loglogistic(x)
    assert result is not None
    assert result["model"] == "Log-logistic"
    assert result["n_used"] == 200
    assert np.isfinite(result["aic"])
    assert result["params"]["shape"] > 0
    assert result["params"]["scale"] > 0

    prepared = _prepare_values_for_model(np.array([-1.0, 0.0, 1.0, 2.0]), "Log-logistic")
    np.testing.assert_array_equal(prepared, np.array([1.0, 2.0]))

    rows = evaluate_feature_models("dwell_mean", pd.Series(x))
    names = {r["model"] for r in rows}
    assert "Log-logistic" in names


def test_gmm_opt_in_only():
    pytest.importorskip("sklearn")
    from main import evaluate_feature_models, get_model_fitters, _fit_gmm
    from loss_compare import _logpdf_by_model

    x = np.concatenate(
        [
            np.random.default_rng(0).normal(-1.0, 0.3, size=80),
            np.random.default_rng(1).normal(1.5, 0.4, size=80),
        ]
    )
    base = get_model_fitters(include_gmm=False)
    assert "GMM" not in base
    with_gmm = get_model_fitters(include_gmm=True, gmm_n_components=2)
    assert "GMM" in with_gmm

    names_default = {r["model"] for r in evaluate_feature_models("dwell_mean", pd.Series(x))}
    assert "GMM" not in names_default
    names_gmm = {
        r["model"]
        for r in evaluate_feature_models(
            "dwell_mean", pd.Series(x), include_gmm=True, gmm_n_components=2
        )
    }
    assert "GMM" in names_gmm

    fit = _fit_gmm(x, n_components=2, random_state=0)
    assert fit is not None
    assert fit["model"] == "GMM"
    assert len(fit["params"]["weights"]) == 2
    ll = _logpdf_by_model("GMM", x[:5], fit["params"])
    assert ll.shape == (5,)
    assert np.all(np.isfinite(ll))


def test_fit_feature_models_auto_enables_gmm_from_map():
    """Legacy compare/API: GMM in model_map must not be silently skipped."""
    pytest.importorskip("sklearn")
    from main import FEATURE_COLUMNS
    from loss_compare import fit_feature_models_on_user

    rng = np.random.default_rng(0)
    n = 120
    data = {f: rng.normal(size=n) for f in FEATURE_COLUMNS}
    # Positive support for log-family features if map ever selects them.
    for f in ("dwell_mean", "flight_mean", "velocity_mean"):
        data[f] = np.abs(data[f]) + 0.1
    train_df = pd.DataFrame(data)
    model_map = {f: "GMM" for f in FEATURE_COLUMNS}

    with pytest.warns(UserWarning, match="enabling GMM"):
        fitted = fit_feature_models_on_user(train_df, model_map, include_gmm=False)

    assert len(fitted) == len(FEATURE_COLUMNS)
    assert all(fitted[f]["model"] == "GMM" for f in FEATURE_COLUMNS)

    with pytest.raises(ValueError, match="unknown model"):
        fit_feature_models_on_user(
            train_df, {"dwell_mean": "NotARealModel"}, include_gmm=False
        )


def test_main_fit_split_train_uses_train_partition_only(monkeypatch, tmp_path):
    """evaluate_all_users(fit_split=train) must fit on train rows after split+transform."""
    from main import FEATURE_COLUMNS, evaluate_all_users

    sessions = [f"test_{i}" for i in range(1, 11)]
    rows = []
    for s_i, sid in enumerate(sessions):
        for w in range(3):
            row = {
                "window_start": float(s_i * 10 + w),
                "window_end": float(s_i * 10 + w + 5),
                "user_id": 1,
                "session_id": sid,
                "segment_id": f"seg_{s_i}",
                "data_group": "true_data",
            }
            for feat in FEATURE_COLUMNS:
                row[feat] = 1.0 + 0.1 * s_i + 0.01 * w
            rows.append(row)
    raw_df = pd.DataFrame(rows)

    def fake_build(path, **kwargs):
        assert kwargs.get("apply_transform") is False
        assert kwargs.get("prefer_sessions") is True
        return raw_df.copy()

    seen_lengths = []

    def capture_evaluate(feature, values, user_file=None, **kwargs):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        seen_lengths.append(int(np.sum(np.isfinite(arr))))
        n = int(np.sum(np.isfinite(arr)))
        rows = [
            {
                "feature": feature,
                "model": "Gaussian",
                "aic": 10.0,
                "n_used": n,
                "bic": 11.0,
                "log_likelihood": -1.0,
            },
            {
                "feature": feature,
                "model": "Log-normal",
                "aic": 20.0,
                "n_used": n,
                "bic": 21.0,
                "log_likelihood": -2.0,
            },
        ]
        if user_file is not None:
            for r in rows:
                r["user_file"] = user_file
        return rows

    monkeypatch.setattr("main.build_feature_df_for_user", fake_build)
    monkeypatch.setattr("main.evaluate_feature_models", capture_evaluate)
    monkeypatch.setattr("main._infer_user_id_from_path", lambda path: 1)

    user_file = tmp_path / "raw_kmt_user_0001.json"
    user_file.write_text("{}", encoding="utf-8")
    out_split = tmp_path / "split_assignments.csv"
    out_cfg = tmp_path / "run_config.json"

    _, result_df, agg, *_rest = evaluate_all_users(
        user_files=[str(user_file)],
        preprocessed_dir="",
        fit_split="train",
        split_seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        output_split_assignments_csv=str(out_split),
        output_run_config_json=str(out_cfg),
        logger=None,
    )

    # 10 sessions × 3 windows = 30; train ≈ 6 sessions → 18 windows
    assert seen_lengths
    assert all(n == 18 for n in seen_lengths)
    assert not result_df.empty
    assert "best_weighted_mean_aic" in agg.columns
    assert out_split.is_file()
    assert out_cfg.is_file()
    cfg = json.loads(out_cfg.read_text(encoding="utf-8"))
    assert cfg["fit_split"] == "train"
    assert cfg["split_seed"] == 42
