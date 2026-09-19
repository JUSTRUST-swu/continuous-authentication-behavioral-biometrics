"""Scoring/model-selection helpers shared across auth/compare/api."""

from loss_compare import (  # noqa: F401
    _logpdf_by_model,
    discover_user_ids,
    fit_feature_models_on_user,
    resolve_include_gmm_for_model_map,
    score_user_against_fitted_models,
    select_models_by_aic_on_user,
    user_json_path,
)
from main import get_model_fitters  # noqa: F401

