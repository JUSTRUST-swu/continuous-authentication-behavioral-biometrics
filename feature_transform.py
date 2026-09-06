"""Train-only feature clipping + log1p transform (leakage-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureTransformParams:
    """Percentile clip bounds fitted on train only; then optional log1p."""

    clip_low: Dict[str, float]
    clip_high: Dict[str, float]
    use_log1p: bool = True
    feature_columns: tuple = ()

    def to_dict(self) -> dict:
        return {
            "clip_low": dict(self.clip_low),
            "clip_high": dict(self.clip_high),
            "use_log1p": bool(self.use_log1p),
            "feature_columns": list(self.feature_columns),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FeatureTransformParams":
        cols = tuple(data.get("feature_columns") or ())
        return cls(
            clip_low={str(k): float(v) for k, v in dict(data.get("clip_low") or {}).items()},
            clip_high={str(k): float(v) for k, v in dict(data.get("clip_high") or {}).items()},
            use_log1p=bool(data.get("use_log1p", True)),
            feature_columns=cols,
        )


def fit_transform_params(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    percentiles: Sequence[float] = (1.0, 99.0),
    use_log1p: bool = True,
) -> FeatureTransformParams:
    """
    Estimate per-feature clip bounds from ``df`` (train only).

    Bounds use finite values; if fewer than 2 finite values, that feature is left unclipped
    (lo=-inf, hi=+inf) so transform still applies log1p safely.
    """
    if len(percentiles) != 2:
        raise ValueError("percentiles must be a pair (low, high)")
    lo_p, hi_p = float(percentiles[0]), float(percentiles[1])
    clip_low: Dict[str, float] = {}
    clip_high: Dict[str, float] = {}
    for col in feature_columns:
        if col not in df.columns:
            clip_low[col] = float("-inf")
            clip_high[col] = float("inf")
            continue
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = arr[np.isfinite(arr)]
        if vals.size >= 2:
            lo, hi = np.percentile(vals, [lo_p, hi_p])
            clip_low[col] = float(lo)
            clip_high[col] = float(hi)
        else:
            clip_low[col] = float("-inf")
            clip_high[col] = float("inf")
    return FeatureTransformParams(
        clip_low=clip_low,
        clip_high=clip_high,
        use_log1p=bool(use_log1p),
        feature_columns=tuple(feature_columns),
    )


def transform_features(
    df: pd.DataFrame,
    params: FeatureTransformParams,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Apply fitted clip bounds (+ optional log1p) without re-estimating percentiles.

    Values <= -1 become NaN before log1p (same policy as legacy apply_clip_log_transform).
    Metadata columns are preserved unchanged.
    """
    cols = list(feature_columns) if feature_columns is not None else list(params.feature_columns)
    if not cols:
        cols = [c for c in params.clip_low.keys()]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(arr)
        transformed = np.full_like(arr, np.nan, dtype=float)
        if not np.any(valid):
            out[col] = transformed
            continue

        vals = arr[valid].astype(float, copy=True)
        lo = params.clip_low.get(col, float("-inf"))
        hi = params.clip_high.get(col, float("inf"))
        if np.isfinite(lo) or np.isfinite(hi):
            vals = np.clip(vals, lo, hi)

        if params.use_log1p:
            vals[vals <= -1] = np.nan
            pos = np.isfinite(vals)
            vals[pos] = np.log1p(vals[pos])

        transformed[valid] = vals
        out[col] = transformed
    return out


def transform_params_to_rows(params: FeatureTransformParams, enrolled_user: int) -> List[dict]:
    """Flatten params to CSV-friendly rows."""
    rows = []
    cols = params.feature_columns or tuple(sorted(set(params.clip_low) | set(params.clip_high)))
    for feature in cols:
        rows.append(
            {
                "enrolled_user": int(enrolled_user),
                "feature": feature,
                "clip_low": params.clip_low.get(feature),
                "clip_high": params.clip_high.get(feature),
                "use_log1p": bool(params.use_log1p),
            }
        )
    return rows
