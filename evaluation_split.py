"""Session / segment / time-block train-val-test splits (leakage-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class UserSplitAssignment:
    user_id: int
    split_unit: str  # "session" | "segment" | "time_block"
    train: List[str]
    validation: List[str]
    test: List[str]
    fallback_reason: Optional[str] = None

    def groups_for(self, split: str) -> List[str]:
        if split == "train":
            return list(self.train)
        if split in ("validation", "val"):
            return list(self.validation)
        if split == "test":
            return list(self.test)
        raise ValueError(f"Unknown split: {split}")

    def assert_disjoint(self) -> None:
        t, v, te = set(self.train), set(self.validation), set(self.test)
        if not t.isdisjoint(v):
            raise AssertionError(f"user {self.user_id}: train ∩ validation != ∅")
        if not t.isdisjoint(te):
            raise AssertionError(f"user {self.user_id}: train ∩ test != ∅")
        if not v.isdisjoint(te):
            raise AssertionError(f"user {self.user_id}: validation ∩ test != ∅")

    def to_rows(self) -> List[dict]:
        rows = []
        for split, groups in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            for gid in groups:
                rows.append(
                    {
                        "user_id": int(self.user_id),
                        "group_id": gid,
                        "split": split,
                        "split_unit": self.split_unit,
                        "fallback_reason": self.fallback_reason or "",
                    }
                )
        return rows


@dataclass
class SplitAssignments:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    by_user: Dict[int, UserSplitAssignment] = field(default_factory=dict)

    def assert_all_disjoint(self) -> None:
        for asg in self.by_user.values():
            asg.assert_disjoint()

    def to_rows(self) -> List[dict]:
        rows = []
        for asg in sorted(self.by_user.values(), key=lambda a: a.user_id):
            rows.extend(asg.to_rows())
        return rows


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    s = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {s}")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("ratios must be non-negative")


def allocate_split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """
    Integer allocation for n units under ratios, requiring each split >= 1 when n >= 3.
    """
    validate_ratios(train_ratio, val_ratio, test_ratio)
    if n < 3:
        raise ValueError(f"Need at least 3 units for train/val/test, got n={n}")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Fix rounding so each gets at least 1
    counts = {"train": n_train, "validation": n_val, "test": n_test}
    for name in SPLIT_NAMES:
        if counts[name] < 1:
            donor = max(
                (k for k in SPLIT_NAMES if k != name and counts[k] > 1),
                key=lambda k: counts[k],
                default=None,
            )
            if donor is None:
                raise ValueError(f"Cannot allocate train/val/test for n={n}")
            counts[donor] -= 1
            counts[name] += 1

    if counts["train"] + counts["validation"] + counts["test"] != n:
        raise ValueError(f"Allocation mismatch for n={n}: {counts}")
    return counts["train"], counts["validation"], counts["test"]


def _shuffle_ids(group_ids: Sequence[str], seed: int, user_id: int) -> List[str]:
    rng = np.random.default_rng(int(seed) + int(user_id) * 1_000_003)
    ids = list(group_ids)
    rng.shuffle(ids)
    return ids


def split_group_ids(
    group_ids: Sequence[str],
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    user_id: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    ids = _shuffle_ids(group_ids, seed=seed, user_id=user_id)
    n_train, n_val, n_test = allocate_split_counts(len(ids), train_ratio, val_ratio, test_ratio)
    train = ids[:n_train]
    validation = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val : n_train + n_val + n_test]
    return train, validation, test


def make_user_split(
    user_id: int,
    session_ids: Sequence[str],
    segment_ids: Sequence[str],
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    min_units: int = 3,
) -> UserSplitAssignment:
    """
    Prefer session → segment → synthetic time_block ids.

    ``time_block`` fallback creates three synthetic ids when neither sessions nor
    segments provide ``min_units`` groups. Callers must map windows onto those
    contiguous blocks separately.
    """
    sessions = [str(x) for x in session_ids if str(x).strip()]
    segments = [str(x) for x in segment_ids if str(x).strip()]

    if len(sessions) >= min_units:
        train, validation, test = split_group_ids(
            sessions,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            user_id=user_id,
        )
        asg = UserSplitAssignment(
            user_id=int(user_id),
            split_unit="session",
            train=train,
            validation=validation,
            test=test,
        )
        asg.assert_disjoint()
        return asg

    if len(segments) >= min_units:
        train, validation, test = split_group_ids(
            segments,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            user_id=user_id,
        )
        asg = UserSplitAssignment(
            user_id=int(user_id),
            split_unit="segment",
            train=train,
            validation=validation,
            test=test,
            fallback_reason="insufficient_sessions",
        )
        asg.assert_disjoint()
        return asg

    # Contiguous time-block fallback: always 3 synthetic blocks (60/20/20 by index bands)
    block_ids = ["time_block_train", "time_block_validation", "time_block_test"]
    asg = UserSplitAssignment(
        user_id=int(user_id),
        split_unit="time_block",
        train=[block_ids[0]],
        validation=[block_ids[1]],
        test=[block_ids[2]],
        fallback_reason="insufficient_sessions_and_segments",
    )
    asg.assert_disjoint()
    return asg


def assign_time_block_labels(
    n_windows: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> List[str]:
    """
    Contiguous index bands over ordered windows (no shuffle) for time_block fallback.
    Windows are assumed sorted by time; each band is an independent block.
    """
    if n_windows <= 0:
        return []
    validate_ratios(train_ratio, val_ratio, test_ratio)
    n_train = max(1, int(n_windows * train_ratio))
    n_val = max(1, int(n_windows * val_ratio))
    if n_train + n_val >= n_windows:
        # Shrink so test gets at least one when possible
        if n_windows >= 3:
            n_test = 1
            rem = n_windows - 1
            n_train = max(1, int(round(rem * train_ratio / (train_ratio + val_ratio))))
            n_val = rem - n_train
            if n_val < 1:
                n_val = 1
                n_train = rem - 1
        else:
            # Degenerate: put all in train (caller should avoid scoring)
            return ["time_block_train"] * n_windows
    else:
        n_test = n_windows - n_train - n_val

    labels = (
        ["time_block_train"] * n_train
        + ["time_block_validation"] * n_val
        + ["time_block_test"] * n_test
    )
    if len(labels) != n_windows:
        raise RuntimeError("time_block label length mismatch")
    return labels


def build_split_assignments(
    user_units: Mapping[int, Mapping[str, Sequence[str]]],
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SplitAssignments:
    """
    ``user_units[user_id] = {"session_ids": [...], "segment_ids": [...]}``.
    """
    validate_ratios(train_ratio, val_ratio, test_ratio)
    out = SplitAssignments(
        seed=int(seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )
    for user_id in sorted(user_units.keys()):
        units = user_units[user_id]
        out.by_user[int(user_id)] = make_user_split(
            int(user_id),
            session_ids=units.get("session_ids") or [],
            segment_ids=units.get("segment_ids") or [],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    out.assert_all_disjoint()
    return out
