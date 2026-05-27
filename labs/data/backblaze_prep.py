"""Backblaze hard-drive failure dataset loader for Labs 8B and 10B.

Source: Backblaze Drive Stats (CC BY 4.0). The vendored subset
`labs/data/backblaze_subset.csv` is a 1576-drive, ~30-day slice of the
2016 Q1 release filtered to ST4000DM000 drives. See `labs/data/README.md`
and `labs/data/backblaze_preprocess.py` for the full preprocessing
pipeline.

Public API:

    from backblaze_prep import load_backblaze
    df = load_backblaze(chapter=8)    # Lab 8B: per-drive day-7/day-14 features
    df = load_backblaze(chapter=10)   # Lab 10B: per-drive X/M/Y summary

The loader also drops `smart_188_raw`. That column is mis-parsed by pandas
as subnormal float64 (the upstream values overflow int64-as-float
inference), and none of the labs need it.

Implemented slices: chapters 8 and 10.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).parent
_CSV = _DATA / "backblaze_subset.csv"

# SMART columns we trust (smart_188_raw is mis-parsed; drop it).
_SMART = [
    "smart_5_raw",     # Reallocated Sector Count - early-warning treatment
    "smart_187_raw",   # Reported Uncorrectable Errors
    "smart_197_raw",   # Current Pending Sector Count - the mediator for Lab 10B
    "smart_198_raw",   # Offline Uncorrectable
    "smart_199_raw",   # UDMA CRC Error Count
]


def _load_raw() -> pd.DataFrame:
    if not _CSV.exists():
        raise FileNotFoundError(
            f"Backblaze subset CSV not found at {_CSV}.\n"
            "See labs/data/README.md for the preprocessing recipe."
        )
    df = pd.read_csv(_CSV, parse_dates=["date"])
    df = df.drop(columns=["smart_188_raw"], errors="ignore")
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    df["day_index"] = df.groupby("serial_number").cumcount()
    return df


def load_backblaze(chapter: int, **kwargs) -> pd.DataFrame:
    if chapter == 8:
        return _slice_chapter8(**kwargs)
    if chapter == 10:
        return _slice_chapter10(**kwargs)
    raise ValueError(
        f"No Backblaze slice defined for chapter {chapter}. "
        "Currently implemented: 8, 10."
    )


def _slice_chapter8(stage1_day: int = 7, stage2_day: int = 14,
                    min_days: int = 7) -> pd.DataFrame:
    """Ch 8 - DTR: per-drive features at two decision stages.

    Each row is one drive that survived at least `min_days` of observation
    (default 7, matching stage1_day) — drives that fail too early have no
    stage-1 decision to make. Drives that fail between stage1_day and
    stage2_day are kept: stage-1 fitting uses them, stage-2 fitting filters
    them out (they are already terminal). State features at each stage are
    the cumulative max of each SMART column up to (but not including) the
    stage day:

        state1_smart_5_raw   = max(smart_5_raw)  over day_index in [0, stage1_day)
        state2_smart_5_raw   = max(smart_5_raw)  over day_index in [0, stage2_day)

    The cumulative-max representation captures the "ever-elevated" condition
    that drives are clinically diagnosed on, and matches how a maintenance
    engineer would inspect the trajectory.

    Outcome columns (Lab 8B uses these as reward inputs after defining
    synthetic actions):

        fail_day              0-indexed day of failure (-1 if no failure)
        failed_after_stage1   1 if fail_day >= stage1_day
        failed_after_stage2   1 if fail_day >= stage2_day
        failed_overall        1 if fail_day >= 0
    """
    df = _load_raw()
    days_observed = df.groupby("serial_number").size().rename("days_observed")

    fail_day = (
        df.loc[df["failure"] == 1]
          .set_index("serial_number")["day_index"]
          .rename("fail_day")
    )

    stage1 = (
        df[df["day_index"] < stage1_day]
          .groupby("serial_number")[_SMART]
          .max()
          .rename(columns={c: f"state1_{c}" for c in _SMART})
    )
    stage2 = (
        df[df["day_index"] < stage2_day]
          .groupby("serial_number")[_SMART]
          .max()
          .rename(columns={c: f"state2_{c}" for c in _SMART})
    )

    out = (
        pd.DataFrame(index=df["serial_number"].unique())
          .join(days_observed)
          .join(fail_day)
          .join(stage1)
          .join(stage2)
    )
    out["fail_day"] = out["fail_day"].fillna(-1).astype(int)
    out["failed_overall"] = (out["fail_day"] >= 0).astype(int)
    out["failed_after_stage1"] = (
        (out["fail_day"] >= stage1_day) & (out["fail_day"] >= 0)
    ).astype(int)
    out["failed_after_stage2"] = (
        (out["fail_day"] >= stage2_day) & (out["fail_day"] >= 0)
    ).astype(int)
    for c in [f"state1_{c}" for c in _SMART] + [f"state2_{c}" for c in _SMART]:
        out[c] = out[c].fillna(0.0)
    out.index.name = "serial_number"
    out = out.reset_index()
    # Keep drives that (a) have enough history for stage-1 state and (b)
    # didn't fail before the stage-1 decision could be made.
    out = out[
        (out["days_observed"] >= min_days)
        & ((out["fail_day"] == -1) | (out["fail_day"] >= stage1_day))
    ].reset_index(drop=True)
    return out


def _slice_chapter10() -> pd.DataFrame:
    """Ch 10 - Mediation: per-drive X / M / Y summary.

    Returns one row per drive (no filter on observation length, since
    mediation works on whatever the drive's lifetime was). Columns:

        serial_number     drive id
        days_observed     observation length in days
        smart_*_max       per-drive max of each SMART column (covariates)
        X                 1 if smart_5_raw  ever exceeded 0 (treatment)
        M                 1 if smart_197_raw ever exceeded 0 (mediator)
        Y                 1 if drive failed in the window (outcome)

    The thresholding at zero is Backblaze-standard: SMART_5 (Reallocated
    Sectors) and SMART_197 (Pending Sectors) are zero by design until the
    drive's firmware detects a bad sector. The first non-zero value is the
    diagnostic event Lab 10B's chain models.
    """
    df = _load_raw()
    g = df.groupby("serial_number")
    summary = pd.DataFrame({
        "days_observed": g.size(),
        "smart_5_max":   g["smart_5_raw"].max(),
        "smart_187_max": g["smart_187_raw"].max(),
        "smart_197_max": g["smart_197_raw"].max(),
        "smart_198_max": g["smart_198_raw"].max(),
        "smart_199_max": g["smart_199_raw"].max(),
        "Y":             g["failure"].max().astype(int),
    }).reset_index()
    summary["X"] = (summary["smart_5_max"] > 0).astype(int)
    summary["M"] = (summary["smart_197_max"] > 0).astype(int)
    return summary
