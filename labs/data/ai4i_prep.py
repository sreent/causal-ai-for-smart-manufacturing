"""AI4I 2020 Predictive Maintenance dataset loader for Lab 2B.

Source: Matzka (2020), "Explainable AI for Predictive Maintenance Applications",
        2020 Third Intl. Conf. on Artificial Intelligence for Industries (AI4I),
        69-74. CC BY 4.0. UCI ML Repository, dataset 601.

10,000 synthetic-but-physically-grounded samples from a milling-machine setup.
Per row: 5 numeric process variables, a Type categorical (L/M/H product variant),
a binary machine_failure label, and 5 specific failure-mode flags.

Columns the loader returns (renamed for ergonomics):
  type             {L, M, H} product variant
  air_temp_K       Air temperature (K)
  process_temp_K   Process temperature (K)
  rot_speed_rpm    Tool rotational speed (rpm)
  torque_Nm        Torque (Nm)
  tool_wear_min    Cumulative tool wear (min)
  failure          Binary machine_failure (1 = at least one failure mode triggered)
  twf, hdf, pwf, osf, rnf  Five specific failure-mode flags

The dataset is committed as `labs/data/ai4i2020.csv` (~520 KB, CC BY 4.0).
"""
from pathlib import Path

import numpy as np
import pandas as pd


_CSV = Path(__file__).parent / "ai4i2020.csv"

_RENAME = {
    "Type":                     "type",
    "Air temperature [K]":      "air_temp_K",
    "Process temperature [K]":  "process_temp_K",
    "Rotational speed [rpm]":   "rot_speed_rpm",
    "Torque [Nm]":              "torque_Nm",
    "Tool wear [min]":          "tool_wear_min",
    "Machine failure":          "failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf",
}


def _fetch_raw() -> pd.DataFrame:
    if not _CSV.exists():
        raise FileNotFoundError(
            f"AI4I CSV not found at {_CSV}. Expected to be committed in the repo."
        )
    df = pd.read_csv(_CSV)
    df = df.rename(columns=_RENAME)
    return df.drop(columns=[c for c in ("UDI", "Product ID") if c in df.columns])


def load_ai4i(chapter: int) -> pd.DataFrame:
    """Return a chapter-specific AI4I slice."""
    df = _fetch_raw()
    if chapter == 2:
        return _slice_chapter2(df)
    raise ValueError(
        f"No AI4I slice defined for chapter {chapter}. Currently implemented: 2."
    )


def _slice_chapter2(df: pd.DataFrame) -> pd.DataFrame:
    """Ch 2 - back-door on a single-machine SCM.

    Returns the cleaned AI4I frame with `type` one-hot encoded as `type_M` and
    `type_H` (L is the reference). The lab itself chooses the treatment column
    and binarises it; this slice just makes the columns regression-ready.
    """
    out = df.copy()
    out["type_M"] = (out["type"] == "M").astype(int)
    out["type_H"] = (out["type"] == "H").astype(int)
    out = out.drop(columns=["type"])
    return out
