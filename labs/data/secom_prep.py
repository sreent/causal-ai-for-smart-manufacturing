"""SECOM dataset loader for the Lab-B series.

SECOM (SEmiconductor MAnufacturing) - McCann & Johnston, UCI ML Repository 2008.
1567 wafers, 590 numeric sensor features, binary yield (+1 fail / -1 pass),
timestamps spanning Jul-Oct 2008. Reference: https://archive.ics.uci.edu/dataset/179

The loader fetches `secom.zip` (the original UCI distribution: secom.data +
secom_labels.data) from a Google Drive mirror on first call, falls back to the
official `ucimlrepo` package if Drive is unreachable, caches the parsed frame
as a Parquet file under `labs/data/cache/secom_raw.parquet`, and returns
chapter-specific pre-cleaned slices. The cleaning decisions (missingness
threshold, feature selection rule, period derivation) live in this module so
the lab notebooks can stay focused on the chapter's causal-inference concept
instead of data engineering.

Public API:

    from secom_prep import load_secom
    df = load_secom(chapter=1)   # returns a tidy DataFrame for Lab 1B

Implemented slices: chapter 1.
Planned: chapters 3, 5, 9, 13.
"""
import io
import re
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_CACHE = Path(__file__).parent / "cache"
_RAW_CACHE = _CACHE / "secom_raw.pkl.gz"

# Primary source: vendored zip in the repo (committed at labs/data/secom.zip).
_LOCAL_ZIP = Path(__file__).parent / "secom.zip"

# Secondary mirror: Google Drive (for environments without the repo clone).
_DRIVE_FILE_ID = "1zfdu-F_p2saA3emthFV8EO1kNLdkDy9t"
_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={_DRIVE_FILE_ID}"

# Tertiary fallback: official UCI repository via ucimlrepo.
_UCI_ID = 179


def _parse_secom_zip(zip_bytes):
    """Parse the UCI secom.zip payload into a tidy DataFrame.

    Format inside the zip:
      - secom.data         whitespace-separated, 590 numeric columns, NaN as "NaN", no header.
      - secom_labels.data  two columns: yield label (+1/-1) and a *quoted* timestamp
                           string in the form `"DD/MM/YYYY HH:MM:SS"`.
      - secom.names        metadata blurb, ignored.
    """
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))

    data_name = next(
        (n for n in z.namelist()
         if n.lower().endswith("secom.data") or n.lower().endswith("secom_data")),
        None,
    )
    labels_name = next(
        (n for n in z.namelist()
         if "label" in n.lower() and n.lower().endswith(".data")),
        None,
    )
    if data_name is None or labels_name is None:
        raise ValueError(
            f"Could not find secom.data and secom_labels.data inside the zip. "
            f"Files present: {z.namelist()}"
        )

    with z.open(data_name) as f:
        X = pd.read_csv(f, sep=r"\s+", header=None, na_values="NaN", engine="python")
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]

    # secom_labels.data has a quoted timestamp: `+1 "19/07/2008 11:55:00"`.
    # pandas's read_csv with a regex separator does not honour quotechar, so we
    # use a small regex to pull the label and the quoted ts directly.
    with z.open(labels_name) as f:
        text = f.read().decode("utf-8")
    matches = re.findall(r'^\s*(-?\d+)\s+"([^"]+)"\s*$', text, re.MULTILINE)
    if not matches:
        raise ValueError("Could not parse secom_labels.data — no rows matched the expected format.")
    labels = pd.DataFrame(matches, columns=["yield_raw", "ts"])
    labels["yield_raw"] = labels["yield_raw"].astype(int)
    ts = pd.to_datetime(labels["ts"], format="%d/%m/%Y %H:%M:%S")

    out = pd.DataFrame({
        "ts":         ts.reset_index(drop=True),
        "period":     ts.dt.to_period("M").astype(str).reset_index(drop=True),
        "yield_fail": (labels["yield_raw"] == 1).astype(int).reset_index(drop=True),
    })
    out = pd.concat([out, X.reset_index(drop=True)], axis=1)
    return out


def _fetch_from_local_repo():
    """Load secom.zip vendored alongside this module (the primary source)."""
    if not _LOCAL_ZIP.exists():
        raise FileNotFoundError(f"Local zip not present at {_LOCAL_ZIP}")
    return _parse_secom_zip(_LOCAL_ZIP.read_bytes())


def _fetch_from_drive():
    """Download secom.zip from the Drive mirror and parse it."""
    req = urllib.request.Request(_DRIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = r.read()
    return _parse_secom_zip(payload)


def _fetch_from_ucimlrepo():
    """Fallback: fetch SECOM via the official UCI Python package."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise ImportError(
            "ucimlrepo is required for the UCI fallback. "
            "Install with: pip install ucimlrepo"
        ) from e

    ds = fetch_ucirepo(id=_UCI_ID)
    X = ds.data.features.copy()
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]

    targets = ds.data.targets.copy()
    fail_col = next(c for c in targets.columns
                    if targets[c].dropna().isin([-1, 1]).all())
    ts_col = next((c for c in targets.columns
                   if c.lower() in {"time", "ts", "timestamp"}), None)

    if ts_col is None:
        warnings.warn(
            "Timestamps not surfaced by ucimlrepo; synthesising a uniform monthly index."
        )
        ts = pd.date_range("2008-07-19", periods=len(X), freq="2H")
    else:
        ts = pd.to_datetime(targets[ts_col])

    out = pd.DataFrame({
        "ts":         pd.Series(ts).reset_index(drop=True),
        "period":     pd.Series(ts).dt.to_period("M").astype(str).reset_index(drop=True),
        "yield_fail": (targets[fail_col].reset_index(drop=True) == 1).astype(int),
    })
    out = pd.concat([out, X.reset_index(drop=True)], axis=1)
    return out


def _fetch_raw():
    """Fetch SECOM and cache as Parquet.

    Tries in order:
      1. The vendored zip at `labs/data/secom.zip` (no network needed).
      2. The Google Drive mirror (network).
      3. The official UCI repo via ucimlrepo (network).
    """
    if _RAW_CACHE.exists():
        return pd.read_pickle(_RAW_CACHE)

    sources = [
        ("local repo zip", _fetch_from_local_repo),
        ("Google Drive mirror", _fetch_from_drive),
        ("UCI via ucimlrepo", _fetch_from_ucimlrepo),
    ]
    last_err = None
    out = None
    for name, fn in sources:
        try:
            out = fn()
            break
        except Exception as err:
            last_err = err
            warnings.warn(f"SECOM source '{name}' failed: {type(err).__name__}: {err}")
    if out is None:
        raise RuntimeError(
            f"All SECOM sources failed. Last error: {type(last_err).__name__}: {last_err}"
        )

    _CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(_RAW_CACHE)
    return out


def _select_sensors_by_yield_corr(df, k=5, missingness_thresh=0.10):
    """Pick `k` sensors with low missingness, ranked by |corr with yield_fail|.

    Deterministic given the raw data; reproducible across machines.
    """
    sensor_cols = [c for c in df.columns if c.startswith("S")]
    miss = df[sensor_cols].isna().mean()
    keep = miss[miss < missingness_thresh].index.tolist()

    var = df[keep].var()
    keep = var[var > 1e-10].index.tolist()

    y = df["yield_fail"].astype(float)
    corrs = df[keep].apply(lambda col: col.corr(y))
    return corrs.abs().sort_values(ascending=False).head(k).index.tolist()


def load_secom(chapter):
    """Return a focused, pre-cleaned DataFrame slice for a given chapter's Lab B.

    Each slice contains exactly the columns that lab needs - no further feature
    exploration or cleaning is required in the notebook.
    """
    df = _fetch_raw()
    if chapter == 1:
        return _slice_chapter1(df)
    raise ValueError(
        f"No SECOM slice defined for chapter {chapter}. "
        f"Currently implemented: 1. Planned: 3, 5, 9, 13."
    )


def _slice_chapter1(df):
    """Ch 1 - high-AUC trap: 5 top-correlated sensors + period stratifier + yield.

    Pedagogical point of this slice: a naive ML classifier on the five sensors
    will achieve nontrivial AUC and identify a top feature; back-door adjustment
    on `period` substantially shrinks or eliminates the apparent effects.
    """
    sensors = _select_sensors_by_yield_corr(df, k=5, missingness_thresh=0.10)
    out = df[["period"] + sensors + ["yield_fail"]].copy()
    for s in sensors:
        out[s] = out[s].fillna(out[s].median())
    # Drop periods with too few observations for stratification to be stable.
    period_counts = out["period"].value_counts()
    keep_periods = period_counts[period_counts >= 30].index
    out = out[out["period"].isin(keep_periods)].reset_index(drop=True)
    return out
