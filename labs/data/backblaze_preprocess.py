"""Preprocess a Backblaze Drive Stats CSV into a vendor-friendly slice.

Run this ONCE locally on the 1GB Drive file (CSV or ZIP). It writes a small
~5MB CSV that the lab loaders read directly. Commit the output CSV into
labs/data/ alongside this script.

Usage
-----
    python backblaze_preprocess.py <input_file> <output_csv>

Examples
--------
    # If your input is a CSV
    python backblaze_preprocess.py backblaze_2023_Q1.csv labs/data/backblaze_subset.csv

    # If your input is a ZIP of daily CSVs (the canonical Backblaze release)
    python backblaze_preprocess.py backblaze_2023_Q1.zip labs/data/backblaze_subset.csv

What it does
------------
1. Reads all rows (handles either a single CSV or a ZIP of daily CSVs).
2. Filters to the most-prevalent drive model in the file (a single model
   keeps the population homogeneous for the labs).
3. Keeps only these columns:
       date, serial_number, model, capacity_bytes, failure
       smart_5_raw   (Reallocated Sectors Count)
       smart_187_raw (Reported Uncorrectable Errors)
       smart_188_raw (Command Timeout)
       smart_197_raw (Current Pending Sector Count)
       smart_198_raw (Offline Uncorrectable)
       smart_199_raw (UDMA CRC Error Count)
4. Keeps only drives with at least 30 days of observations AND optionally
   all observed failures + a sample of survivors so the failure rate in the
   subset is high enough for the labs (~5%).
5. Writes a long-format CSV (one row per drive-day) sorted by serial,date.

If the input has different SMART column names (older Backblaze schemas vary),
edit SMART_COLS below.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

SMART_COLS = [
    "smart_5_raw",     # Reallocated sectors
    "smart_187_raw",   # Reported uncorrectable errors
    "smart_188_raw",   # Command timeout
    "smart_197_raw",   # Current pending sector count
    "smart_198_raw",   # Offline uncorrectable
    "smart_199_raw",   # UDMA CRC error count
]

KEEP_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"] + SMART_COLS


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        frames = []
        with zipfile.ZipFile(path) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            print(f"  Found {len(csv_names)} CSV files inside the ZIP")
            for name in csv_names:
                with z.open(name) as f:
                    frames.append(pd.read_csv(f, low_memory=False))
        return pd.concat(frames, ignore_index=True)
    return pd.read_csv(path, low_memory=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--min-days", type=int, default=30,
                    help="Min observed days per drive to keep (default 30)")
    ap.add_argument("--survivor-sample", type=int, default=1000,
                    help="Cap on survivor drives to retain (default 1000)")
    args = ap.parse_args()

    print(f"Reading {args.input_path} ...")
    df = read_input(args.input_path)
    print(f"  Raw shape: {df.shape}")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns: {missing}")
        print(f"  Available columns: {sorted(df.columns)[:20]} ...")
        sys.exit(1)

    df = df[KEEP_COLS].copy()

    # Pick the most prevalent drive model.
    top_model = df["model"].value_counts().head(5)
    print(f"  Top 5 models by row count:\n{top_model}")
    target_model = top_model.index[0]
    print(f"  Selected model: {target_model}")
    df = df[df["model"] == target_model].copy()

    # Filter to drives with enough history.
    days_per_drive = df.groupby("serial_number").size()
    keep_drives = days_per_drive[days_per_drive >= args.min_days].index
    df = df[df["serial_number"].isin(keep_drives)].copy()
    print(f"  After min-days filter: {df.shape}, {df['serial_number'].nunique()} drives")

    # Keep all drives that ever failed + a sample of survivors.
    failed_drives = df[df["failure"] == 1]["serial_number"].unique()
    survivor_drives = [d for d in df["serial_number"].unique() if d not in failed_drives]
    if len(survivor_drives) > args.survivor_sample:
        import numpy as np
        rng = np.random.default_rng(0)
        survivor_drives = rng.choice(survivor_drives, size=args.survivor_sample, replace=False).tolist()
    keep = list(failed_drives) + list(survivor_drives)
    df = df[df["serial_number"].isin(keep)].copy()
    print(f"  After failure+survivor sampling: {df.shape}")
    print(f"    {len(failed_drives)} failed drives kept")
    print(f"    {len(survivor_drives)} survivor drives kept")
    print(f"    Per-drive failure rate: {len(failed_drives) / df['serial_number'].nunique():.3%}")

    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    size_mb = args.output_csv.stat().st_size / (1024 * 1024)
    print(f"  Wrote {args.output_csv}  ({size_mb:.1f} MB, {len(df)} rows)")


if __name__ == "__main__":
    main()
