"""Preprocess a Backblaze Drive Stats CSV into a vendor-friendly subset.

This is the script used to produce `labs/data/backblaze_subset.csv` from the
Backblaze Drive Stats public release. The full release is ~1 GB per quarter and
not vendored; the small subset is.

Source
------
    Backblaze Drive Stats (per-day SMART telemetry per drive):
        https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
        Each quarterly ZIP contains per-day CSVs with ~150K rows/day across
        ~95 columns (date, serial_number, model, capacity_bytes, failure,
        and ~90 SMART metrics in `normalized` + `raw` form).
    License: CC BY 4.0.

Usage
-----
    # Single CSV (e.g. one day or pre-concatenated month):
    python backblaze_preprocess.py harddrive.csv backblaze_subset.csv

    # ZIP of daily CSVs (canonical Backblaze quarterly release):
    python backblaze_preprocess.py 2016_Q1.zip backblaze_subset.csv

What it does
------------
    1. Reads the input (single CSV or ZIP-of-daily-CSVs).
    2. Keeps only columns needed for Labs 8B (DTR) and 10B (mediation):
           date, serial_number, model, failure
           smart_5_raw    (Reallocated Sector Count -- early-warning indicator)
           smart_187_raw  (Reported Uncorrectable Errors)
           smart_188_raw  (Command Timeout)
           smart_197_raw  (Current Pending Sector Count -- the mediator)
           smart_198_raw  (Offline Uncorrectable)
           smart_199_raw  (UDMA CRC Error Count)
       `capacity_bytes` is intentionally dropped: Backblaze stores it as a
       large integer that pandas occasionally parses as a subnormal float;
       none of the labs need it.
    3. Filters to the single most-prevalent drive model in the file (ST4000DM000
       in every Backblaze release we have seen). A single model keeps the
       population homogeneous so confounding by drive family does not dominate.
    4. Drops duplicate (date, serial_number) rows. Backblaze occasionally
       publishes the same drive-day twice; removing the duplicates avoids
       inflating sample sizes.
    5. Keeps ALL drives that fail in the window regardless of history length
       (failures are the rare-event signal we cannot afford to lose).
       Survivors are filtered to those with >= MIN_DAYS of observations and
       then random-sampled down to SURVIVOR_SAMPLE so the per-drive failure
       rate is ~3-5% (enough positive examples for the labs).
    6. Writes a long-format CSV (one row per drive-day) sorted by
       (serial_number, date). Output is ~2 MB for ~1500 drives.

The vendored `backblaze_subset.csv` in this repo was produced from a multi-day
slice of the 2016 Q1 release with MIN_DAYS=5, SURVIVOR_SAMPLE=1500, yielding
1576 drives (76 failed + 1500 survivors), 42050 drive-day rows, 4.8% drive-
level failure rate.
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SMART_COLS = [
    "smart_5_raw",     # Reallocated sectors -- early-warning indicator
    "smart_187_raw",   # Reported uncorrectable errors
    "smart_188_raw",   # Command timeout
    "smart_197_raw",   # Current pending sector count -- the mediator
    "smart_198_raw",   # Offline uncorrectable
    "smart_199_raw",   # UDMA CRC error count
]

KEEP_COLS = ["date", "serial_number", "model", "failure"] + SMART_COLS


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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_path", type=Path)
    ap.add_argument("output_csv", type=Path)
    ap.add_argument("--min-days", type=int, default=5,
                    help="Min observed days per SURVIVOR drive to keep (default 5). "
                         "Failed drives are kept regardless of history length.")
    ap.add_argument("--survivor-sample", type=int, default=1500,
                    help="Cap on never-failed drives to retain (default 1500)")
    ap.add_argument("--target-model", type=str, default=None,
                    help="Restrict to a specific drive model. Default: most-prevalent.")
    args = ap.parse_args()

    print(f"Reading {args.input_path} ...")
    df = read_input(args.input_path)
    print(f"  Raw shape: {df.shape}")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns: {missing}")
        print(f"  Available columns: {sorted(df.columns)[:30]} ...")
        sys.exit(1)

    fail_per_model = df.groupby("model")["failure"].sum().sort_values(ascending=False)
    print(f"\n  Failures per model (top 8):\n{fail_per_model.head(8)}")

    target = args.target_model or fail_per_model.index[0]
    print(f"\n  Selected model: {target}")

    df = df[KEEP_COLS].copy()
    df = df[df["model"] == target].copy()
    df = df.drop_duplicates(subset=["date", "serial_number"]).reset_index(drop=True)
    print(f"  After model filter + dedupe: {df.shape}")
    print(f"  Failures in {target}: {int(df['failure'].sum())}")

    failed_drives = set(df[df["failure"] == 1]["serial_number"].unique())
    print(f"  Total failed drives in {target}: {len(failed_drives)}")

    days_per_drive = df.groupby("serial_number").size()
    survivor_candidates = [
        d for d in df["serial_number"].unique()
        if d not in failed_drives and days_per_drive[d] >= args.min_days
    ]
    if len(survivor_candidates) > args.survivor_sample:
        rng = np.random.default_rng(0)
        survivor_drives = rng.choice(
            survivor_candidates, size=args.survivor_sample, replace=False
        ).tolist()
    else:
        survivor_drives = survivor_candidates

    keep = list(failed_drives) + list(survivor_drives)
    df = df[df["serial_number"].isin(keep)].copy()
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)

    n_drives = df["serial_number"].nunique()
    print(f"\n  Final shape: {df.shape}")
    print(f"    {len(failed_drives)} failed drives (all of them)")
    print(f"    {len(survivor_drives)} survivor drives")
    print(f"    Drive-level failure rate: {len(failed_drives)/max(n_drives,1):.3%}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    size_mb = args.output_csv.stat().st_size / (1024 * 1024)
    print(f"\n  Wrote {args.output_csv}  ({size_mb:.2f} MB, {len(df)} rows)")


if __name__ == "__main__":
    main()
