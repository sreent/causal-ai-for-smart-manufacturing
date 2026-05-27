"""Preprocess Severson 2019 LFP battery data into vendor-friendly subsets.

This is the script used to produce `labs/data/lfp_cell_summary.csv` and
`labs/data/lfp_cell_cycle.csv` from the per-cell pickle files published by
Peter Mattia's reanalysis of Severson 2019. The upstream pickles are ~1.5 GB
total and not vendored; the small CSVs are.

Source
------
    Severson, K. A. et al. (2019), "Data-driven prediction of battery cycle
    life before capacity degradation", Nature Energy 4, 383-391.
    Re-published with cleaner Python tooling by Peter Mattia (2021):
        https://github.com/petermattia/revisit-severson-et-al  (MIT)
    Original raw data:
        https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

The three batches in Severson 2019 correspond to three collection periods on
the same LFP/graphite chemistry, each running a different (overlapping) set of
fast-charging protocols. Total: 124 cells (b1=41, b2=43, b3=40). Severson uses
the batch split as a train/test split; we name the batches accordingly:

    b1 -> train  (41 cells)
    b2 -> test1  (43 cells)
    b3 -> test2  (40 cells)

Mattia pickle structure
-----------------------
Each `batch{N}.pkl` is a dict keyed by cell-id (e.g. "b1c0"). Each cell entry:

    {
      "cycle_life": float,        # cycles-to-EOL (capacity drop to 80%)
      "summary":  {"QD": [...], ...},     # per-cycle scalars
      "cycles":   {"10": {"Qdlin": [1000-vec], "Qd": [...], "V": [...], ...},
                   "11": {...}, ..., "100": {...}},
    }

`Qdlin` is the discharge capacity curve re-sampled to 1000 linearly-spaced
voltage positions; this is the Severson canonical feature input.

Outputs
-------
    lfp_cell_summary.csv  (124 rows x 7 cols)
        cell_id            str   batch + 1-indexed cell number (e.g. "train_cell1")
        batch              str   {train, test1, test2}
        cycle_life         float cycles-to-80%-capacity (the prediction target)
        log_var_deltaQ     float log10(var(Qdlin_cyc100 - Qdlin_cyc10)) across
                                 the 1000 voltage positions -- the dominant
                                 Severson feature
        max_cap_cyc10      float max discharge capacity at cycle 10
        max_cap_cyc100     float max discharge capacity at cycle 100
        fade_cyc10_to_100  float max_cap_cyc100 - max_cap_cyc10

    lfp_cell_cycle.csv  (~12,276 rows x 6 cols)
        cell_id, batch, cycle (2..100), max_cap, mean_cap, cycle_life

Why these features
------------------
Lab 4B (DID) and 6B (CATE) want a cohort variable (`batch`) and a strong
covariate (`log_var_deltaQ`) that predicts the outcome (`cycle_life`). Severson
2019 showed `log_var_deltaQ` alone gives MAPE < 10% on `cycle_life`, so it is
the right feature for revealing or controlling heterogeneity.

Lab 7B (time-varying) wants per-cycle observations of a covariate that
co-evolves with the eventual outcome. `max_cap` and `mean_cap` per cycle form
a 99-step trajectory that admits time-varying-treatment-effect framings.

Usage
-----
    # After downloading Mattia's three batch pickles to ./raw/:
    python lfp_preprocess.py raw/batch1.pkl raw/batch2.pkl raw/batch3.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BATCH_NAMES = {1: "train", 2: "test1", 3: "test2"}
SUMMARY_CYCLE_START = 10
SUMMARY_CYCLE_END = 100
CYCLE_TRAJECTORY_RANGE = range(2, 101)  # cycles 2..100 inclusive


def _q_at_cycle(cell: dict, cycle: int) -> np.ndarray:
    """Return the 1000-point Qdlin curve at the given cycle."""
    return np.asarray(cell["cycles"][str(cycle)]["Qdlin"], dtype=float)


def _qd_at_cycle(cell: dict, cycle: int) -> np.ndarray:
    """Return the raw discharge-capacity samples (Qd) at the given cycle."""
    return np.asarray(cell["cycles"][str(cycle)]["Qd"], dtype=float)


def _summary_row(cell_id: str, batch_name: str, cell: dict) -> dict:
    q10  = _q_at_cycle(cell, SUMMARY_CYCLE_START)
    q100 = _q_at_cycle(cell, SUMMARY_CYCLE_END)
    dq   = q100 - q10
    qd10  = _qd_at_cycle(cell, SUMMARY_CYCLE_START)
    qd100 = _qd_at_cycle(cell, SUMMARY_CYCLE_END)
    max_cap_10  = float(np.max(qd10))
    max_cap_100 = float(np.max(qd100))
    return {
        "cell_id": cell_id,
        "batch": batch_name,
        "cycle_life": float(cell["cycle_life"]),
        "log_var_deltaQ": float(np.log10(np.var(dq))),
        "max_cap_cyc10": max_cap_10,
        "max_cap_cyc100": max_cap_100,
        "fade_cyc10_to_100": max_cap_100 - max_cap_10,
    }


def _cycle_rows(cell_id: str, batch_name: str, cell: dict) -> list[dict]:
    cycle_life = float(cell["cycle_life"])
    rows = []
    for cyc in CYCLE_TRAJECTORY_RANGE:
        if str(cyc) not in cell["cycles"]:
            continue  # rare: a cell may be missing an intermediate cycle
        qd = _qd_at_cycle(cell, cyc)
        rows.append({
            "cell_id": cell_id,
            "batch": batch_name,
            "cycle": cyc,
            "max_cap": float(np.max(qd)),
            "mean_cap": float(np.mean(qd)),
            "cycle_life": cycle_life,
        })
    return rows


def _process_batch(pickle_path: Path, batch_index: int) -> tuple[list[dict], list[dict]]:
    batch_name = BATCH_NAMES[batch_index]
    print(f"  Loading {pickle_path} as batch '{batch_name}' ...")
    with pickle_path.open("rb") as f:
        batch_dict = pickle.load(f)

    summary_rows: list[dict] = []
    cycle_rows: list[dict] = []
    for idx, (raw_key, cell) in enumerate(sorted(batch_dict.items())):
        cell_id = f"{batch_name}_cell{idx + 1}"
        summary_rows.append(_summary_row(cell_id, batch_name, cell))
        cycle_rows.extend(_cycle_rows(cell_id, batch_name, cell))
    print(f"    {len(summary_rows)} cells, {len(cycle_rows)} cycle rows")
    return summary_rows, cycle_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("batch1", type=Path, help="batch1.pkl (b1 -> train)")
    ap.add_argument("batch2", type=Path, help="batch2.pkl (b2 -> test1)")
    ap.add_argument("batch3", type=Path, help="batch3.pkl (b3 -> test2)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent,
                    help="Destination directory for the two CSVs (default: this file's dir)")
    args = ap.parse_args()

    all_summary: list[dict] = []
    all_cycle: list[dict] = []
    for idx, path in enumerate([args.batch1, args.batch2, args.batch3], start=1):
        s, c = _process_batch(path, idx)
        all_summary.extend(s)
        all_cycle.extend(c)

    summary_df = pd.DataFrame(all_summary)
    cycle_df = pd.DataFrame(all_cycle)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "lfp_cell_summary.csv"
    cycle_path = args.out_dir / "lfp_cell_cycle.csv"
    summary_df.to_csv(summary_path, index=False)
    cycle_df.to_csv(cycle_path, index=False)

    print(f"\n  Wrote {summary_path}  ({summary_df.shape[0]} rows x {summary_df.shape[1]} cols)")
    print(f"  Wrote {cycle_path}    ({cycle_df.shape[0]} rows x {cycle_df.shape[1]} cols)")
    print(f"  Batches: {summary_df['batch'].value_counts().to_dict()}")
    print(f"  cycle_life range: [{summary_df['cycle_life'].min():.0f}, "
          f"{summary_df['cycle_life'].max():.0f}]")


if __name__ == "__main__":
    main()
