"""LFP lithium-ion battery dataset loader for Labs 4B, 6B, 7B.

Source: Pre-processed from the per-cell voltage arrays published in
        Mattia, P. (2021), Statistical learning for accurate and interpretable
        battery lifetime prediction, J. Electrochem. Soc., a follow-up to
        Severson, K. A. et al. (2019), Data-driven prediction of battery cycle life
        before capacity degradation, Nature Energy 4, 383-391.
        Upstream repo: https://github.com/petermattia/revisit-severson-et-al (MIT)
        Original data: data.matr.io/1/projects/5c48dd2bc625d700019f3204

The upstream files are per-cell voltage-vs-capacity arrays of shape
(1000 voltage positions x 99 cycles). This module ships two pre-processed
tables suitable for cell-level and per-cycle analyses:

    lfp_cell_summary.csv  — one row per cell (124 cells x 7 cols)
    lfp_cell_cycle.csv    — one row per (cell, cycle) (12,276 rows x 6 cols)

Columns in `lfp_cell_summary.csv`:
    cell_id            cell identifier (cell1, cell2, ...)
    batch              {train, test1, test2}  -- collection batch, proxy for
                       fast-charge-protocol-family + collection-period cohort
    cycle_life         cycles-to-EOL (target outcome for the Severson labs)
    log_var_deltaQ     log of variance of Q(V)_cyc100 - Q(V)_cyc10 across the
                       1000 voltage positions; the dominant Severson feature
    max_cap_cyc10      peak discharge capacity in cycle 10
    max_cap_cyc100     peak discharge capacity in cycle 100
    fade_cyc10_to_100  max_cap_cyc100 - max_cap_cyc10

Columns in `lfp_cell_cycle.csv`:
    cell_id, batch, cycle (2..100), max_cap, mean_cap, cycle_life

Public API:

    from lfp_prep import load_lfp
    df = load_lfp(chapter=4)   # Lab 4B: cell-level summary with batch as cohort
    df = load_lfp(chapter=6)   # Lab 6B: cell-level summary for CATE
    df = load_lfp(chapter=7)   # Lab 7B: per-cycle trajectory for time-varying

The CSVs are committed in `labs/data/` so the labs run with no external download.
"""
from pathlib import Path
import pandas as pd

_DATA = Path(__file__).parent
_SUMMARY = _DATA / "lfp_cell_summary.csv"
_CYCLE   = _DATA / "lfp_cell_cycle.csv"


def _load_summary() -> pd.DataFrame:
    if not _SUMMARY.exists():
        raise FileNotFoundError(f"LFP summary CSV not found at {_SUMMARY}.")
    return pd.read_csv(_SUMMARY)


def _load_cycle() -> pd.DataFrame:
    if not _CYCLE.exists():
        raise FileNotFoundError(f"LFP per-cycle CSV not found at {_CYCLE}.")
    return pd.read_csv(_CYCLE)


def load_lfp(chapter: int) -> pd.DataFrame:
    """Return a chapter-specific LFP slice."""
    if chapter == 4:
        return _slice_chapter4()
    if chapter == 6:
        return _slice_chapter6()
    if chapter == 7:
        return _slice_chapter7()
    raise ValueError(
        f"No LFP slice defined for chapter {chapter}. Currently implemented: 4, 6, 7."
    )


def _slice_chapter4() -> pd.DataFrame:
    """Ch 4 - IV / DID: cell-level summary with batch as cohort.

    The three batches in Severson 2019 were collected over different time periods
    with overlapping but distinct fast-charge-protocol families. We treat batch
    membership as the natural treatment cohort for a DID-style analysis: did the
    protocol revisions across batches change the mean cycle life, after
    controlling for the early-cycle log_var_deltaQ feature?
    """
    return _load_summary().copy()


def _slice_chapter6() -> pd.DataFrame:
    """Ch 6 - CATE: same cell-level summary, used for heterogeneity by batch.

    The CATE question: does the marginal effect of the log_var_deltaQ feature
    on predicted cycle life vary by batch (chemistry / collection period)?
    """
    return _load_summary().copy()


def _slice_chapter7() -> pd.DataFrame:
    """Ch 7 - time-varying: per-cycle trajectory with cell_id and cycle_life.

    For each cell, 99 rows of (cycle, max_cap, mean_cap). Lab 7B treats each
    cycle as a time step and asks how the per-cycle capacity-fade trajectory
    relates to the cell's eventual cycle_life.
    """
    return _load_cycle().copy()
