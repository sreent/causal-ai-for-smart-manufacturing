"""Tennessee Eastman dataset loader for Labs 11B and 12B.

The TE problem (Downs & Vogel 1993) is a canonical industrial-control simulator.
We use the `tep2py` Python wrapper around the Fortran reference code; the source
is vendored under `te_simulator/` and may be re-built with f2py for users who
want to re-simulate scenarios beyond the two committed CSVs.

Public API:

    from te_prep import load_te
    df = load_te("logged")     # Lab 11B behaviour-policy trajectory
    df = load_te("candidate")  # Lab 11B evaluation-policy trajectory

Each CSV contains 500 samples (at the 3-min TE sampling interval) of:
  - ts_min      : sample index
  - action_idv1 : the per-step value of IDV(1) (the action in the logged scenario)
  - XMEAS(1..41): TE process measurements
  - XMV(1..12)  : TE actuator setpoints

Reward / outcome definitions live in each lab (since the chapter-specific
estimand differs between 11B and 12B).
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent
_SCENARIOS = {
    "logged":    "te_logged.csv",
    "candidate": "te_candidate.csv",
}


def load_te(scenario: str) -> pd.DataFrame:
    """Return a pre-generated TE trajectory.

    Parameters
    ----------
    scenario : {"logged", "candidate"}
        - "logged"    : 500 samples with IDV(1) toggled Bernoulli(0.5) per step.
                        Used by Lab 11B as the behaviour-policy logged data.
        - "candidate" : 500 samples with IDV(1) = 0 always.
                        Used by Lab 11B as the ground-truth evaluation trajectory.
    """
    if scenario not in _SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: {list(_SCENARIOS)}"
        )
    path = _DATA_DIR / _SCENARIOS[scenario]
    if not path.exists():
        raise FileNotFoundError(
            f"TE scenario CSV not found at {path}. The file is committed in the "
            f"public repo; re-clone or re-run the simulator (see te_simulator/README.md)."
        )
    return pd.read_csv(path)


def simulate_te(idata: np.ndarray) -> pd.DataFrame:
    """Run the live TE simulator on a custom IDV pattern.

    Requires the vendored Fortran simulator to be built (gfortran + f2py). See
    te_simulator/README.md for build instructions. This function is for
    extension exercises; the pre-generated CSVs cover the canonical labs.

    Parameters
    ----------
    idata : ndarray of shape (n_steps, 20)
        Per-step values of the 20 IDV (disturbance) flags, 0 or 1.

    Returns
    -------
    DataFrame with one row per step and TE's 52 process columns.
    """
    sim_dir = _DATA_DIR / "te_simulator"
    import sys
    sys.path.insert(0, str(sim_dir))
    try:
        from tep2py import tep2py  # noqa: WPS433
    except ImportError as e:
        raise ImportError(
            "Could not import tep2py. Build the Fortran extension first:\n"
            f"  cd {sim_dir} && python -m numpy.f2py -c temain_mod.pyf temain_mod.f teprob.f -m temain_mod"
        ) from e
    tep = tep2py(idata)
    tep.simulate()
    return tep.process_data
