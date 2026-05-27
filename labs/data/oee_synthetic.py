"""Curated synthetic OEE log dataset.

OEE = Availability x Performance x Quality is the standard manufacturing
KPI; an OEE drop is the operational signal a plant lead investigates. The
question every root-cause review asks: *which of the three drivers caused
the drop?* That is a multi-mediator decomposition - exactly what Chapter
10's NDE/NIE machinery is for. No public real dataset exposes the
manipulable A x P x Q structure with a controllable intervention, so this
generator is curated synthetic.

The data-generating process is fully documented below so the capstone has
ground truth to validate against; that is the point of the synthetic
choice. Any student should be able to read the SCM here, derive the true
NDE / NIE via the mediation formula, and verify their estimators recover
the true decomposition.

Public API
----------
    from oee_synthetic import load_oee
    df = load_oee(n_shifts=1000, seed=0)

Returns a per-shift DataFrame with columns:
    shift_id    integer index
    line_id     one of L1, L2, L3, L4 (production line; a confounder)
    program     0 or 1 (the maintenance program intervention)
    A           Availability fraction, in (0, 1)
    P           Performance fraction, in (0, 1)
    Q           Quality fraction, in (0, 1)
    OEE         A * P * Q (the composite KPI; the outcome)

Structural causal model
-----------------------
The four production lines have different baseline OEE drivers:

    base_A[L1] = 0.85, base_A[L2] = 0.88, base_A[L3] = 0.82, base_A[L4] = 0.86
    base_P[L1] = 0.92, base_P[L2] = 0.90, base_P[L3] = 0.91, base_P[L4] = 0.93
    base_Q[L1] = 0.97, base_Q[L2] = 0.96, base_Q[L3] = 0.97, base_Q[L4] = 0.96

The maintenance-program assignment is *biased toward older lines* that need
maintenance more often (this is the back-door confound a capstone has to
adjust for):

    P(program = 1 | L1) = 0.50      # older line; high maintenance need
    P(program = 1 | L2) = 0.30
    P(program = 1 | L3) = 0.55      # oldest line
    P(program = 1 | L4) = 0.35

The maintenance program causes shifts in each of the three drivers (the
causal effects we want to recover):

    A = clip(base_A[line] + 0.04 * program + Normal(0, 0.02),  0.5, 1)
    P = clip(base_P[line] + 0.01 * program + Normal(0, 0.015), 0.5, 1)
    Q = clip(base_Q[line] + 0.003 * program + Normal(0, 0.005), 0.5, 1)

The composite KPI:

    OEE = A * P * Q

Note that the SCM has NO direct program -> OEE path; every effect of
program on OEE flows through one of the three drivers. The true NDE of
program on OEE (the part not flowing through A/P/Q) is zero by
construction. The true NIE through each driver path is derivable from
the coefficients above; on the OEE scale, the indirect effects sum to
approximately:

    NIE_A = 0.040 * (base_P x base_Q)  ~  +0.036 OEE points
    NIE_P = 0.010 * (base_A x base_Q)  ~  +0.008 OEE points
    NIE_Q = 0.003 * (base_A x base_P)  ~  +0.002 OEE points
    Total ~ +0.046 OEE points (4.6 percentage points)

So A is the dominant mediator; a defensible capstone will recover this.

Why not real data
-----------------
A real OEE log with a controllable maintenance intervention requires
either internal access to a plant's CMMS or a public dataset that
includes a documented program rollout. Neither is available at the
detail this analysis needs (we need both the assignment AND the three
driver components, separately recorded). The closest public analog
(MetroPT) has the time-series structure but not the explicit
intervention. Curated synthetic is the honest substitute, and the
documented SCM makes the capstone validation tractable.
"""
from pathlib import Path
import numpy as np
import pandas as pd

LINE_IDS = ["L1", "L2", "L3", "L4"]

# Per-line baseline driver values (these are the confounders).
BASE_A = {"L1": 0.85, "L2": 0.88, "L3": 0.82, "L4": 0.86}
BASE_P = {"L1": 0.92, "L2": 0.90, "L3": 0.91, "L4": 0.93}
BASE_Q = {"L1": 0.97, "L2": 0.96, "L3": 0.97, "L4": 0.96}

# Program assignment probabilities per line (biased toward older lines).
P_PROGRAM_GIVEN_LINE = {"L1": 0.50, "L2": 0.30, "L3": 0.55, "L4": 0.35}

# Causal effects of the program on each driver (the truth we want to recover).
EFFECT_PROGRAM_ON_A = 0.04
EFFECT_PROGRAM_ON_P = 0.01
EFFECT_PROGRAM_ON_Q = 0.003

# Noise std per driver.
NOISE_A_STD = 0.02
NOISE_P_STD = 0.015
NOISE_Q_STD = 0.005


def load_oee(n_shifts: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Generate a per-shift OEE log.

    Parameters
    ----------
    n_shifts : int
        Number of shifts (rows) to generate. Default 1000.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns
        shift_id, line_id, program, A, P, Q, OEE
    """
    rng = np.random.default_rng(seed)

    # Assign each shift to a production line uniformly.
    line_id = rng.choice(LINE_IDS, size=n_shifts)

    # Program assignment conditional on line (the back-door confound).
    program_probs = np.array([P_PROGRAM_GIVEN_LINE[l] for l in line_id])
    program = rng.binomial(1, program_probs)

    # Per-line baseline driver values.
    base_a = np.array([BASE_A[l] for l in line_id])
    base_p = np.array([BASE_P[l] for l in line_id])
    base_q = np.array([BASE_Q[l] for l in line_id])

    # Per-shift driver realisations (program effect + line baseline + noise).
    A = np.clip(base_a + EFFECT_PROGRAM_ON_A * program + rng.normal(0, NOISE_A_STD, n_shifts), 0.5, 1.0)
    P = np.clip(base_p + EFFECT_PROGRAM_ON_P * program + rng.normal(0, NOISE_P_STD, n_shifts), 0.5, 1.0)
    Q = np.clip(base_q + EFFECT_PROGRAM_ON_Q * program + rng.normal(0, NOISE_Q_STD, n_shifts), 0.5, 1.0)

    OEE = A * P * Q

    return pd.DataFrame({
        "shift_id": np.arange(n_shifts),
        "line_id":  line_id,
        "program":  program.astype(int),
        "A":        A,
        "P":        P,
        "Q":        Q,
        "OEE":      OEE,
    })


def true_oee_decomposition() -> dict:
    """Return the analytic ground-truth NDE / NIE decomposition for this SCM.

    Useful for the capstone to *validate* its NDE/NIE estimator: recovering
    these numbers from `load_oee()`-generated data is the success criterion.
    """
    base_a = np.mean(list(BASE_A.values()))
    base_p = np.mean(list(BASE_P.values()))
    base_q = np.mean(list(BASE_Q.values()))
    nie_a = EFFECT_PROGRAM_ON_A * base_p * base_q
    nie_p = EFFECT_PROGRAM_ON_P * base_a * base_q
    nie_q = EFFECT_PROGRAM_ON_Q * base_a * base_p
    return {
        "NDE":   0.0,             # the SCM has no direct program -> OEE path
        "NIE_via_A": float(nie_a),
        "NIE_via_P": float(nie_p),
        "NIE_via_Q": float(nie_q),
        "total_indirect": float(nie_a + nie_p + nie_q),
    }


if __name__ == "__main__":
    df = load_oee(n_shifts=2000, seed=0)
    print(f"Generated {len(df)} shifts across {df['line_id'].nunique()} lines.")
    print(f"Program rate by line:")
    print(df.groupby('line_id')['program'].mean().round(3).to_string())
    print()
    print(f"OEE summary by program:")
    print(df.groupby('program')[['A', 'P', 'Q', 'OEE']].mean().round(4).to_string())
    print()
    print(f"True decomposition (ground truth from SCM):")
    for k, v in true_oee_decomposition().items():
        print(f"  {k}: {v:+.5f}")
