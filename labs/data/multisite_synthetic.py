"""Curated synthetic multi-site industrial dataset for transportability capstones.

A clean two-plant scenario where the *structural causal model is the same*
but the *covariate distributions differ*. The capstone question:
estimate the average treatment effect at Plant A; predict it at Plant B
*without* training on Plant B's data; compare to the direct Plant-B
estimate; report the source -> target gap and what closed it.

No public real cross-site industrial dataset exists in clean enough form
to support this exercise (most multi-site data is internal to one
company; published cross-site studies anonymise the sites and aggregate
away the effect-modifier structure). Curated synthetic is the honest
substitute, and a documented SCM makes the transportability claim
analytically verifiable.

Public API
----------
    from multisite_synthetic import load_multisite, true_ate_per_site
    df = load_multisite(n_per_site=600, seed=0)
    true_ates = true_ate_per_site()    # ground truth for the capstone

Returns a per-unit DataFrame with columns:
    unit_id     integer index
    site        one of A, B (plant label)
    raw_grade   continuous in (0, 1) - the EFFECT MODIFIER
    treatment   0 or 1 (randomised within site)
    outcome     continuous outcome (e.g., scaled yield)

Structural causal model
-----------------------
Both sites share the same SCM in symbolic form. Only the *distribution* of
the effect modifier `raw_grade` differs.

    site ~ Categorical([A, B], 50/50)

    # Effect modifier - DIFFERENT distribution per site (this is the only
    # structural shift; the SCM equations are identical).
    raw_grade | site = A ~ Beta(2, 5)    # mean ~0.29, skewed LOW
    raw_grade | site = B ~ Beta(5, 2)    # mean ~0.71, skewed HIGH

    # Treatment - cleanly randomised within site (50/50). This is the
    # "controlled trial in each plant" idealisation; the capstone's
    # job is purely the transport step, not the identification step.
    treatment ~ Bernoulli(0.5)

    # Outcome - treatment effect SCALES with raw_grade. This is the
    # effect-modifier structure transportability has to handle.
    tau(raw_grade) = 0.1 + 0.4 * raw_grade   # in [0.1, 0.5]
    outcome       = 2.0 + 0.5 * raw_grade + tau(raw_grade) * treatment + Normal(0, 0.3)

Ground truth per site (expected ATE under each plant's raw_grade distribution):

    ATE_A = E_{raw_grade ~ Beta(2,5)}[tau(raw_grade)] = 0.1 + 0.4 * (2/7)   = 0.214
    ATE_B = E_{raw_grade ~ Beta(5,2)}[tau(raw_grade)] = 0.1 + 0.4 * (5/7)   = 0.386

The naive "fit on A, apply on B" estimate would give ~0.214 (Plant A's
ATE), missing the true Plant-B ATE of ~0.386 by ~0.17 units - a 45 %
under-estimate.

A correctly transported estimate via Plant-B's raw_grade distribution
(re-weighting source to target) recovers ATE_B from Plant-A data alone:

    ATE_B_transported = E_{raw_grade ~ Beta(5,2)}[E[Y|do(T=1), grade] - E[Y|do(T=0), grade]]
                      ~ 0.386

This number is computable from the analytic SCM and is what the
capstone's reweighted-source estimate must approximate.

Why not real data
-----------------
Cross-site manufacturing data with documented effect modifiers is almost
always internal. Public multi-site datasets either aggregate sites
(losing the modifier structure) or anonymise sites (preventing
domain-grounded transport arguments). Curated synthetic with a known
modifier structure gives the capstone what real data cannot: a
*verifiable* ground truth for the transported estimate.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SITES = ["A", "B"]

# Site-specific Beta parameters for the raw_grade effect modifier.
BETA_PARAMS_PER_SITE = {
    "A": (2.0, 5.0),    # mean ~0.29 (low-grade-heavy)
    "B": (5.0, 2.0),    # mean ~0.71 (high-grade-heavy)
}

# Treatment effect function (the same across sites).
TAU_INTERCEPT = 0.1
TAU_SLOPE_ON_GRADE = 0.4

# Outcome model coefficients (the same across sites).
OUTCOME_INTERCEPT = 2.0
OUTCOME_SLOPE_ON_GRADE = 0.5
OUTCOME_NOISE_STD = 0.3


def _tau(raw_grade):
    return TAU_INTERCEPT + TAU_SLOPE_ON_GRADE * raw_grade


def load_multisite(n_per_site: int = 600, seed: int = 0) -> pd.DataFrame:
    """Generate a per-unit multi-site frame.

    Parameters
    ----------
    n_per_site : int
        Units per site. Default 600 -> 1200 total.
    seed : int
        RNG seed.

    Returns
    -------
    DataFrame with columns
        unit_id, site, raw_grade, treatment, outcome
    """
    rng = np.random.default_rng(seed)

    rows = []
    for site in SITES:
        alpha, beta = BETA_PARAMS_PER_SITE[site]
        raw_grade = rng.beta(alpha, beta, size=n_per_site)
        treatment = rng.binomial(1, 0.5, size=n_per_site)
        tau = _tau(raw_grade)
        outcome = (OUTCOME_INTERCEPT
                   + OUTCOME_SLOPE_ON_GRADE * raw_grade
                   + tau * treatment
                   + rng.normal(0, OUTCOME_NOISE_STD, n_per_site))
        rows.append(pd.DataFrame({
            "site":      site,
            "raw_grade": raw_grade,
            "treatment": treatment.astype(int),
            "outcome":   outcome,
        }))

    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "unit_id", np.arange(len(df)))
    return df


def true_ate_per_site() -> dict:
    """Return the analytic ground-truth ATE per site (and the transported ATEs).

    Capstone validation: estimating these from `load_multisite()`-generated
    data is the success criterion. The transported_AB / transported_BA
    entries are what a correctly executed reweighting from source to
    target should recover.
    """
    # ATE_site = E_{raw_grade ~ Beta(alpha, beta)}[tau(raw_grade)]
    # mean of Beta(alpha, beta) = alpha / (alpha + beta)
    out = {}
    for site, (alpha, beta) in BETA_PARAMS_PER_SITE.items():
        mean_grade = alpha / (alpha + beta)
        ate = TAU_INTERCEPT + TAU_SLOPE_ON_GRADE * mean_grade
        out[f"ATE_{site}"] = float(ate)
    # The transported ATE from A to B (fit on A, apply on B's grade dist)
    # equals ATE_B under the SCM's effect-modifier setup, since tau is a
    # function of grade alone and the outcome model coefficients are
    # constant across sites. The naive (un-transported) estimate equals
    # the source ATE.
    out["naive_A_applied_to_B"] = out["ATE_A"]
    out["naive_B_applied_to_A"] = out["ATE_B"]
    out["transported_A_to_B"]   = out["ATE_B"]
    out["transported_B_to_A"]   = out["ATE_A"]
    return out


if __name__ == "__main__":
    df = load_multisite(n_per_site=1000, seed=0)
    print(f"Generated {len(df)} units across {df['site'].nunique()} sites.")
    print()
    print("Per-site summary:")
    print(df.groupby('site').agg(
        n=('unit_id', 'size'),
        mean_grade=('raw_grade', 'mean'),
        mean_outcome_treated=('outcome', lambda x: x[df.loc[x.index, 'treatment'] == 1].mean()),
        mean_outcome_control=('outcome', lambda x: x[df.loc[x.index, 'treatment'] == 0].mean()),
    ).round(3).to_string())
    print()
    print("Naive within-site ATE (mean outcome difference, no covariate adjustment):")
    for site in SITES:
        sub = df[df['site'] == site]
        ate = sub.loc[sub['treatment'] == 1, 'outcome'].mean() - sub.loc[sub['treatment'] == 0, 'outcome'].mean()
        print(f"  Plant {site}: {ate:+.4f}")
    print()
    print("True ATE (from analytic SCM):")
    for k, v in true_ate_per_site().items():
        print(f"  {k}: {v:+.4f}")
