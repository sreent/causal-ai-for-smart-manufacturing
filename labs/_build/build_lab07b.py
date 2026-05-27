"""Build labs/ch07/lab07b.ipynb — time-varying capacity-drop exposure on LFP cycle life."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 7B — Time-Varying Capacity-Drop Exposure on LFP Cycle Life

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch07/lab07b.ipynb)

**Companion to Lab 7A.** Lab 7A built a synthetic time-varying-treatment SCM with known time-varying confounding, applied g-computation and IPTW marginal structural models, and verified each recovers the true cumulative effect. **Lab 7B applies the same machinery to per-cycle capacity-drop data from the Severson 2019 LFP cells.**

The deliverable: an honest estimate of how *cumulative early-cycle high-drop exposure* relates to eventual cycle life, with **time-varying confounding** (past capacity level affects both current drop rate and future EOL) handled explicitly via g-computation.

**Dataset.** 124 LFP cells × 99 cycles (the pre-processed `lfp_cell_cycle.csv` slice with per-cycle `max_cap` and `mean_cap`). The cell-level `cycle_life` is the final outcome."""),

md("""## What this lab is *not* doing

- **Full counterfactual trajectory simulation.** We use a one-shot g-comp on the cumulative-exposure summary, not a sequential simulator. The chapter's Markov-decision treatment of dynamic regimes is in Lab 8B.
- **Censoring corrections.** Some Severson cells have right-censored cycle_lives; the pre-processed slice ships them as continuous values.
- **Operating-condition controls.** Temperature, current, etc., are not in the vendored slice (only capacity). A richer analysis would condition on these too."""),

code("""%pip install -q numpy pandas matplotlib scikit-learn"""),

code("""import os, sys, urllib.request, pathlib

DATA = pathlib.Path("/content")
for name in ("lfp_prep.py", "lfp_cell_summary.csv", "lfp_cell_cycle.csv"):
    p = DATA / name
    if not p.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/{name}",
            p,
        )
sys.path.insert(0, str(DATA))

from lfp_prep import load_lfp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load per-cycle trajectories"""),

code("""cyc = load_lfp(chapter=7)
print(f"Rows (cell-cycles): {len(cyc)}")
print(f"Unique cells:       {cyc['cell_id'].nunique()}")
print(f"Cycles per cell:    {cyc.groupby('cell_id').size().mean():.0f}")
print()
print(cyc.head(3).to_string())"""),

md("""## Part 2 — Compute per-cycle capacity drop and define the time-varying treatment

For each cell, the per-cycle *drop* in `max_cap` is the difference from the previous cycle. We define `A_t = 1` if the drop at cycle $t$ is in the *top quartile* of all per-cycle drops across the dataset (a high-stress cycle), else 0. This is the time-varying treatment whose cumulative exposure we are studying."""),

code("""cyc = cyc.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
cyc["drop"] = cyc.groupby("cell_id")["max_cap"].diff(-1)   # positive = capacity decreasing
high_thresh = cyc["drop"].quantile(0.75)
cyc["A_t"] = (cyc["drop"] >= high_thresh).astype(int)

print(f"High-drop threshold (top quartile cutoff): {high_thresh:+.5f}")
print(f"Cycles with A_t=1: {cyc['A_t'].sum()} of {len(cyc)} ({100*cyc['A_t'].mean():.1f}%)")"""),

md("""## Part 3 — Build per-cell exposure summary (first 50 cycles)

We focus on *early-cycle* cumulative exposure as the predictor: how many of the first 50 cycles had a high drop, and what was the capacity level at cycle 50 (the time-varying confounder)."""),

code("""early = cyc[cyc["cycle"] <= 50]
exposure = early.groupby("cell_id").agg(
    n_high_cycles=("A_t", "sum"),
    cap_at_50=("max_cap", "last"),
    cap_at_2=("max_cap", "first"),
    cycle_life=("cycle_life", "first"),
).reset_index()
exposure["cap_drop_so_far"] = exposure["cap_at_2"] - exposure["cap_at_50"]

print(exposure.head(5).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print()
print(f"Range of n_high_cycles: {exposure['n_high_cycles'].min()} to {exposure['n_high_cycles'].max()}")
print(f"Correlation cycle_life vs n_high_cycles:  {exposure['cycle_life'].corr(exposure['n_high_cycles']):+.3f}")"""),

md("""## Part 4 — The naive estimate

Regress `cycle_life` on the cumulative early-cycle exposure `n_high_cycles` without any adjustment. This is what an analyst who ignored time-varying confounding would report."""),

code("""naive = LinearRegression().fit(exposure[["n_high_cycles"]].values, exposure["cycle_life"].values)
print(f"Naive slope: {float(naive.coef_[0]):+.2f} cycles per high-drop cycle")
print(f"Naive intercept: {float(naive.intercept_):+.0f}")
print(f"R^2: {naive.score(exposure[['n_high_cycles']].values, exposure['cycle_life'].values):.3f}")"""),

md("""## Part 5 — The g-computation estimate

The time-varying confounder `cap_drop_so_far` is the post-treatment, pre-outcome history that confounds the exposure–EOL relationship (a cell that drops more in early cycles is *both* more likely to accumulate high-drop cycles *and* less likely to live long, independent of any causal effect of the high-drop cycles themselves).

G-computation:
1. Fit $\\hat{E}[Y \\mid A, L]$ — outcome conditional on cumulative exposure and the confounding history.
2. For each cell, predict the counterfactual $\\hat{Y}$ under each level of $A$ holding $L$ at the empirical distribution.
3. Average to get the marginal effect."""),

code("""# Model the outcome given exposure and the confounder.
features = exposure[["n_high_cycles", "cap_drop_so_far"]].values
y = exposure["cycle_life"].values
model = LinearRegression().fit(features, y)

print(f"Adjusted slope on n_high_cycles:        {float(model.coef_[0]):+.2f} cycles per high-drop cycle")
print(f"Adjusted slope on cap_drop_so_far:      {float(model.coef_[1]):+.1f} cycles per unit capacity drop")
print()

# G-comp marginal effect: shift n_high_cycles by 1 across the empirical L distribution.
counter_lo = features.copy(); counter_lo[:, 0] -= 0.5
counter_hi = features.copy(); counter_hi[:, 0] += 0.5
g_marginal = float((model.predict(counter_hi) - model.predict(counter_lo)).mean())

print(f"G-comp marginal effect (per +1 high-drop cycle, averaging over L):  {g_marginal:+.2f}")
print(f"Naive  marginal effect (no L):                                       {float(naive.coef_[0]):+.2f}")
print(f"Shift from confounding adjustment:                                   {g_marginal - float(naive.coef_[0]):+.2f}")"""),

md("""**Read the shift.** If the g-comp estimate is substantially smaller in magnitude than the naive, time-varying confounding was inflating the apparent harm of high-drop cycles. If it is similar, the time-varying confounder was not biasing the naive estimate much (the chapter's \"informative dropout\" diagnostic).

A *sign flip* between naive and g-comp would indicate that capacity history is a strong enough confounder to invert the conclusion — worth checking before any deployment-relevant claim."""),

md("""## Part 6 — IPTW marginal structural model (sketch)

The MSM treatment from §7's chapter weighs each cell by the inverse of the cumulative probability of receiving its observed exposure given history. For our coarse one-shot summary, this collapses to weighting by 1 / P(n_high_cycles | cap_drop_so_far).

We fit a regression of n_high_cycles on cap_drop_so_far for the propensity, then apply Hajek weights."""),

code("""# Coarse: predict expected exposure from history; weight each cell by inverse density.
prop = LinearRegression().fit(exposure[["cap_drop_so_far"]].values, exposure["n_high_cycles"].values)
expected = prop.predict(exposure[["cap_drop_so_far"]].values)
resid = exposure["n_high_cycles"].values - expected
resid_std = max(resid.std(ddof=1), 1e-6)
from scipy.stats import norm
w = 1.0 / np.clip(norm.pdf(resid / resid_std) / resid_std, 1e-3, None)
w = w / w.mean()    # Hajek normalisation

msm = LinearRegression()
msm.fit(exposure[["n_high_cycles"]].values, exposure["cycle_life"].values, sample_weight=w)
print(f"MSM (IPTW-weighted) slope: {float(msm.coef_[0]):+.2f}")
print(f"G-comp slope:               {g_marginal:+.2f}")
print(f"Naive slope:                {float(naive.coef_[0]):+.2f}")"""),

md("""## Part 7 — Decision

Three bullets:

1. **Effect of one additional early-cycle high-drop cycle on eventual cycle life** — the g-comp number from Part 5 and the IPTW number from Part 6 should agree within sampling error if both modelling steps are well-specified. The reported number is whichever the modelling diagnostics favour.

2. **The naive estimate is biased by time-varying confounding.** Show the shift (g-comp minus naive) as evidence; if the shift is large, naive analyses of cumulative cycle-life exposure overstate the harm of high-drop cycles.

3. **Engineering implication.** If the effect of an additional high-drop cycle is small relative to the cell's natural cycle-life variation, fast-charge protocols that increase early-cycle drop within a tolerable range are pareto-improvements (more cycling capacity per unit time). If the effect is large, throttling fast-charge severity in the first 50 cycles preserves cycle life."""),

md("""## Reflection

**The time-varying confounder is a feature you can compute, not a hypothesis you can avoid.** Past capacity level is a measurable variable; ignoring it because the data is messy gives a biased estimate. G-comp and IPTW are the two principled ways to handle it — they should agree, and disagreement is diagnostic.

**Coarse summarisation is a modelling choice with consequences.** We collapsed 50 cycles of trajectory into two numbers (cumulative exposure + final capacity). A richer analysis would model the full per-cycle treatment-confounder sequence, with a separate propensity per cycle. That is Lab 8B's territory."""),

md("""## What's next

Lab 8B revisits time-varying treatments on Backblaze drive SMART telemetry, where the actions (replace / keep) are discrete and the dynamic-treatment-regime framing matches the chapter exactly."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch07" / "lab07b.ipynb", cells)
print("Built lab07b.ipynb")
