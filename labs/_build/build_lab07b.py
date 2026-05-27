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

- **Full per-cycle sequential simulation across all 50 cycles.** Parts 4-6 use a one-shot g-comp / IPTW on a cumulative early-cycle exposure summary. Part 7 then implements a proper *sequential* g-formula (iterated conditional expectation) on two cycle bins to demonstrate the chapter's full machinery and compare the two estimates side-by-side. A 50-bin version is a straightforward but unwieldy extension.
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

md("""## Part 7 — Sequential g-formula on two cycle bins  *(advanced)*

> The lab from a Severson-engineer's perspective is essentially complete after Part 6 — you have a one-shot g-comp estimate (Part 5) and an IPTW MSM (Part 6) and the comparison between them. **Part 7 is the methodological extension** that demonstrates the chapter's *full* sequential machinery on the same data, so you can see what the coarsening in Parts 5-6 left on the table. A first-pass reader can skip to Part 8 (Decision) and return here once the basics are comfortable.

**Why a second pass at this?** Parts 5-6 estimated the effect of *cumulative* exposure $A = $ "total high-drop cycles in days 2-50". This collapses the trajectory into a single number. The chapter's *full* time-varying g-formula keeps the sequence intact: at each time step $t$, the time-varying confounder $L_t$ depends on past treatment $A_{<t}$, the next-step treatment $A_t$ depends on $L_t$, and the outcome depends on the entire path. Whether this granularity matters is an *empirical* question; if $A_1$ barely shifts $L_2$, the one-shot and sequential answers agree. If it shifts $L_2$ substantially, only the sequential answer is correct, and Parts 5-6 are *biased*.

**Estimator family map.** The lab now demonstrates four estimators on the same data, in increasing order of methodological rigour:

| Part | Estimator | Treats time as | What it captures |
|------|-----------|----------------|------------------|
| 4 | Naive OLS | None — cumulative summary only | The marginal $A$-$Y$ association (biased by L) |
| 5 | One-shot g-comp | None — cumulative $(A, L)$ summary | The summary-level $A$-$Y$ effect, conditioning on $L$ once |
| 6 | One-shot IPTW MSM | None — cumulative $(A, L)$ summary | Same target as Part 5, by weighting rather than adjustment |
| 7 | Sequential ICE g-formula (this Part) | Two cycle bins, with $L_t$ propagated | The chapter's full machinery: counterfactual $L_t$ under each $A_{<t}$ trajectory |

If Parts 5-6 and Part 7 give similar answers, the coarsening was safe. If they diverge, Part 7 is the principled estimator and Parts 5-6 are biased.

**Setup.** We demonstrate the sequential machinery on two cycle bins:

- **Bin 1:** cycles 2-25. $A_1$ = 1 if the cell had above-median high-drop cycles in this bin; $L_1$ = `max_cap` at cycle 25.
- **Bin 2:** cycles 26-50. $A_2$ = 1 if above-median in this bin; $L_2$ = `max_cap` at cycle 50.

The causal sequence is $A_1 \\to L_2 \\to A_2 \\to Y$ (with $L_1$ and $L_2$ as time-varying confounders). The **iterated-conditional-expectation (ICE) g-formula** estimates $E[Y(a_1, a_2)]$ by:

1. Fit $\\hat{L}_2(A_1, L_1)$ — how the next-period capacity responds to past treatment.
2. Fit $\\hat{Y}(A_1, L_1, A_2, L_2)$ — terminal outcome model.
3. For each cell and each $(a_1, a_2)$ trajectory, predict the counterfactual $\\hat{L}_2(a_1, L_1^{\\text{obs}})$ and then $\\hat{Y}(a_1, L_1^{\\text{obs}}, a_2, \\hat{L}_2)$. Average over cells.

This *propagates the counterfactual* through the time-varying confounder, which the one-shot version does not."""),

code("""BIN1_END = 25
BIN2_END = 50

# Per-cell two-bin panel: A_1, L_1, A_2, L_2, Y
cyc_sorted = cyc.sort_values(['cell_id', 'cycle']).reset_index(drop=True)
bin1 = cyc_sorted[(cyc_sorted['cycle'] >= 2) & (cyc_sorted['cycle'] <= BIN1_END)]
bin2 = cyc_sorted[(cyc_sorted['cycle'] > BIN1_END) & (cyc_sorted['cycle'] <= BIN2_END)]

panel = (
    pd.DataFrame({'cell_id': cyc_sorted['cell_id'].unique()})
      .merge(bin1.groupby('cell_id')['A_t'].sum().rename('n_high_b1').reset_index(), on='cell_id', how='left')
      .merge(bin2.groupby('cell_id')['A_t'].sum().rename('n_high_b2').reset_index(), on='cell_id', how='left')
      .merge(cyc_sorted[cyc_sorted['cycle'] == BIN1_END][['cell_id', 'max_cap']].rename(columns={'max_cap': 'L_1'}), on='cell_id', how='left')
      .merge(cyc_sorted[cyc_sorted['cycle'] == BIN2_END][['cell_id', 'max_cap']].rename(columns={'max_cap': 'L_2'}), on='cell_id', how='left')
      .merge(cyc_sorted.groupby('cell_id')['cycle_life'].first().reset_index(), on='cell_id', how='left')
).dropna().reset_index(drop=True)

panel['A_1'] = (panel['n_high_b1'] > panel['n_high_b1'].median()).astype(int)
panel['A_2'] = (panel['n_high_b2'] > panel['n_high_b2'].median()).astype(int)

print(f"Two-bin panel shape: {panel.shape}")
print(f"  A_1 prevalence: {panel['A_1'].mean():.2%}")
print(f"  A_2 prevalence: {panel['A_2'].mean():.2%}")
print()
print("Joint trajectory counts (A_1, A_2):")
print(panel.groupby(['A_1', 'A_2']).size().rename('count').reset_index().to_string(index=False))"""),

code("""# Step 1: How does L_2 respond to past treatment A_1 and L_1?
L2_model = LinearRegression().fit(panel[['A_1', 'L_1']].values, panel['L_2'].values)
beta_A1_on_L2 = float(L2_model.coef_[0])
print(f"Coefficient of A_1 on L_2: {beta_A1_on_L2:+.5f}  (a cell with A_1=1 ends bin 2 with capacity {beta_A1_on_L2:+.4f} relative to A_1=0)")

# Step 2: Terminal outcome model
Y_model = LinearRegression().fit(panel[['A_1', 'L_1', 'A_2', 'L_2']].values, panel['cycle_life'].values)
print(f"\\nTerminal Y coefficients:")
for name, val in zip(['A_1', 'L_1', 'A_2', 'L_2'], Y_model.coef_):
    print(f"  {name}: {float(val):+.2f}")

# Step 3: Iterated counterfactual prediction
def E_Y_seq(a1, a2):
    L1 = panel['L_1'].values
    L2_counterfactual = L2_model.predict(np.column_stack([np.full(len(panel), a1), L1]))
    Y_hat = Y_model.predict(np.column_stack([
        np.full(len(panel), a1), L1,
        np.full(len(panel), a2), L2_counterfactual,
    ]))
    return float(Y_hat.mean())

E00 = E_Y_seq(0, 0)
E01 = E_Y_seq(0, 1)
E10 = E_Y_seq(1, 0)
E11 = E_Y_seq(1, 1)

print(f"\\nSequential g-formula counterfactuals:")
print(f"  E[Y(A_1=0, A_2=0)] = {E00:.0f}  (never high)")
print(f"  E[Y(A_1=1, A_2=0)] = {E10:.0f}  (early high only)")
print(f"  E[Y(A_1=0, A_2=1)] = {E01:.0f}  (late high only)")
print(f"  E[Y(A_1=1, A_2=1)] = {E11:.0f}  (always high)")
print()
ATE_always = E11 - E00
ATE_early  = E10 - E00
ATE_late   = E01 - E00
print(f"  ATE always-high vs never-high:       {ATE_always:+.1f} cycles")
print(f"  ATE early-only vs never:              {ATE_early:+.1f} cycles")
print(f"  ATE late-only vs never:               {ATE_late:+.1f} cycles")"""),

code("""# Side-by-side: sequential g-formula vs the one-shot g-comp from Part 5.
# The one-shot Part-5 slope was per +1 high-drop cycle; convert to a comparable
# scale by multiplying by the bin's mean n_high_cycles for binary A_b.
mean_bin_n_high = panel.loc[panel['A_1'] == 1, 'n_high_b1'].mean() - panel.loc[panel['A_1'] == 0, 'n_high_b1'].mean()
one_shot_implied_2bin = float(naive.coef_[0]) * mean_bin_n_high * 2  # both bins binarised
g_one_shot_implied   = g_marginal * mean_bin_n_high * 2

print(f"On the 'always-high vs never-high' contrast:")
print(f"  Naive (one-shot, implied)            = {one_shot_implied_2bin:+.1f} cycles")
print(f"  G-comp (one-shot, implied)           = {g_one_shot_implied:+.1f} cycles")
print(f"  Sequential g-formula (ICE, Part 7)   = {ATE_always:+.1f} cycles")
print()
if abs(beta_A1_on_L2) < 0.001:
    print("Interpretation: A_1's effect on L_2 is essentially zero in this dataset.")
    print("That means the time-varying confounder isn't propagating through past")
    print("treatment, so the one-shot and sequential estimates should agree closely.")
else:
    print("Interpretation: A_1 does shift L_2 (the time-varying confounder responds")
    print("to past treatment). The sequential g-formula propagates this counterfactual")
    print("shift; the one-shot version doesn't.")"""),

md("""**Read the comparison.** Three things to watch:

1. **Step 1's coefficient: does $A_1$ shift $L_2$?** If it is near zero, the *direct* time-varying-confounding effect — past treatment moving the future confounder — is small. In the Severson data this coefficient is ~$-0.002$, a few thousandths of a unit of capacity, which is tiny in absolute terms.

2. **L₂'s coefficient in the Y model is large.** A small counterfactual shift in $L_2$ multiplied by a large outcome slope on $L_2$ still propagates to a non-trivial change in $\\hat{Y}$. This is the *mechanism* by which the sequential g-formula and the one-shot estimate can differ even when the $A_1 \\to L_2$ link looks weak in isolation.

3. **Why the one-shot and sequential estimates still differ in our run.** The biggest reason is *not* that the LFP chain is strong — it isn't — but that the two estimators use *different feature representations*: the one-shot regresses on a 2-feature summary $(A_{\\text{cum}}, L_{\\text{cum}})$, while the sequential ICE uses the four-feature $(A_1, L_1, A_2, L_2)$ panel. Different bias-variance regimes. With $n = 124$ cells, the four-feature OLS is also more sensitive to the collinearity between $L_1$ and $L_2$ (note the large opposing coefficients in the terminal model — a sign that capacity at two nearby checkpoints is highly correlated). A regularised estimator or a saturated nonparametric alternative would tighten the comparison.

**The chapter's lesson, applied here.** The sequential ICE g-formula is the principled estimator when the time-varying chain matters; the one-shot is a coarser approximation. In Severson LFP the chain is weak in absolute terms, so a serious analyst would report the sequential number with a wider uncertainty band and note that the one-shot lands in the same neighbourhood. The diagnostic to trust is *Step 1* — the magnitude of $A_t$'s effect on the next-period confounder — not the closeness of the final point estimates."""),

md("""## Part 8 — Decision

Three bullets:

1. **Effect of one additional early-cycle high-drop cycle on eventual cycle life** — the g-comp number from Part 5 and the IPTW number from Part 6 should agree within sampling error if both modelling steps are well-specified. The reported number is whichever the modelling diagnostics favour.

2. **The naive estimate is biased by time-varying confounding.** Show the shift (g-comp minus naive) as evidence; if the shift is large, naive analyses of cumulative cycle-life exposure overstate the harm of high-drop cycles.

3. **Engineering implication.** If the effect of an additional high-drop cycle is small relative to the cell's natural cycle-life variation, fast-charge protocols that increase early-cycle drop within a tolerable range are pareto-improvements (more cycling capacity per unit time). If the effect is large, throttling fast-charge severity in the first 50 cycles preserves cycle life."""),

md("""## Reflection

**The time-varying confounder is a feature you can compute, not a hypothesis you can avoid.** Past capacity level is a measurable variable; ignoring it because the data is messy gives a biased estimate. G-comp and IPTW are the two principled ways to handle it on a summary; the sequential ICE g-formula in Part 7 is what you reach for when the chain $A_t \\to L_{t+1} \\to A_{t+1}$ matters substantively.

**Granularity is a modelling choice, not a methodological one.** Lab 7B implements the same chapter machinery at two granularities — one bin (Parts 4-6) and two bins (Part 7). A 5-bin or 49-bin version is the same recipe with more terms; the curve of estimates as you refine the binning is itself a diagnostic. If the answer stabilises by two bins, two bins is enough; if it drifts as you refine, the coarsening was hiding signal.

**When to use each.** The Severson chain is weak: $A_1$ barely shifts $L_2$, so the coarsened g-comp gives a number close to the sequential ICE. For datasets where treatment has strong dynamical feedback on the confounder (e.g., a medical PM cohort where an early intervention permanently alters the patient's state), the sequential machinery is the only valid choice."""),

md("""## What's next

Lab 8B revisits time-varying treatments on Backblaze drive SMART telemetry, where the actions (replace / keep) are discrete and the dynamic-treatment-regime framing matches the chapter exactly."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch07" / "lab07b.ipynb", cells)
print("Built lab07b.ipynb")
