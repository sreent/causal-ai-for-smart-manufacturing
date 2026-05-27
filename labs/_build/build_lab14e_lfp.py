"""Build labs/ch14/lab14e_lfp.ipynb — guided capstone Starter C on LFP batteries."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402


def reveal(content: str) -> dict:
    return md("<details>\n<summary><b>Click to reveal sample answer</b></summary>\n\n"
              + content + "\n\n</details>")


cells = [

md("""# Lab 14E — Guided Capstone (Starter C): Cumulative Early-Cycle High-Drop Exposure and LFP Cycle Life

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14e_lfp.ipynb)

**Starter C** of the §14.7 capstone. This is the *methodologically deep* starter: instead of a one-shot ATE, the capstone targets a **sequential time-varying treatment effect** on the Severson LFP cell data. The estimator stack is the chapter-7 family (g-formula, IPTW MSM, sequential ICE), and the deliverable is an honest report on whether high-drop early-cycle exposure *causes* shorter cycle life.

**The capstone question.** *What is the causal effect of cumulative early-cycle high-drop exposure (cycles 2-50) on a cell's eventual cycle_life, accounting for the time-varying confounding by the previous cycle's capacity?*

**Companion labs.** Lab 7B is the methodological reference (one-shot vs sequential g-formula); Lab 13B is the sensitivity tooling.

**Why this starter is "methodologically deep".** The chapter-7 sequential machinery is the part of the course where most real-data analyses cheat — they collapse the trajectory into a single number and apply a cross-sectional estimator. The capstone done well runs both the one-shot and the sequential ICE estimator, compares them, and reports the *direction of disagreement* as a diagnostic. That is the methodological depth this starter rewards."""),

md("""## Setup"""),

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
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

rng = np.random.default_rng(0)
cyc = load_lfp(chapter=7).sort_values(['cell_id', 'cycle']).reset_index(drop=True)
print(f'Rows: {len(cyc)}   Unique cells: {cyc[\"cell_id\"].nunique()}')"""),

md("""## Artifact 1 — Problem Statement and DAG

### Q1.1 Write the problem statement.

*Hint.* The decision a battery-engineering team makes is whether to *throttle fast-charge severity* in the first 50 cycles. Their tradeoff: more aggressive charging gets cells to-spec faster but may shorten life.

*Your turn.*"""),

md("""*[Problem statement.]*"""),

reveal("""A battery manufacturer can tune the fast-charge protocol's severity during the first 50 cycles. More aggressive protocols cause more *high-drop cycles* (per-cycle capacity loss in the top-quartile), but it is unclear whether the extra exposure reduces *cycle_life* enough to justify throttling. The team needs the causal effect of cumulative early-cycle high-drop exposure on cycle_life, *accounting for the time-varying confounding* that cells with low early capacity are both more likely to suffer high-drop cycles AND more likely to die early. If $\\hat\\tau$ (per +1 high-drop cycle) is more negative than −5 cycles of life, throttle; otherwise leave the aggressive protocol in place."""),

md("""### Q1.2 Draw the DAG with time-varying L and A.

*Hint.* Two time-steps is enough to show the structure. The key edges: $A_t$ affects $L_{t+1}$ (treatment shifts the future confounder); $L_t$ affects $A_t$ (the confounder drives the next-step treatment).

*Your turn.*"""),

md("""*[DAG.]*"""),

reveal("""```
   L_1 (capacity at end of bin 1) ──► L_2 ──► Y (cycle_life)
       │                              ▲       ▲
       └──► A_2 (high-drop in bin 2) ─┘       │
                                              │
   A_1 (high-drop in bin 1) ──► L_2 ───────────┘ (TIME-VARYING CONFOUNDING)
       │
       └──► Y
```

The defining feature: $A_1$ does NOT just affect $Y$ directly; it *also* shifts $L_2$, which then mediates additional effect on both $A_2$ and $Y$. Naively conditioning on $L_2$ blocks part of the very treatment effect we want to measure (the *indirect* $A_1 \\to L_2 \\to Y$ path). Sequential g-formula handles this by propagating the counterfactual $L_2$ through the change in $A_1$."""),

md("""### Q1.3 Defend the time-varying-confounding structure from battery physics.

*Your turn.*"""),

md("""*[Physics defense.]*"""),

reveal("""**$L_t \\to A_t$ (capacity confounds next high-drop cycle):** cells with already-degraded capacity suffer larger absolute drops in the next cycle simply because the per-cycle wear is proportional to current state; a 100 mAh cell loses more capacity per high-stress cycle than a 1000 mAh cell. This is the *informative-dropout* structure Chapter 7 warns about.

**$A_t \\to L_{t+1}$ (treatment shifts next confounder):** a high-drop cycle reduces capacity going into the next cycle by definition — that's what "high-drop" means. The future confounder is mechanistically downstream of the past treatment."""),

md("""## Artifact 2 — Estimand

### Q2.1 Write the sequential estimand in counterfactual notation.

*Hint.* The chapter-7 estimand for two bins compares $Y(\\bar A = 1)$ to $Y(\\bar A = 0)$ — the "always high" vs "never high" trajectories.

*Your turn.*"""),

md("""*[Estimand.]*"""),

reveal("""$$\\tau \\;=\\; E[Y(A_1 = 1, A_2 = 1)] - E[Y(A_1 = 0, A_2 = 0)] \\;=\\; \\sum_{l_1, l_2} \\big(\\hat\\mu(1, l_1, 1, l_2) - \\hat\\mu(0, l_1, 0, l_2)\\big) P(l_1) P(l_2 \\mid A_1 = a_1, L_1 = l_1),$$

where $A_t$ is a binary indicator that bin $t$'s n_high_cycles is above-median, $L_t$ is the cell's capacity at the end of bin $t$, $Y$ is cycle_life, and the ICE g-formula integrates $L_2$ under the counterfactual $A_1$ value."""),

md("""### Q2.2 Couple to a decision threshold.

*Your turn.*"""),

md("""*[Decision threshold.]*"""),

reveal("""- $\\hat\\tau < -50$ cycles (always-high reduces cycle life by ≥ 50 cycles): **throttle** fast-charge severity in the first 50 cycles.
- $-50 < \\hat\\tau < 0$: marginal harm; recommend a 6-month controlled trial before any throttle.
- $\\hat\\tau \\geq 0$: no evidence of harm; leave the aggressive protocol in place."""),

md("""## Artifact 3 — Identification

### Q3.1 Name the strategy and the identifying assumptions for sequential g-formula.

*Hint.* The chapter-7 assumptions are: sequential ignorability (no unmeasured confounders at each time step), positivity (overlap at each step), consistency.

*Your turn.*"""),

md("""*[Strategy + assumptions.]*"""),

reveal("""**Strategy.** Sequential g-formula (iterated conditional expectation form). The estimand is identified under three assumptions per time-step:

1. *Sequential ignorability:* $Y(\\bar a) \\perp A_t \\mid \\bar L_t, \\bar A_{t-1}$ — at each time step, the future counterfactual outcome is independent of treatment given the *history* of confounders and prior treatments.
2. *Positivity:* $0 < P(A_t = 1 \\mid \\bar L_t, \\bar A_{t-1}) < 1$ at every history.
3. *Consistency:* the observed $Y$ for a cell that followed $\\bar A$ equals $Y(\\bar A)$ — no measurement drift between potential outcome and observed outcome."""),

md("""### Q3.2 Identify the weakest of the three assumptions and link forward to Artifact 5.

*Your turn.*"""),

md("""*[Weakest assumption + sensitivity link.]*"""),

reveal("""**Sequential ignorability is the weakest.** Cell-level latents (electrolyte batch, electrode-coating thickness, microscopic defects) are not measured in the Severson slice and may drive both high-drop incidence and longevity. Conditioning on prior capacity captures *some* of this (a defective cell shows up as low capacity early), but a residual unmeasured-confounder channel remains.

Artifact 5 bounds this with a γ-sweep sensitivity on the $A_t \\to L_{t+1}$ coefficient: if the analytic effect shrinks toward zero as we attribute more of the $A_t$-on-$L_{t+1}$ shift to a latent confounder, the sequential estimate is fragile; if it stays stable, the conclusion is robust."""),

md("""## Artifact 4 — Estimator

### Q4.1 Build the two-bin panel (A_1, L_1, A_2, L_2, Y) per cell.

*Hint.* The two-bin sequential structure (Lab 7B Part 7): bin the cycles into two consecutive intervals; for each cell, compute $A_t$ as a binary indicator that bin $t$'s n_high_cycles is above the cohort median, and $L_t$ as the cell's max_cap *at the end of bin $t$*. The resulting panel has one row per cell with columns $\\{A_1, L_1, A_2, L_2, Y\\}$. Bin 1 = cycles 2-25; Bin 2 = cycles 26-50.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
high_thresh = cyc['max_cap'].diff(-1).quantile(0.75)   # top-quartile high-drop threshold
cyc['drop'] = cyc.groupby('cell_id')['max_cap'].diff(-1)
cyc['A_t'] = (cyc['drop'] >= high_thresh).astype(int)
BIN1_END, BIN2_END = 25, 50

bin1 = cyc[(cyc['cycle'] >= 2) & (cyc['cycle'] <= BIN1_END)]
bin2 = cyc[(cyc['cycle'] > BIN1_END) & (cyc['cycle'] <= BIN2_END)]
panel = (
    pd.DataFrame({'cell_id': cyc['cell_id'].unique()})
    .merge(bin1.groupby('cell_id')['A_t'].sum().rename('n_high_b1').reset_index(), on='cell_id', how='left')
    .merge(bin2.groupby('cell_id')['A_t'].sum().rename('n_high_b2').reset_index(), on='cell_id', how='left')
    .merge(cyc[cyc['cycle'] == BIN1_END][['cell_id', 'max_cap']].rename(columns={'max_cap': 'L_1'}), on='cell_id', how='left')
    .merge(cyc[cyc['cycle'] == BIN2_END][['cell_id', 'max_cap']].rename(columns={'max_cap': 'L_2'}), on='cell_id', how='left')
    .merge(cyc.groupby('cell_id')['cycle_life'].first().reset_index(), on='cell_id', how='left')
).dropna().reset_index(drop=True)
panel['A_1'] = (panel['n_high_b1'] > panel['n_high_b1'].median()).astype(int)
panel['A_2'] = (panel['n_high_b2'] > panel['n_high_b2'].median()).astype(int)
print(f'Panel: {len(panel)} cells. A_1 prevalence {panel[\"A_1\"].mean():.2%}, A_2 prevalence {panel[\"A_2\"].mean():.2%}')
```
"""),

md("""## The ICE algorithm in 90 seconds (before you implement it)

Sequential iterated-conditional-expectation (ICE) g-formula sounds dense but is straightforward once you trace it on *one* counterfactual trajectory. Consider the trajectory $(a_1 = 0, a_2 = 0)$ — *never-high-drop* — for a single cell with observed $L_1$:

1. **Step 1: model how the time-varying confounder responds to past treatment.** Fit a regression of the *observed* $L_2$ on $(A_1, L_1)$ in the panel. Read off the fitted function $\\hat{L}_2 = g(A_1, L_1)$.
2. **Step 2: predict the counterfactual confounder.** For this cell under $a_1 = 0$, substitute: $\\hat{L}_2^{(a_1=0)} = g(0, L_1^{\\text{obs}})$. This is what $L_2$ *would have been* if we had set $A_1 = 0$, given the cell's observed $L_1$.
3. **Step 3: model the terminal outcome.** Fit a regression of $Y$ on $(A_1, L_1, A_2, L_2)$ using the observed panel. Read off the fitted function $\\hat{Y} = h(A_1, L_1, A_2, L_2)$.
4. **Step 4: predict the counterfactual outcome.** $\\hat{Y}^{(a_1=0, a_2=0)} = h(0, L_1^{\\text{obs}}, 0, \\hat{L}_2^{(a_1=0)})$ — note we plug in the *counterfactual* $L_2$ from Step 2, not the observed one. Average over cells.

Repeating Steps 2 and 4 for the other three counterfactuals — $(0, 1), (1, 0), (1, 1)$ — gives the four $E[Y(a_1, a_2)]$ values.

**Why this differs from the one-shot.** A one-shot g-comp regresses $Y$ on cumulative exposure plus $L$ once. When we ask "what would $Y$ be under always-high?", it plugs in the *observed* $L$ — but observed $L$ already reflects the damage from the *observed* treatment, partially cancelling the very effect we want to measure. The sequential ICE corrects this by replacing the observed $L_2$ with $g(a_1^{\\text{counterfactual}}, L_1)$ at every step. The difference between the two estimators tells you *how much the time-varying-confounder feedback matters* on this dataset."""),

md("""### Q4.2 Run the sequential ICE g-formula AND the one-shot g-comp; compare.

*Hint.* The ICE has two regressions: $L_2 \\sim A_1 + L_1$ (next-confounder model from Step 1 above) and $Y \\sim A_1 + L_1 + A_2 + L_2$ (terminal outcome from Step 3). For one-shot, just regress $Y$ on cumulative n_high (plus $L$ for the confounding adjustment).

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
# Sequential ICE
L2_model = LinearRegression().fit(panel[['A_1', 'L_1']].values, panel['L_2'].values)
Y_model = LinearRegression().fit(panel[['A_1', 'L_1', 'A_2', 'L_2']].values, panel['cycle_life'].values)

def E_Y_seq(a1, a2):
    L1 = panel['L_1'].values
    L2_cf = L2_model.predict(np.column_stack([np.full(len(panel), a1), L1]))
    Y_hat = Y_model.predict(np.column_stack([np.full(len(panel), a1), L1, np.full(len(panel), a2), L2_cf]))
    return float(Y_hat.mean())

ATE_seq = E_Y_seq(1, 1) - E_Y_seq(0, 0)
print(f'Sequential ICE ATE (always-high vs never-high): {ATE_seq:+.1f} cycles')

# One-shot g-comp
exposure = panel[['n_high_b1', 'n_high_b2', 'L_1', 'L_2']].copy()
exposure['cum_high'] = exposure['n_high_b1'] + exposure['n_high_b2']
exposure['cap_drop'] = exposure['L_1'] - exposure['L_2']
naive = LinearRegression().fit(exposure[['cum_high']].values, panel['cycle_life'].values)
g_one = LinearRegression().fit(exposure[['cum_high', 'cap_drop']].values, panel['cycle_life'].values)
mean_bin = panel.loc[panel['A_1']==1, 'n_high_b1'].mean() - panel.loc[panel['A_1']==0, 'n_high_b1'].mean()
one_shot_implied = float(g_one.coef_[0]) * mean_bin * 2
print(f'One-shot g-comp implied ATE (always-vs-never): {one_shot_implied:+.1f} cycles')
print(f'Disagreement: {ATE_seq - one_shot_implied:+.1f} cycles')
```
"""),

md("""**Discussion — interpret the *direction* of disagreement.**

Two pieces of evidence together pin down the right interpretation:

1. *Step-1 coefficient on $A_1 \\to L_2$ (printed in the ICE fit):* is it sizable or near-zero? This tells you whether past treatment shifts the next-period confounder *mechanistically* — i.e., whether the time-varying-confounding structure is meaningfully present in this dataset.
2. *Sign and magnitude of sequential − one-shot:* this is the empirical correction the sequential machinery applies on top of the cross-sectional estimate.

**Reading the cases:**

- **Sequential more negative than one-shot, with $A_1 \\to L_2$ coefficient sizable.** The one-shot estimator was *under-stating* the harm of high-drop exposure: it conditioned on the *observed* $L_2$ (which already reflects the damage from $A_1$), partially adjusting away the very effect we want to measure. The sequential ICE is the correct, larger estimate.
- **Sequential less negative than one-shot, with $A_1 \\to L_2$ coefficient sizable.** The one-shot estimator was *over-stating* the harm: it failed to credit the time-varying confounder with the part of the variance the sequential approach correctly assigns to $L_2$'s natural drift. The sequential ICE is the correct, smaller estimate.
- **Sequential ≈ one-shot, regardless of sign.** The chain is weak; treatment doesn't materially shift the next-period confounder; the one-shot estimator is a defensible approximation.

In every case, **the sequential ICE estimate is the principled headline number for the deployment decision**, with the one-shot serving as the simpler-but-coarser reference. The disagreement (whether sequential is *more* or *less* extreme) is the *diagnostic* about how much time-varying machinery actually buys you on this dataset."""),

md("""### Q4.3 Report the four-trajectory counterfactual table.

*Hint.* The ICE gives you $E[Y(a_1, a_2)]$ for all four $(a_1, a_2)$ combinations.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
counterfactual_table = pd.DataFrame({
    'trajectory':  ['(0,0) never high', '(1,0) early high only', '(0,1) late high only', '(1,1) always high'],
    'E[Y]':         [E_Y_seq(0, 0), E_Y_seq(1, 0), E_Y_seq(0, 1), E_Y_seq(1, 1)],
})
counterfactual_table['ATE_vs_never'] = counterfactual_table['E[Y]'] - counterfactual_table['E[Y]'].iloc[0]
print(counterfactual_table.round(1).to_string(index=False))
```
"""),

md("""## Artifact 5 — Sensitivity Analysis

### Q5.1 Sensitivity to the $A_1 \\to L_2$ model.

*Hint.* Perturb the coefficient of $A_1$ in the $L_2$ regression by a fraction of its residual std; re-run the ICE; report how the ATE moves.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
residual_L2 = panel['L_2'].values - L2_model.predict(panel[['A_1', 'L_1']].values)
res_std = float(np.std(residual_L2, ddof=1))
rng_sens = np.random.default_rng(1)
for sigma in [0.0, 0.05, 0.10, 0.20]:
    coef_perturbed = L2_model.coef_.copy()
    coef_perturbed[0] += rng_sens.normal(0, sigma * res_std)
    L2_p = LinearRegression(); L2_p.coef_ = coef_perturbed; L2_p.intercept_ = L2_model.intercept_
    def E_Y_p(a1, a2):
        L1 = panel['L_1'].values
        L2_cf = L2_p.predict(np.column_stack([np.full(len(panel), a1), L1]))
        return float(Y_model.predict(np.column_stack([np.full(len(panel), a1), L1, np.full(len(panel), a2), L2_cf])).mean())
    print(f'sigma={sigma:.2f}: ICE ATE = {E_Y_p(1,1) - E_Y_p(0,0):+.1f} cycles')
```
"""),

md("""### Q5.2 Verdict — robust, moderate, or fragile?

*Your turn.*"""),

md("""*[Verdict.]*"""),

reveal("""If the ATE direction is stable across $\\sigma \\in [0, 0.2]$ and the magnitude shifts by < 20%, the estimate is **robust**. If the direction is stable but magnitude varies > 20% — **moderate** (deploy with monitoring). If the direction flips at any $\\sigma \\leq 0.2$ — **fragile** (controlled trial before deployment)."""),

md("""## Artifact 6 — Deployment-Readiness Checklist

### Q6.1 Target population + transportability.

*Your turn.*"""),

md("""*[Target population + transportability statement.]*"""),

reveal("""Target: LFP cells of the same chemistry/format as the Severson cohort (commercial LiFePO4 18650), in the same fast-charge regime used during data collection (CC-CV up to a defined SOC threshold). Transportability to a different cell chemistry (e.g., NMC) requires a separate analysis; even within LFP, transportability to a different cell form factor (e.g., pouch instead of 18650) needs validation."""),

md("""### Q6.2 Three deployment monitors.

*Your turn.*"""),

md("""*[Three monitors with thresholds.]*"""),

reveal("""1. **Weekly per-protocol cycle_life distribution.** Alarm if the median cycle_life in the most-recent fortnight shifts by > 100 cycles from the deployment baseline. Could indicate the throttling policy is degrading other characteristics.
2. **Monthly $L_t \\to L_{t+1}$ residual check.** Re-fit the next-confounder model on the most-recent month of new cells; alarm if the $A_t \\to L_{t+1}$ coefficient changes sign. Time-varying-confounding structure may have changed.
3. **Quarterly sensitivity recompute.** Re-run Q5.1's perturbation sweep on the most-recent quarter's cells. Alarm if the *direction* becomes unstable at any $\\sigma < 0.10$.

Rollback: revert to the un-throttled fast-charge protocol via a single firmware-config update on the production charger."""),

md("""## Closing

Variants of this capstone:
- Replace the 2-bin sequential ICE with a 5-bin or 10-bin version; report whether the estimate converges as bins are refined.
- Run IPTW MSM alongside the ICE; the two should agree if both modelling steps are well-specified.
- Compute the analogous CATE by battery batch (Lab 6B's pattern): does the throttling-effect vary by Severson batch (train / test1 / test2)?

The time-varying machinery is the part of the course where most observational analyses go wrong — they treat trajectory data as cross-sectional. The capstone done well shows the disagreement between one-shot and sequential, and reports the sequential as the principled answer. Refer to [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the per-artifact standards."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14e_lfp.ipynb", cells)
print("Built lab14e_lfp.ipynb")
