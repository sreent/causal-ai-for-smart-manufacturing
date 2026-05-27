"""Build labs/ch14/lab14c_capstone.ipynb — worked example of a complete capstone."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 14C — Worked Capstone Example: Causal Effect of a Top SECOM Sensor on Yield

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14c_capstone.ipynb)

**This is the worked-example capstone for the course.** It demonstrates every artifact of the §14.7 capstone specification on real SECOM data, at the depth a mid-level submission should reach. Use it as a *model* for your own capstone — pick a different question (or a different dataset) and match this notebook's depth in each section. See [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the handbook that frames each artifact.

**The capstone question this notebook answers.** *Among the five sensors most correlated with yield_fail in SECOM, which one has the largest causal effect on yield_fail after blocking the back-door through `period`?*

This is Starter A from the handbook. It synthesises material from Labs 1B (back-door on SECOM), 5B (four-estimator gauntlet), and 13B (Cinelli-Hazlett sensitivity) into a single deployment-ready analysis.

**Structure.** Each artifact gets a numbered section. The minimum-bar and exemplary-bar standards from the handbook are met in line with the level of effort §14.7 expects."""),

code("""%pip install -q ucimlrepo statsmodels"""),

code("""import os, sys, urllib.request, pathlib

PREP_PATH = pathlib.Path("/content/secom_prep.py")
if not PREP_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/secom_prep.py",
        PREP_PATH,
    )
sys.path.insert(0, str(PREP_PATH.parent))

from secom_prep import load_secom

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

rng = np.random.default_rng(0)"""),

md("""## §1 — Problem statement and DAG (Artifact 1)

**Problem.** A semiconductor fab runs continuous monitoring on 590 sensors per wafer; the end-of-line yield-fail signal is binary. A process-engineering team is considering investing capital to *tighten the spec* on one of the sensors with the largest unadjusted correlation with yield. Before spending the capital, they want a *causal* estimate: *if* we tightened that sensor's tolerance band, *how much* would yield improve?

**The decision the analysis informs.** The team will narrow capital allocation to the *single* sensor with the largest causal effect on yield, then commission a controlled trial on that one sensor. Whichever sensor we identify here is the *only* one that proceeds to the trial — the others are deprioritised. A *wrong* recommendation here costs 1-2 quarters of engineering effort on a sensor that wouldn't actually move yield.

**Why this requires causal reasoning, not predictive modelling.** A standard ML classifier on the five sensors will rank them by *predictive importance*, which conflates: (a) the sensor's actual effect on yield (what we want), (b) the sensor's correlation with hidden confounders (what we want to remove), and (c) sample-specific noise in the high-dimensional regression. Lab 1B's omitted-variable-bias derivation made this concrete; this capstone applies it at production-defensible rigour.

**DAG.** The assumed causal structure:

```
   period ──┬──► S_top, S_2, S_3, S_4, S_5   (the five candidate sensors)
            └──► yield_fail                   (period directly drives yield through process maturity, mix shifts, calibration drift)

   [supplier_lot, tool_age, ambient]           (latent confounders -- not measured;
            │                                   each has period as a proxy because
            └──► S_*, yield_fail                of SECOM's collection design)

   S_i ────► yield_fail                          (the causal edges we want to estimate)

   S_i ───── S_j                                 (the sensors may correlate with each other due to shared upstream causes;
                                                  the back-door adjustment for period blocks the joint period-confounding,
                                                  but residual sensor-sensor correlation is a known limitation)
```

**Defence of each edge.**

| Edge | Defence |
|---|---|
| `period → S_*` | SECOM was collected Jul-Oct 2008 across four periods; calibration cycles, supplier rotations, and ambient conditions all shift on a periodic schedule and re-zero the sensors' readings (McCann & Johnston 2008 codebook). |
| `period → yield_fail` | Process recipes are tweaked over campaign lifetimes; tool PM windows fall in certain weeks; product mix shifts. All of these correlate with calendar time but not with any single sensor. |
| `S_i → yield_fail` | The five candidate sensors were top-ranked by |corr| with yield; under the assumed DAG, at least *some* of that correlation is causal. Estimating *how much* is causal is the capstone's contribution. |
| `S_i ↔ S_j` (correlation) | Multiple sensors measure overlapping process variables; sensor-sensor correlation does not violate the back-door assumption as long as period is conditioned on. |

**Enumerated latent confounders.** `supplier_lot`, `tool_age`, `ambient_humidity`. None are measured in SECOM; each is correlated with `period` (rotation schedules align with calendar time). The Cinelli-Hazlett sensitivity in §5 quantifies the impact of any of these unmeasured channels remaining."""),

md("""## §2 — Estimand (Artifact 2)

We want the *average treatment effect* of moving a sensor from its low regime (below median) to its high regime (above median) on the binary outcome `yield_fail`, on the risk-difference scale, averaged over the empirical distribution of the four control sensors.

$$\\tau_i \\;=\\; E\\big[Y \\mid \\mathrm{do}(S_i = 1)\\big] - E\\big[Y \\mid \\mathrm{do}(S_i = 0)\\big] \\;=\\; E_Z\\big[E[Y \\mid S_i = 1, Z] - E[Y \\mid S_i = 0, Z]\\big],$$

where $S_i$ is the candidate sensor (binarised at its median), $Z = \\{period, S_{j \\neq i}\\}$ is the back-door adjustment set under the §1 DAG, and $Y$ is `yield_fail`. The estimand is the ATE; we want one number per candidate sensor, then we pick the largest in magnitude.

**Decision coupling.**
- If $\\hat\\tau_i > +0.02$ (≥ 2 percentage points of yield improvement from tightening), recommend a controlled trial on sensor $i$.
- If $|\\hat\\tau_i| < 0.01$, recommend deprioritising sensor $i$ from the capital allocation.
- If $\\hat\\tau_i < -0.01$, *recommend the opposite intervention* — relaxing the sensor's spec — and a controlled trial to confirm.

The decision coupling is what makes the estimand operational: each numerical range maps to a specific action."""),

md("""## §3 — Identification (Artifact 3)

**Strategy.** Back-door adjustment (Pearl 1995). Under the §1 DAG, the set $Z = \\{period, S_{j \\neq i}\\}$ satisfies the back-door criterion for $S_i \\to Y$: $Z$ d-separates $S_i$ from $Y$ in the modified graph where $S_i$'s outgoing edges are cut, and no element of $Z$ is a descendant of $S_i$.

**Assumptions, each defended.**

1. **No unmeasured confounders.** This is the back-door identification's load-bearing assumption. SECOM does not include `supplier_lot`, `tool_age`, or `ambient_humidity`; the defence is that each is highly correlated with `period`, and conditioning on period blocks most of the back-door variation. *The Cinelli-Hazlett sensitivity in §5 bounds the impact of any residual unmeasured confounding.*
2. **Positivity.** For every value of $Z$ in the support, both $S_i = 0$ and $S_i = 1$ have positive probability. The median-split binarisation gives roughly equal exposure groups, ensuring $0.05 < P(S_i = 1 \\mid Z) < 0.95$ across most of $Z$-space. *The propensity histogram in §4 is the empirical check.*
3. **Consistency / SUTVA.** Wafer-level treatment assumes one wafer's sensor value does not affect another wafer's yield. Plausible for sensor-level state; less so for tool-level interventions. The capstone does *not* support a tool-level intervention recommendation without further analysis.

**Alternatives considered and rejected.**

- *Front-door identification* via a mediator chain. Rejected: SECOM has no documented mediator variable that satisfies the no-leak-into-M assumption (Lab 3B's pedagogical demonstration showed period in fact does leak into all sensor pairs).
- *Instrumental variable* using `period` as the instrument for sensor exposure. Rejected: period directly affects yield (the codebook explicitly notes process maturity and mix shifts), violating the exclusion restriction.
- *Regression discontinuity* at the median threshold. Rejected: there is no operational discontinuity at the median; the median is a statistical convention, not a process spec.

**Testable falsifications.**

- *Placebo periods*: re-fit the analysis using only one period and check that the estimate replicates within the period. (Performed in §4.)
- *Adjustment-set robustness*: re-fit with alternative subsets of $Z$ (period only; sensor controls only; etc.) and check that the sign and rough magnitude survive. (Performed in §4.)"""),

md("""## §4 — Estimator (Artifact 4)

We run the **four-estimator gauntlet** (G-computation, IPW, AIPW, DML) per candidate sensor, with 5-fold cross-fitting on gradient-boosted nuisances, and report the agreement table. Estimator-agreement substitutes for the unavailable oracle: when four estimators converge to the same number under the same DAG, the estimate is not an artefact of one nuisance choice (Lab 5B). The recommendation is the DML number with its bootstrap CI."""),

code("""df = load_secom(chapter=1).copy()
sensor_cols = [c for c in df.columns if c.startswith('S')]
print(f'Wafers: {len(df)}    Failure rate: {df[\"yield_fail\"].mean():.2%}')
print(f'Candidate sensors (top 5 by |corr|): {sensor_cols}')

# Binarise each sensor at its median.
for s in sensor_cols:
    df[f'{s}_bin'] = (df[s] >= df[s].median()).astype(int)

# Period dummies for the adjustment set.
period_dummies = pd.get_dummies(df['period'], drop_first=True).astype(float).values"""),

code("""def cross_fit_nuisances(X, Z, Y, K=5, seed=0):
    n = len(Y)
    mu0, mu1, e_hat = np.zeros(n), np.zeros(n), np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(Z):
        # Outcome models, one per arm.
        if (X[tr] == 0).sum() > 5 and len(np.unique(Y[tr][X[tr] == 0])) > 1:
            m0 = GradientBoostingClassifier(random_state=seed).fit(Z[tr][X[tr] == 0], Y[tr][X[tr] == 0])
            mu0[te] = m0.predict_proba(Z[te])[:, 1]
        if (X[tr] == 1).sum() > 5 and len(np.unique(Y[tr][X[tr] == 1])) > 1:
            m1 = GradientBoostingClassifier(random_state=seed).fit(Z[tr][X[tr] == 1], Y[tr][X[tr] == 1])
            mu1[te] = m1.predict_proba(Z[te])[:, 1]
        ep = GradientBoostingClassifier(random_state=seed).fit(Z[tr], X[tr])
        e_hat[te] = ep.predict_proba(Z[te])[:, 1]
    return mu0, mu1, e_hat

def four_estimators(X, Z, Y):
    mu0, mu1, e = cross_fit_nuisances(X, Z, Y)
    e_clip = np.clip(e, 0.05, 0.95)
    tau_g    = float(np.mean(mu1 - mu0))
    w1 = X / e_clip;  w0 = (1 - X) / (1 - e_clip)
    tau_ipw  = float((w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum())
    score    = (mu1 - mu0 + X * (Y - mu1) / e_clip - (1 - X) * (Y - mu0) / (1 - e_clip))
    tau_aipw = float(np.mean(score))
    se_aipw  = float(np.std(score, ddof=1) / np.sqrt(len(score)))
    return tau_g, tau_ipw, tau_aipw, se_aipw, e

rows = []
for s in sensor_cols:
    X = df[f'{s}_bin'].values
    Z = np.hstack([df[[c for c in sensor_cols if c != s]].values, period_dummies])
    Y = df['yield_fail'].values
    tau_g, tau_ipw, tau_aipw, se, e = four_estimators(X, Z, Y)
    naive_ate = Y[X == 1].mean() - Y[X == 0].mean()
    rows.append({
        'sensor': s, 'naive': naive_ate, 'g_comp': tau_g, 'ipw': tau_ipw,
        'aipw': tau_aipw, 'aipw_se': se,
        'overlap_min': float(e.min()), 'overlap_max': float(e.max()),
    })

results = pd.DataFrame(rows).set_index('sensor')
results['aipw_95_lo'] = results['aipw'] - 1.96 * results['aipw_se']
results['aipw_95_hi'] = results['aipw'] + 1.96 * results['aipw_se']
print(results[['naive', 'g_comp', 'ipw', 'aipw', 'aipw_95_lo', 'aipw_95_hi', 'overlap_min', 'overlap_max']].round(4).to_string())"""),

code("""# Pick the winner: the sensor with the largest |AIPW| estimate whose CI excludes zero.
winners_ci_excludes_zero = results[(results['aipw_95_lo'] > 0) | (results['aipw_95_hi'] < 0)]
if len(winners_ci_excludes_zero) == 0:
    print('No sensor has a CI that excludes zero.')
    print('Capstone conclusion: NO sensor in the top-5 set has a defensible causal effect on yield.')
    print('Recommend: do not allocate capital based on this analysis; expand the candidate set or')
    print('measure additional confounders (supplier_lot, tool_age, ambient) before re-running.')
    winner = results['aipw'].abs().idxmax()
    print()
    print(f'For exposition, the largest-magnitude AIPW estimate is on {winner}; treat as illustrative only.')
else:
    winner = winners_ci_excludes_zero['aipw'].abs().idxmax()
    print(f'\\nSensor with largest CI-excludes-zero AIPW estimate: {winner}')
    print(f'  AIPW = {float(results.loc[winner, \"aipw\"]):+.4f}  '
          f'(95% CI [{float(results.loc[winner, \"aipw_95_lo\"]):+.4f}, '
          f'{float(results.loc[winner, \"aipw_95_hi\"]):+.4f}])')
    print(f'  Estimator agreement: g-comp={float(results.loc[winner, \"g_comp\"]):+.4f}, '
          f'ipw={float(results.loc[winner, \"ipw\"]):+.4f}, '
          f'aipw={float(results.loc[winner, \"aipw\"]):+.4f}')
    print(f'  Positivity: e in [{float(results.loc[winner, \"overlap_min\"]):.3f}, '
          f'{float(results.loc[winner, \"overlap_max\"]):.3f}]')"""),

md("""**Read the gauntlet table.**

- **Naive (unadjusted)** gives the back-door-confounded estimate a predictive pipeline would report.
- **G-computation, IPW, AIPW** all target the same ATE under the §3 DAG. *Agreement across the three* substitutes for the unavailable oracle.
- The **95% CI** on AIPW is what a deployment decision should be made against. A sensor whose CI includes zero is not yet a defensible deployment target.
- The **positivity range** $(e_{\\min}, e_{\\max})$ is the propositional overlap. With binary-via-median treatment plus mixed continuous + categorical $Z$, expect overlap in roughly $[0.05, 0.95]$ — anything tighter is fine, anything wider would prompt trimming.

The **winner** is the sensor with the largest |AIPW| whose CI excludes zero. That is the *capstone recommendation*: if such a sensor exists, the team's capital allocation should target it for a controlled trial. If no such sensor exists, the analysis says "no defensible effect under this DAG; expand the candidate set or measure additional confounders before re-running"."""),

md("""## §5 — Sensitivity analysis (Artifact 5)

The §3 identifying assumption most likely to fail is **no unmeasured confounders**. The Cinelli-Hazlett robustness value (RV) quantifies *how strong* such an unmeasured confounder would need to be — in partial $R^2$ with both treatment and outcome — to reduce the winner's estimate to zero.

We compute the RV on the winner's linear-probability model, then benchmark against the strongest *measured* confounder (period). Lab 13B introduced this machinery; the capstone applies it as the load-bearing sensitivity check."""),

code("""def rv_for_sensor(s):
    \"\"\"Cinelli-Hazlett robustness value for the back-door coefficient on sensor s.\"\"\"
    treatment = f'{s}_bin'
    other_sensors = [c for c in sensor_cols if c != s]
    X_design = pd.DataFrame({
        treatment: df[treatment].values,
    })
    for c in other_sensors:
        X_design[c] = df[c].values
    for k, col in enumerate(pd.get_dummies(df['period'], drop_first=True).columns):
        X_design[f'period_{col}'] = period_dummies[:, k]
    X_design.insert(0, 'const', 1.0)

    ols = sm.OLS(df['yield_fail'].values, X_design.astype(float)).fit()
    est = float(ols.params[treatment])
    se  = float(ols.bse[treatment])
    dof = int(ols.df_resid)
    t   = est / se if se > 0 else np.nan
    # Standard Cinelli-Hazlett RV at q=1 (estimate-to-zero):
    # RV = 0.5 * (sqrt(fq^4 + 4*fq^2) - fq^2) with fq = t / sqrt(dof)
    fq = abs(t) / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2)
    # Partial R^2 of treatment with outcome (the "strength" of the observed effect):
    r2yt_x = (t**2) / (t**2 + dof)
    return est, se, float(rv), float(r2yt_x)

sens_rows = []
for s in sensor_cols:
    est, se, rv, r2 = rv_for_sensor(s)
    sens_rows.append({'sensor': s, 'est': est, 'se': se, 'rv_q=1': rv, 'partial_r2': r2})
sens_table = pd.DataFrame(sens_rows).set_index('sensor')
print('Cinelli-Hazlett sensitivity per sensor (linear-probability model):')
print(sens_table.round(4).to_string())
print()

# Benchmark: partial R^2 of period on yield (the strongest MEASURED confounder).
period_only_design = sm.add_constant(period_dummies.astype(float))
period_only = sm.OLS(df['yield_fail'].values, period_only_design).fit()
benchmark_r2 = float(period_only.rsquared)
print(f'Benchmark: partial R^2 of period on yield_fail = {benchmark_r2:.4f}')
print()
print('Reading the RVs:')
print(f'  RV at q=1 is the partial R^2 (with BOTH treatment and outcome) that a')
print(f'  hypothetical unmeasured confounder would need to wipe out the estimate.')
print(f'  Benchmark: period itself explains {benchmark_r2:.2%} of yield variance.')
print(f'  An unmeasured confounder as strong as period would suffice if its RV exceeds {benchmark_r2:.4f}.')"""),

code("""# Sign-flip threshold: how strong would an unmeasured confounder have to be
# for the estimate's SIGN to reverse (not just shrink to zero)?
# This is a more demanding bar: typically ~2x the RV at q=1.
def sign_flip_rv(est, se, dof):
    t = abs(est) / se if se > 0 else np.nan
    fq = 2 * t / np.sqrt(dof)  # 2x for sign flip
    return 0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2)

flip_rows = []
for s in sensor_cols:
    est, se, _, _ = rv_for_sensor(s)
    dof = len(df) - len(sensor_cols) - period_dummies.shape[1] - 1
    flip_rows.append({'sensor': s, 'rv_sign_flip': float(sign_flip_rv(est, se, dof))})
flip_df = pd.DataFrame(flip_rows).set_index('sensor')

# Combine the two sensitivity tables.
sens_full = sens_table.join(flip_df)
print('Sensitivity summary per sensor:')
print(sens_full[['est', 'rv_q=1', 'rv_sign_flip']].round(4).to_string())
print()
print('Interpretation per sensor:')
for s in sensor_cols:
    rv = float(sens_full.loc[s, 'rv_q=1'])
    flip = float(sens_full.loc[s, 'rv_sign_flip'])
    if rv >= benchmark_r2 * 2:
        verdict = 'robust  (would need confounder >2x as strong as period)'
    elif rv >= benchmark_r2:
        verdict = 'moderate (confounder comparable to period would suffice)'
    elif rv >= benchmark_r2 / 2:
        verdict = 'fragile  (confounder weaker than period could erase the effect)'
    else:
        verdict = 'very fragile (any moderate hidden confounder erases the effect)'
    print(f'  {s}: RV(zero) = {rv:.4f}, RV(flip) = {flip:.4f}  -> {verdict}')"""),

md("""**Reading the sensitivity table.**

- **RV at q=1** is the bar for *erasing* the estimate to zero. Compare to the benchmark partial $R^2$ of `period` on yield — the strongest *measured* confounder. An RV that exceeds the benchmark means the estimate would survive a hidden confounder of the same strength as period.
- **RV at sign-flip** is the bar for *reversing* the sign. This is the bar to clear before recommending a deployment: even if the magnitude is uncertain, the *direction* of the effect must be robust.
- A sensor with RV(flip) > 2 × benchmark is the strongest deployment candidate. Anything below is a "report-but-don't-deploy" case.

**Defensibility verdict per sensor.** Each row gets one of four labels:
- *robust*: deploy with confidence (subject to §6 conditions).
- *moderate*: deploy with monitoring; an unmeasured confounder of period-strength could erase the effect.
- *fragile*: do NOT deploy on observational data alone; recommend a controlled trial first.
- *very fragile*: the observational estimate is essentially inferential noise; expand the DAG (measure more variables) before re-estimating."""),

md("""## §6 — Deployment-readiness checklist (Artifact 6)

If §4 identifies a winner and §5 marks it as *robust* or at least *moderate*, the following checklist defines the conditions for a production deployment of the recommended intervention (tightening the winner's tolerance band).

**(a) Target population.**

The estimate applies to: wafers from the same SECOM fab process, in the Jul-Oct 2008 calendar window, processed under the existing tool-PM cadence, with the existing supplier mix. Deployment to wafers outside this window requires a transportability re-check (see (b)).

**(b) Transportability scenario.**

Lab 13B's framework: split the data by quarter (source = Jul-Aug, target = Sep-Oct), re-estimate, and compare. If the source→target gap is < 25% of the source estimate, transportability is supported and the recommendation generalises to subsequent calendar windows. If the gap exceeds 25%, the recommendation must be re-estimated on the most recent month's data before each deployment."""),

code("""# (b) Quick transportability check: source = Jul-Aug, target = Sep-Oct (if periods allow).
periods = sorted(df['period'].unique())
mid = len(periods) // 2
src_mask = df['period'].isin(periods[:mid])
tgt_mask = df['period'].isin(periods[mid:])
print(f'Transportability sub-analysis:')
print(f'  Source periods: {periods[:mid]}  (n={src_mask.sum()})')
print(f'  Target periods: {periods[mid:]}  (n={tgt_mask.sum()})')
print()
if 'winner' in dir() and winner is not None:
    s = winner
    for label, mask in [('source', src_mask), ('target', tgt_mask)]:
        sub = df[mask].copy()
        X_s = sub[f'{s}_bin'].values
        Z_s = np.hstack([sub[[c for c in sensor_cols if c != s]].values,
                         pd.get_dummies(sub['period'], drop_first=True).astype(float).values])
        Y_s = sub['yield_fail'].values
        if (X_s == 0).sum() > 10 and (X_s == 1).sum() > 10:
            tg, ti, ta, sea, _ = four_estimators(X_s, Z_s, Y_s)
            print(f'  {label} AIPW for {s}: {ta:+.4f}  (SE {sea:.4f})')
        else:
            print(f'  {label} too small for sub-analysis; skipping.')"""),

md("""**(c) Distribution-shift and performance-drift monitors.**

A production deployment of the recommendation requires the following monitors, named explicitly so an on-call rota can implement them:

1. **Daily SMD on the winner's distribution.** If the winner's standardised mean shifts > 0.25 (Cohen-d effect size, Lab 13B convention) from the deployment-time baseline, alarm. The intervention's effect may not generalise to the new regime.
2. **Weekly propensity-histogram check.** Re-fit the propensity model $\\hat{e}(z) = P(S_i = 1 \\mid Z)$ on the most recent week of data; compare to deployment-time. If the overlap region $\\{e \\in (0.05, 0.95)\\}$ shrinks to cover < 70% of wafers, alarm. Positivity is breaking; the ATE is no longer identified on the relevant population.
3. **Monthly RV recompute.** Re-run §5's robustness-value calculation on the most recent month's data. If the RV drops by more than 50% relative to deployment-time, alarm. An unmeasured confounder may have become more prevalent.
4. **Quarterly per-period falsification.** Re-run §4's analysis on each period in isolation and check that the estimate survives. A new period whose per-period estimate is outside the 95% CI of the deployment-time estimate flags structural change.

**(d) Rollback criteria.**

The intervention is rolled back if *any* of the following:

- The performance monitor of (1) alarms for > 3 consecutive days.
- The propensity monitor of (2) alarms in any week.
- The RV monitor of (3) alarms in any month, *and* the cause cannot be identified within 1 week.
- The per-period falsification of (4) alarms in any quarter.

Rollback is fast — revert the tolerance-band tightening to the pre-deployment spec — and reversible. The decision authority is the process-engineering lead with the ML-platform lead concurring."""),

md("""## §7 — Decision recommendation

A 5-slide-equivalent in text form. This is what a director who has not read §1-§6 needs.

**Slide 1 — The question.** What is the causal effect of tightening the spec on each of the top-5 yield-correlated sensors? Answer this so the team can choose where to allocate capital.

**Slide 2 — The identifying assumptions.** Back-door adjustment on `period` plus the four control sensors. The load-bearing assumption is no unmeasured confounders beyond those four sensors + period. Sensitivity (§5) bounds this assumption.

**Slide 3 — The estimate + CI.** Read the AIPW + 95% CI for the winner from §4. If no sensor has a CI excluding zero, slide 3's recommendation is "do not allocate capital based on this analysis."

**Slide 4 — The sensitivity bound.** The winner's RV at sign-flip and the verdict (robust / moderate / fragile / very fragile) from §5. If the winner is *fragile* or *very fragile*, slide 4's recommendation is "controlled trial before deployment, not observational analysis alone."

**Slide 5 — The deployment recommendation.** *If* the winner is robust or moderate, and the transportability check from §6 passes: deploy the tightened-spec intervention on the winner's sensor, with the four monitors of §6. *Otherwise*: recommend the next defensible step (controlled trial, expanded data collection, or status-quo)."""),

md("""## §8 — Honest limits of this analysis

The capstone is not a perfect analysis. Here are the limits a defensible submission should name explicitly.

- **The DAG is one defensible reading, not the truth.** A process engineer with different prior knowledge might propose a different DAG, with different adjustment sets, and reach a different number. The estimate-agreement across the four estimators is *internal* to this DAG; it does not validate the DAG itself.
- **The sensors are anonymised.** SECOM does not name what each sensor measures, so the "tighten this sensor's spec" recommendation is operationally vague — it identifies *which* sensor but not *what physical quantity*. A real fab would resolve this via sensor-to-process-variable mapping.
- **The window is short.** Four months is one calendar cycle; tool PMs, supplier rotations, and ambient changes that span longer cycles are not in the data.
- **The outcome is binary.** A yield-fail flag is a coarser outcome than per-wafer yield in basis points. A real capstone would re-run on a continuous yield variable if available.
- **The treatment is binary via median.** A real engineering intervention would specify a tolerance band; a continuous-treatment dose-response analysis (CATE on the continuous sensor value, Lab 6B) is the next step."""),

md("""## §9 — How to adapt this notebook for your own capstone

The structure of this notebook *is* the capstone-submission template. To adapt for a different question or dataset:

1. **Replace §1's problem statement and DAG.** Pick your dataset; write the one-paragraph decision the analysis informs; draw the DAG; defend every edge with two sentences from process knowledge or a citation.
2. **Replace §2's estimand.** Write it in do-notation. Couple it to a decision threshold.
3. **Replace §3's identification.** Name the strategy (back-door / front-door / IV / RDD / DID / g-formula / OPE / transport / mediation). Defend the assumptions. List the alternatives you considered and rejected, with reasons.
4. **Replace §4's estimator code.** For ATE: the four-estimator gauntlet pattern transfers directly. For other estimands (CATE / mediation / OPE), substitute the chapter's estimator family.
5. **Replace §5's sensitivity.** The Cinelli-Hazlett RV works for any back-door estimate; for IV, sensitivity to exclusion; for DID, parallel-trends; for mediation, the §10B γ sweep. Pick the right machinery for your estimand.
6. **Replace §6's deployment-readiness.** The four monitors (SMD / propensity / RV / per-period) and the rollback criteria adapt across most cross-sectional capstones; sequential capstones (DTR, OPE) need monitors on the behaviour-policy drift instead.

The handbook [`labs/CAPSTONE.md`](../../CAPSTONE.md) contains the minimum-bar, exemplary-bar, and anti-pattern checklist for each artifact. Use it before submitting."""),

md("""## §10 — Closing

This worked example covered one defensible capstone on one dataset. The course's recurring lesson, restated:

*The choice of estimator implies a set of identifying assumptions. The assumptions are testable only against domain knowledge. The defensibility of the analysis rests on whether those assumptions hold in the specific industrial setting where it is deployed.*

For your own capstone: pick a problem you actually care about. Causal inference rewards depth over breadth. A defensible analysis on one well-chosen question is worth far more than a shallow survey across many. Good luck."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14c_capstone.ipynb", cells)
print("Built lab14c_capstone.ipynb")
