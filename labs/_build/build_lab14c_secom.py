"""Build labs/ch14/lab14c_secom.ipynb — guided capstone Starter A on SECOM."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402


def reveal(content: str) -> dict:
    """A click-to-reveal sample answer block, rendered as a Jupyter-compatible HTML <details>."""
    return md("<details>\n<summary><b>Click to reveal sample answer</b></summary>\n\n"
              + content + "\n\n</details>")


cells = [

md("""# Lab 14C — Guided Capstone (Starter A): Effect of a Top SECOM Sensor on Yield

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14c_secom.ipynb)

This is **Starter A** of the §14.7 capstone. The notebook *guides you through* each of the six capstone artifacts as a series of questions, with click-to-reveal sample answers and discussion. The deliverable is your own end-to-end capstone analysis on SECOM, structured like this notebook — produced by *writing the cells*, not just reading them.

**The capstone question for this starter.** *Among the five sensors most correlated with yield_fail in SECOM, which one has the largest causal effect on yield_fail after blocking the back-door through `period`?* If the analysis identifies a defensible winner, the team's capital allocation targets it for a controlled trial. If no defensible winner exists, the analysis recommends expanding the candidate set or measuring additional confounders.

**Who this is for.** A learner who has completed Labs 1B-14B and wants to produce a defensible capstone deliverable on the dataset they know best (SECOM). The matching handbook is [`labs/CAPSTONE.md`](../../CAPSTONE.md).

**How to use this notebook.**
1. Read each question and the hint.
2. Write your answer in the *Your turn* cell — code or markdown.
3. Click *Reveal sample answer* to compare.
4. Read the discussion to see *why* the sample answer made the choices it did.

The sample answers are *one* defensible solution per question. A different answer can be equally defensible if you can defend it — that is the entire point of the capstone."""),

md("""## Setup"""),

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

rng = np.random.default_rng(0)
df = load_secom(chapter=1).copy()
sensor_cols = [c for c in df.columns if c.startswith('S')]
print(f'Wafers: {len(df)}    Failure rate: {df[\"yield_fail\"].mean():.2%}')
print(f'Candidate sensors: {sensor_cols}')"""),

md("""---

## Artifact 1 — Problem Statement and DAG

### Q1.1 Write the one-paragraph problem statement.

*What it should contain:* the manufacturing decision the analysis informs, who would act on the answer, and a sentence explaining why a causal analysis (not predictive ML) is the right tool for the question. Aim for 80-150 words.

*Hint.* Look at Lab 1B's framing. Your statement should end with a sentence stating *who acts on the answer* — vague statements like "we want to improve yield" cannot drive a defensible analysis.

*Your turn — write your problem statement in the markdown cell below.*"""),

md("""*[Write your problem statement here.]*"""),

reveal("""A semiconductor fab runs continuous monitoring on 590 sensors per wafer; the end-of-line yield-fail signal is binary. A process-engineering team is considering investing capital to *tighten the spec* on one of the sensors with the largest unadjusted correlation with yield. Before spending the capital, they want a *causal* estimate: *if* we tightened that sensor's tolerance band, *how much* would yield improve?

The team will narrow capital allocation to the *single* sensor with the largest causal effect on yield, then commission a controlled trial on that one sensor. A *wrong* recommendation here costs 1-2 quarters of engineering effort on a sensor that wouldn't actually move yield."""),

md("""**Discussion.** A good problem statement is *specific* about (a) the data, (b) the decision, and (c) the team who acts. The "wrong recommendation costs 1-2 quarters" sentence anchors the *stakes* — it tells the analyst when to stop tuning the estimator and ship. If your statement says only "we want to find sensors that matter", you cannot defend any stopping rule because the decision was never sharp."""),

md("""### Q1.2 Draw the DAG over the variables in this analysis.

*Variables to include:* the five candidate sensors (treatment candidates), `period` (the observed confounder), `yield_fail` (outcome), and any latent confounders you want to name.

*Your turn — draw the DAG below (ASCII art is fine).*"""),

md("""*[Draw your DAG here. ASCII art:*

```
   ?
```

*]*"""),

reveal("""```
   period ──┬──► S_top, S_2, S_3, S_4, S_5   (the five candidate sensors)
            └──► yield_fail                   (period drives yield through process maturity / mix shifts / calibration drift)

   [supplier_lot, tool_age, ambient]          (LATENT — not in data; each correlates with period)
            │
            └──► S_*, yield_fail

   S_i ────► yield_fail                        (the causal edges we want to estimate)

   S_i ───── S_j                               (sensors correlate via shared upstream causes;
                                                back-door adjustment for period blocks the JOINT
                                                period-confounding)
```
"""),

md("""**Discussion.** Three patterns matter here. (1) `period` is a *common cause* — it confounds every sensor → yield edge. That makes it a back-door variable; conditioning on it identifies the direct effect under the assumed DAG. (2) The bracketed nodes are *latent* — not measurable in SECOM but plausibly real (supplier rotations, tool age, ambient drift). Sensitivity analysis in Artifact 5 will bound them. (3) Sensor-sensor correlation does NOT violate back-door identification as long as we condition on the common parent (`period`); the other sensors can be IN the adjustment set as additional controls."""),

md("""### Q1.3 Defend three of the DAG's edges from process knowledge.

For each edge, write a two-sentence justification. The currency of capstone defense is *citation or named mechanism*, not assertion.

*Your turn — defend three edges below.*"""),

md("""*[Defend three edges, two sentences each.]*"""),

reveal("""**`period → sensor_i`.** SECOM was collected Jul-Oct 2008 across four periods; calibration cycles re-zero each sensor's reading on a periodic schedule, and supplier rotations shift the upstream chemistry. Both mechanisms are documented in the SECOM codebook (McCann & Johnston 2008, §2).

**`period → yield_fail`.** Process recipes are tweaked over campaign lifetimes; tool preventive-maintenance windows fall in specific weeks; product mix can shift between months. None of these is driven by any individual sensor, so the period → yield edge is direct, not mediated by sensors.

**`S_i → yield_fail`.** Each candidate sensor measures a physical quantity (anonymised in SECOM but real); at least *some* of the observed sensor-yield correlation is causal under any plausible chemistry. Whether it is *enough* to be worth a capital allocation is what the estimator in Artifact 4 will quantify."""),

md("""**Discussion.** "Documented in the SECOM codebook" beats "we assume" every time. When you cannot point to a citation, the next-best move is to *name the physical mechanism* — calibration cycles, supplier rotations, mix shifts. If a defense reads like "our team thought so", a domain reviewer will reject it; if it reads like "the calibration-cycle re-zeroes the sensor on a 30-day schedule (codebook §2)", they will engage with it.

*Your defense does not need to match the reveal verbatim.* The reveal cites specific sources because that is the *highest-rigour* version of an edge defense. A defense written in operator-vocabulary — "calibration drifts every 30 days because the laboratory recalibrates the gauges between maintenance shifts" — without a paper citation is *equally valid* for the capstone if it names a real mechanism. The capstone rewards *defensible*, not *cited*."""),

md("""---

## Artifact 2 — Estimand

### Q2.1 Write the estimand in do-notation.

*What to specify:* the treatment (binary), the adjustment set Z, and the outcome.

*Hint.* For binary treatment + binary outcome, the ATE on the risk-difference scale is conventional. Write it as a difference of expectations under do().

*Your turn — write your estimand below.*"""),

md("""*[Write the estimand in do-notation.]*"""),

reveal("""$$\\tau_i \\;=\\; E\\big[Y \\mid \\mathrm{do}(S_i = 1)\\big] - E\\big[Y \\mid \\mathrm{do}(S_i = 0)\\big] \\;=\\; E_Z\\big[E[Y \\mid S_i = 1, Z] - E[Y \\mid S_i = 0, Z]\\big],$$

where $S_i$ is the candidate sensor binarised at its sample median, $Z = \\{\\text{period}, S_{j \\neq i}\\}$, and $Y$ is `yield_fail`. The estimand is the ATE on the risk-difference scale."""),

md("""### Q2.2 Couple the estimand to a decision threshold.

A bare number does not drive action. Write a 3-bullet decision rule mapping numerical ranges of $\\hat\\tau_i$ to specific recommendations.

*Your turn — write your decision rule.*"""),

md("""*[Write your decision-threshold mapping (3 bullets).]*"""),

reveal("""- $\\hat\\tau_i > +0.02$ (≥ 2 percentage points of yield improvement): **recommend a controlled trial** on sensor $i$'s spec-tightening intervention.
- $-0.01 < \\hat\\tau_i < +0.02$: **deprioritise** sensor $i$ from the capital allocation; effect is too small to justify the engineering cost.
- $\\hat\\tau_i < -0.01$: **recommend the opposite intervention** (relaxing the spec) and a controlled trial to confirm the unexpected direction."""),

md("""**Discussion.** Coupling to a decision threshold is what makes the estimand *operational*. Without it, the analyst doesn't know when to stop polishing the estimator — every direction looks like "more rigour is better". With it, the question becomes binary: did the CI cross the +0.02 threshold or not? A wide CI that includes the threshold means "more data needed", not "more estimators needed"."""),

md("""---

## Artifact 3 — Identification

### Q3.1 Name the identification strategy and write the adjustment set.

*Hint.* Look at the DAG you drew in Q1.2. Which set of variables blocks every back-door path from $S_i$ to $Y$?

*Your turn — name the strategy and the adjustment set.*"""),

md("""*[Name the strategy and Z.]*"""),

reveal("""**Strategy.** Back-door adjustment (Pearl 1995). Under the §1 DAG, the set $Z = \\{period, S_{j \\neq i}\\}$ satisfies the back-door criterion for $S_i \\to Y$: $Z$ d-separates $S_i$ from $Y$ in the modified graph where $S_i$'s outgoing edges are cut, and no element of $Z$ is a descendant of $S_i$."""),

md("""### Q3.2 Defend each identifying assumption against a known SECOM limitation.

Three assumptions to defend: no unmeasured confounders, positivity, SUTVA/consistency.

*Hint.* Be honest about what could fail. The capstone rewards *named* fragility paired with a sensitivity entry, not assertions of robustness.

*Your turn — defend the three assumptions.*"""),

md("""*[Defend NUC, positivity, SUTVA — one paragraph each.]*"""),

reveal("""**No unmeasured confounders.** Load-bearing. SECOM does not include `supplier_lot`, `tool_age`, or `ambient_humidity`; the defence is that each correlates with `period`, and conditioning on period blocks most of the back-door variation. *The Cinelli-Hazlett sensitivity in Artifact 5 bounds the residual.*

**Positivity.** For every value of $Z$, both $S_i = 0$ and $S_i = 1$ have positive probability. The median-split binarisation gives roughly equal exposure groups, so $0.05 < P(S_i = 1 \\mid Z) < 0.95$ across most of $Z$-space. *The propensity histogram in Artifact 4 is the empirical check.*

**Consistency / SUTVA.** Wafer-level treatment assumes one wafer's sensor value does not affect another wafer's yield — plausible for sensor-level state; less so for tool-level interventions. *The capstone does not support a tool-level intervention recommendation without further analysis.*"""),

md("""### Q3.3 Consider an alternative identification strategy and reject it.

*Hint.* Front-door (Lab 3B's strategy), IV using period, or regression discontinuity. Pick one, state why it doesn't apply here.

*Your turn — name an alternative and reject it.*"""),

md("""*[Name an alternative; reject it with a reason.]*"""),

reveal("""**Front-door identification** via a mediator chain. *Rejected:* SECOM has no documented mediator variable that satisfies the *no leak into M* assumption. Lab 3B explicitly showed (on the same data) that period leaks into all sensor pairs, so any candidate mediator built from sensors fails the front-door criterion.

*(Alternative rejections: IV using period — period directly affects yield via process-maturity, violating exclusion. RDD at the median — the median is a statistical convention, not an operational discontinuity.)*"""),

md("""---

## Artifact 4 — Estimator

### Q4.1 Set up the data: binarise each candidate sensor and build the adjustment array.

*Your turn — write the code below.*"""),

code("""# YOUR CODE HERE
# Binarise each sensor at its median; build the period dummies for Z.

"""),

reveal("""```python
# Binarise each sensor at its median.
for s in sensor_cols:
    df[f'{s}_bin'] = (df[s] >= df[s].median()).astype(int)

# Period dummies for the adjustment set.
period_dummies = pd.get_dummies(df['period'], drop_first=True).astype(float).values
print(f'Treatment rates after binarisation:')
print(df[[f'{s}_bin' for s in sensor_cols]].mean().round(3).to_string())
```
"""),

md("""**Discussion.** Median split produces roughly balanced exposure groups (P(X=1) ≈ 0.5), which *helps* but does not *guarantee* positivity — overlap still depends on the joint distribution of (X, Z). The explicit guarantee comes from the `np.clip(e, 0.05, 0.95)` step in Q4.2, which bounds the propensity away from {0, 1} before computing IPW weights. The `drop_first=True` in the dummy encoding avoids perfect collinearity in the regression."""),

md("""### Q4.2 Implement the four-estimator gauntlet (G-comp, IPW, AIPW, DML).

*Hint.* Reuse the cross-fit nuisance pattern from Lab 5B: 5-fold KFold, one outcome model per arm, one propensity model, OOF predictions.

*Your turn — implement.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
def cross_fit_nuisances(X, Z, Y, K=5, seed=0):
    n = len(Y)
    mu0, mu1, e_hat = np.zeros(n), np.zeros(n), np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(Z):
        if (X[tr] == 0).sum() > 5 and len(np.unique(Y[tr][X[tr] == 0])) > 1:
            m0 = GradientBoostingClassifier(random_state=seed).fit(Z[tr][X[tr]==0], Y[tr][X[tr]==0])
            mu0[te] = m0.predict_proba(Z[te])[:, 1]
        if (X[tr] == 1).sum() > 5 and len(np.unique(Y[tr][X[tr] == 1])) > 1:
            m1 = GradientBoostingClassifier(random_state=seed).fit(Z[tr][X[tr]==1], Y[tr][X[tr]==1])
            mu1[te] = m1.predict_proba(Z[te])[:, 1]
        ep = GradientBoostingClassifier(random_state=seed).fit(Z[tr], X[tr])
        e_hat[te] = ep.predict_proba(Z[te])[:, 1]
    return mu0, mu1, e_hat

def four_estimators(X, Z, Y):
    mu0, mu1, e = cross_fit_nuisances(X, Z, Y)
    e_clip = np.clip(e, 0.05, 0.95)
    tau_g = float(np.mean(mu1 - mu0))
    w1 = X / e_clip; w0 = (1 - X) / (1 - e_clip)
    tau_ipw = float((w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum())
    score = (mu1 - mu0 + X * (Y - mu1) / e_clip - (1 - X) * (Y - mu0) / (1 - e_clip))
    return tau_g, tau_ipw, float(np.mean(score)), float(np.std(score, ddof=1) / np.sqrt(len(score))), e
```
"""),

md("""### Q4.3 Run the gauntlet for all five candidate sensors and pick the winner.

*Hint.* Loop over `sensor_cols`. For each, build $Z$ as "all other sensors + period dummies". Report estimates + 95% CI. The *winner* is the sensor whose AIPW CI excludes zero with the largest |estimate|.

*Your turn — run the loop.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
rows = []
for s in sensor_cols:
    X = df[f'{s}_bin'].values
    Z = np.hstack([df[[c for c in sensor_cols if c != s]].values, period_dummies])
    Y = df['yield_fail'].values
    tau_g, tau_ipw, tau_aipw, se, e = four_estimators(X, Z, Y)
    rows.append({'sensor': s, 'g_comp': tau_g, 'ipw': tau_ipw, 'aipw': tau_aipw,
                 'aipw_se': se, 'lo': tau_aipw - 1.96*se, 'hi': tau_aipw + 1.96*se})

results = pd.DataFrame(rows).set_index('sensor')
print(results.round(4).to_string())

# Pick the winner: largest |AIPW| with CI excluding zero.
winners = results[(results['lo'] > 0) | (results['hi'] < 0)]
if len(winners) == 0:
    print('No sensor has a CI excluding zero. Capstone answer: no defensible winner.')
    winner = None
else:
    winner = winners['aipw'].abs().idxmax()
    print(f'\\nWinner: {winner}  (AIPW = {results.loc[winner, \"aipw\"]:+.4f})')
```
"""),

md("""**Discussion.** *Estimator agreement* across G-comp / IPW / AIPW substitutes for the unavailable oracle. If the three differ wildly on a sensor, the nuisance fits are unstable; if they cluster, the estimate is consistent across modelling choices. A CI that excludes zero is the bar for "defensible winner". A CI that includes zero — even with a large point estimate — means "more data needed before deployment", not "ship the recommendation"."""),

md("""---

## Artifact 5 — Sensitivity Analysis

### Q5.1 Compute the Cinelli-Hazlett robustness value for the winner.

*Hint.* The robustness value (RV) is the partial-$R^2$ a hypothetical unmeasured confounder would need to have with *both* the treatment and the outcome to wipe out the estimate (Cinelli & Hazlett 2020 / Lab 13B). At $q=1$ (estimate-to-zero), $\\mathrm{RV} = 0.5 (\\sqrt{f^4 + 4f^2} - f^2)$ with $f = |t| / \\sqrt{\\mathrm{dof}}$ from a linear-probability model fit on $(Y, T, Z)$. A larger RV means the estimate survives a more powerful hidden confounder.

*Your turn — compute the RV for the winner.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
if winner is not None:
    treatment = f'{winner}_bin'
    other = [c for c in sensor_cols if c != winner]
    X_design = pd.DataFrame({treatment: df[treatment].values})
    for c in other:
        X_design[c] = df[c].values
    for k, col in enumerate(pd.get_dummies(df['period'], drop_first=True).columns):
        X_design[f'period_{col}'] = period_dummies[:, k]
    X_design.insert(0, 'const', 1.0)
    ols = sm.OLS(df['yield_fail'].values, X_design.astype(float)).fit()
    est, se = float(ols.params[treatment]), float(ols.bse[treatment])
    dof = int(ols.df_resid)
    f = abs(est / se) / np.sqrt(dof)
    rv = 0.5 * (np.sqrt(f**4 + 4*f**2) - f**2)
    print(f'Winner {winner}: estimate {est:+.4f}, SE {se:.4f}, RV(q=1) {rv:.4f}')
```
"""),

md("""### Q5.2 Benchmark the RV against the strongest measured confounder.

*Hint.* Compute the partial $R^2$ of `period` on `yield_fail` from a period-only regression. An unmeasured confounder "as strong as period" needs $\\mathrm{RV} \\geq$ this benchmark.

*Your turn — benchmark.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
period_only = sm.OLS(df['yield_fail'].values, sm.add_constant(period_dummies.astype(float))).fit()
benchmark_r2 = float(period_only.rsquared)
print(f'Benchmark: partial R^2 of period on yield = {benchmark_r2:.4f}')
print(f'Winner survives an unmeasured confounder as strong as period iff RV >= {benchmark_r2:.4f}.')
```
"""),

md("""### Q5.3 Compute the sign-flip RV (a stricter bar) and write the verdict.

*Hint.* Sign-flip uses $f = 2|t|/\\sqrt{\\mathrm{dof}}$ (twice as strong). Map RV to a verdict: *robust* (RV > 2×benchmark), *moderate* (RV > benchmark), *fragile* (< benchmark), *very fragile* (< benchmark/2).

*Your turn — compute and label.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
f_flip = 2 * abs(est / se) / np.sqrt(dof)
rv_flip = 0.5 * (np.sqrt(f_flip**4 + 4*f_flip**2) - f_flip**2)
print(f'RV(sign-flip) = {rv_flip:.4f}')
if rv >= benchmark_r2 * 2:    verdict = 'robust'
elif rv >= benchmark_r2:       verdict = 'moderate'
elif rv >= benchmark_r2 / 2:   verdict = 'fragile'
else:                          verdict = 'very fragile'
print(f'Verdict for {winner}: {verdict}')
```
"""),

md("""**Discussion.** Without a benchmark, an RV in isolation is meaningless. *Robust* and *moderate* support deployment with monitoring; *fragile* and *very fragile* recommend a controlled trial before any spend. The sign-flip RV is the more honest bar: an estimate whose *direction* could reverse under hidden confounding is not deployment-ready, even if the magnitude is uncertain."""),

md("""---

## Artifact 6 — Deployment-Readiness Checklist

### Q6.1 State the target population in one sentence.

*Hint.* Bound it by unit type, time window, geographic scope, and operational regime.

*Your turn.*"""),

md("""*[Target population, one sentence.]*"""),

reveal("""Wafers from the same SECOM fab process, in the Jul-Oct 2008 calendar window, processed under the existing tool-PM cadence and supplier mix. Deployment to wafers outside this window requires a transportability re-check (Q6.2)."""),

md("""### Q6.2 Run a transportability check by splitting periods into source and target.

*Hint.* Transportability across time periods (Lab 13B's pattern): fit the estimator on the *source* subset (here, the earlier periods), fit it again on the *target* subset (the later periods), compare the two estimates. If the relative gap $|\\hat\\tau_{\\text{src}} - \\hat\\tau_{\\text{tgt}}| / |\\hat\\tau_{\\text{src}}| < 25\\%$, the estimate transports cleanly; if the gap is wider, the source-period estimate cannot be trusted to apply to the target period without a re-fit.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
if winner is not None:
    periods = sorted(df['period'].unique())
    src_mask = df['period'].isin(periods[:2])
    tgt_mask = df['period'].isin(periods[2:])
    estimates = {}
    for label, mask in [('source', src_mask), ('target', tgt_mask)]:
        sub = df[mask]
        Xs = sub[f'{winner}_bin'].values
        Zs = np.hstack([sub[[c for c in sensor_cols if c != winner]].values,
                        pd.get_dummies(sub['period'], drop_first=True).astype(float).values])
        Ys = sub['yield_fail'].values
        if (Xs == 0).sum() > 10 and (Xs == 1).sum() > 10:
            tg, ti, ta, sea, _ = four_estimators(Xs, Zs, Ys)
            estimates[label] = ta
            print(f'  {label} AIPW for {winner}: {ta:+.4f} (SE {sea:.4f})')

    if 'source' in estimates and 'target' in estimates and abs(estimates['source']) > 1e-9:
        gap = estimates['source'] - estimates['target']
        rel_gap = abs(gap) / abs(estimates['source'])
        print(f'  Absolute gap (source - target): {gap:+.4f}')
        print(f'  Relative gap: {100 * rel_gap:.1f}% of source estimate')
        if rel_gap < 0.25:   verdict = 'supported (gap < 25%); recommendation transports'
        elif rel_gap < 0.50: verdict = 'partial (25-50% gap); deploy to target with monitoring + pilot'
        else:                verdict = 'FAILED (gap > 50%); estimate does NOT transport; refit on target before deployment'
        print(f'  Transportability verdict: {verdict}')
```
"""),

md("""**Discussion.** Lab 13B's rule of thumb: source-vs-target AIPW estimates within 25% of each other → transport is supported. The verdict directly informs deployment: if transport fails, the winner sensor's effect that the capstone identified on the full cohort cannot be claimed to generalise to the most recent calendar window. The team's options at that point are (a) refit the analysis on the latest period before deployment, or (b) recommend a controlled trial on the target period to settle the question."""),

md("""### Q6.3 Name three deployment monitors with specific triggers.

*Hint.* Use the four named monitors from Lab 14B's deployment section (daily SMD, weekly propensity, monthly RV, quarterly per-period falsification). Pick three and name a *trigger threshold* for each.

*Your turn.*"""),

md("""*[Three monitors with triggers.]*"""),

reveal("""1. **Daily SMD on the winner's distribution.** Alarm if SMD from deployment-baseline > 0.25 (Cohen-d effect-size threshold).
2. **Weekly propensity-histogram check.** Re-fit $\\hat e(z) = P(S_i = 1 \\mid Z)$ on the most recent week. Alarm if the overlap region $\\{e \\in (0.05, 0.95)\\}$ shrinks below 70% of wafers.
3. **Monthly RV recompute.** Re-run §5's robustness value on the most recent month. Alarm if RV drops > 50% relative to deployment-time.

Each alarm escalates to the process-engineering lead within one business day."""),

md("""### Q6.4 Write the rollback criteria.

*Hint.* Rollback must be *fast* (no re-run of the capstone) and *named* (an on-call rota can execute it without thinking).

*Your turn.*"""),

md("""*[Rollback criteria.]*"""),

reveal("""The intervention (spec tightening on the winner sensor) is rolled back if any of:
- Q6.3 monitor (1) alarms for 3 consecutive days.
- Q6.3 monitor (2) alarms in any week.
- Q6.3 monitor (3) alarms in any month *and* root cause is not identified within 1 week.

Rollback = revert tolerance-band to pre-deployment spec (single config change). Decision authority: process-engineering lead, with ML-platform lead concurring."""),

md("""---

## Closing — write the 5-slide executive summary

The deliverable for the capstone includes a 5-slide deck for a director who has not read the report. Below, write the headline content of each slide.

*Your turn — fill in five slides in 1-2 sentences each.*"""),

md("""*[Slide 1: the question.]*

*[Slide 2: the identifying assumptions.]*

*[Slide 3: the estimate + CI.]*

*[Slide 4: the sensitivity bound + verdict.]*

*[Slide 5: the deployment recommendation.]*"""),

reveal("""**Slide 1 — The question.** Which top-yield-correlated SECOM sensor has the largest causal effect on yield_fail, after blocking the back-door through period?

**Slide 2 — Identifying assumptions.** Back-door adjustment on $Z = \\{$period, other sensors$\\}$. Load-bearing assumption: no unmeasured confounders beyond $Z$.

**Slide 3 — The estimate.** Winner sensor [X] with AIPW = +[Y] (95% CI [lo, hi]). Estimator agreement across G-comp / IPW / AIPW supports the magnitude.

**Slide 4 — Sensitivity.** RV(sign-flip) = [Z] vs measured-confounder benchmark of [B]. Verdict: [robust / moderate / fragile / very fragile].

**Slide 5 — Recommendation.** *If robust/moderate:* deploy spec tightening on the winner sensor with the four §6 monitors. *If fragile:* recommend a controlled trial before deployment; do not spend the capital based on the observational estimate."""),

md("""## Reflection

You've now produced a defensible capstone analysis on SECOM. The five-step skeleton — DAG → identification → estimation → sensitivity → deployment — that you applied here is the same pipeline that will carry over to any other dataset.

**How to adapt this notebook to other starters.**
- *Starter B (AI4I, lab14d_ai4i):* swap the back-door problem for the AI4I rotational-speed question; the DAG defends from physics rather than calendar.
- *Starter C (LFP, lab14e_lfp):* swap binary ATE for *time-varying* effect; the four-estimator gauntlet becomes the one-shot-vs-sequential g-formula comparison.
- *Starter D (Backblaze, lab14f_backblaze):* swap ATE for *DTR policy value*; the estimator becomes Q-learning + OPE.
- *Starter E (OEE synthetic, lab14g_oee):* swap ATE for *multi-mediator NDE/NIE*; the sensitivity becomes a γ-sweep + benchmark against ground truth.
- *Starter F (multi-site synthetic, lab14h_multisite):* swap single-site ATE for *transported ATE*; the deployment checklist becomes a transportability validation.

The book's parting observation, restated: *The methods are not the bottleneck. The choice of estimator implies a set of identifying assumptions; the assumptions are testable only against domain knowledge; the defensibility of the analysis rests on whether those assumptions hold in the specific industrial setting where it is deployed.* The capstone tests whether you can defend an analysis end-to-end. Pick a question you actually care about. Good luck.

Refer to [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the handbook with min-bar / exemplary-bar standards per artifact."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14c_secom.ipynb", cells)
print("Built lab14c_secom.ipynb")
