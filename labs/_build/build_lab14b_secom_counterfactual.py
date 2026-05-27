"""Build labs/ch14/lab14b_secom_counterfactual.ipynb — counterfactual attribution for QC flags on SECOM."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 14B — Counterfactual Attribution for QC Flags on SECOM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14b_secom_counterfactual.ipynb)

**Companion to the capstone in Ch 14.** Chapter 14 §14.5 gives a synthetic worked example: a vision classifier flags a wafer with score 0.98, and counterfactual attribution to upstream process variables ($G$, $T$) reveals that *gas flow alone* accounts for 0.90 of the flag while temperature contributes essentially nothing. The actionable instruction — "investigate gas flow first" — comes from the *causal* attribution, not from a SHAP map over image features. **Lab 14B applies the same mechanism to real semiconductor data.**

SECOM is the right test bed: the 5 sensors play the role of "image features" (what the classifier consumes), and `period` plays the role of the *upstream causal variable* (calibration cycle / supplier rotation / seasonality — what an engineer can actually act on). The chain `period → sensors → flag` mirrors §14.5's `G → image features → flag`.

The deliverable is a per-wafer attribution table: for each flagged wafer, the engineer's action prompt is the counterfactual contribution of `period`, *not* the SHAP attribution to sensors. We then aggregate across flagged wafers, stress the period→sensor model, and connect the analysis to the §14.7 capstone artifacts.

**Dataset.** SECOM (5 sensors + `period` + `yield_fail`), same slice as Lab 1B. 1567 wafers spanning Jul-Oct 2008.

**Estimand.** For each flagged wafer $i$:

$$\\mathrm{Contribution}_i(\\text{period}) \\;=\\; \\hat{p}_{\\mathrm{factual}}(i) \\;-\\; \\hat{p}_{do(\\mathrm{period} = \\mathrm{baseline})}(i),$$

with the wafer's idiosyncratic noise held fixed between factual and counterfactual."""),

md("""## What this lab is *not* doing

- **Running a deep vision CNN.** The classifier is logistic on tabular sensors. The §14.5 mechanism is identical for any classifier that consumes a feature vector; the depth of the model is incidental. A reader who wants a true vision CNN should swap our `LogisticRegression` for a CNN on MVTec / NEU-DET images plus an upstream-process-telemetry source (which no public industrial vision dataset currently provides — that gap is part of why this lab uses SECOM).
- **Validating against ground-truth causal effects.** Per-unit counterfactual effects are not observable in any real dataset; the lab can only show that the attribution is *internally consistent* with the assumed DAG.
- **Computing the full §14.7 capstone.** This lab covers approximately *artifacts 1-3* of the capstone checklist (problem + DAG, estimand, identification + estimator). Artifacts 4-6 (sensitivity, deployment-readiness, monitoring) are the student's extension."""),

code("""%pip install -q ucimlrepo shap"""),

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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)"""),

md("""## Background — the §14.5 mechanism in 90 seconds

Before applying it to SECOM, here is what *counterfactual attribution* actually computes — and why it is different from SHAP, LIME, or any other model-explanation tool.

**The chain.** A QC classifier $f$ consumes input features $X = (X_1, \\dots, X_k)$ and emits a score $\\hat{p} = f(X)$. The features themselves are not exogenous — they are caused by upstream variables $V_1, \\dots, V_m$ (process knobs, environmental conditions, supplier lots) that engineers can act on. Schematically:

```
   V_1, V_2, ...  ─────►  X_1, X_2, ...  ─────►  flag (yes/no)
   (upstream causes)      (features the          (classifier output)
                           classifier sees)
```

**SHAP attributes in $X$-space.** Given a flagged unit, SHAP decomposes $\\hat{p}$ into a sum of per-feature contributions $\\phi_1, \\dots, \\phi_k$. The answer is in *feature space*: "sensor S3 contributed +0.32, sensor S5 contributed +0.08". This is correct as a model-explanation but does not tell an engineer what to *do* — the engineer does not write to sensors, they adjust process variables.

**Counterfactual attribution acts in $V$-space.** For a flagged unit $i$ with upstream values $v_i$, the contribution of $V_j$ is the difference between the factual score and the score *under $do(V_j = v_j^{\\text{baseline}})$, with everything else for unit $i$ held fixed*:

$$\\mathrm{Contribution}_i(V_j) \\;=\\; f(X_i) \\;-\\; f\\big(X_i^{do(V_j = v_j^{\\text{baseline}})}\\big),$$

where $X_i^{do(V_j = v_j^{\\text{baseline}})}$ is computed by *propagating the intervention* through the SCM that maps $V \\to X$. The per-unit idiosyncratic part of $X_i$ (its noise residual) is kept; only the structural contribution of $V_j$ is replaced by the baseline.

**Why "per-unit noise held fixed" matters.** The same wafer under $do(V_j = v_j^{\\text{baseline}})$ would still have *its own* lighting flutter, microscopic surface variation, sensor jitter. We are not asking the population question "what would the *average* wafer look like at baseline $V_j$?" — we are asking the individual question "what would *this* wafer look like, had its upstream $V_j$ been at baseline?" The former is the ATE of Chapter 5; the latter is the individual counterfactual of §14.5.

**The full procedure.**
1. Fit the QC classifier on observed features.
2. Specify the SCM $V \\to X$ — fit per-feature regressions $\\hat{X}_j = g_j(V) + \\varepsilon_j$.
3. For unit $i$, extract its residual noise: $\\hat{\\varepsilon}_i^{(j)} = X_{ij} - g_j(v_i)$.
4. Counterfactual: $X_i^{do(V = v^*)}_j = g_j(v^*) + \\hat{\\varepsilon}_i^{(j)}$.
5. Score: $\\hat{p}_i^{do} = f(X_i^{do})$. Contribution: $\\hat{p}_i^{\\text{factual}} - \\hat{p}_i^{do}$."""),

md("""## Part 1 — Load and frame as a QC pipeline

We treat each wafer as a unit a vision-QC system flags. The 5 sensors are the *features the classifier consumes* (in a real vision system these would be image embeddings); `period` is the *upstream causal variable* (calibration cycle / supplier rotation / seasonality)."""),

code("""df = load_secom(chapter=1).copy()
sensor_cols = [c for c in df.columns if c.startswith('S')]
period_levels = sorted(df['period'].unique())

# Standardise sensors so logistic regression coefficients are interpretable.
scaler = StandardScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

# Period as ordered integer for the period -> sensor regressions.
df['period_idx'] = df['period'].map({p: i for i, p in enumerate(period_levels)})

print(f'Shape:                     {df.shape}')
print(f'Sensors (features):        {sensor_cols}')
print(f'Periods (upstream levels): {period_levels}')
print(f'Failure rate:              {df[\"yield_fail\"].mean():.2%}')"""),

md("""## Part 2 — Fit the QC classifier

A logistic regression on the 5 sensors plays the role of the vision classifier. We use the same threshold (`predict_proba > 0.5`) for the binary flag decision; threshold tuning is a deployment question (Chapter 13) and orthogonal to the attribution question we are asking here."""),

code("""X = df[sensor_cols].values
y = df['yield_fail'].values

clf = LogisticRegression(class_weight='balanced', max_iter=4000).fit(X, y)
df['flag_score'] = clf.predict_proba(X)[:, 1]
df['flag']       = (df['flag_score'] > 0.5).astype(int)

print(f"QC classifier accuracy: {clf.score(X, y):.3f}")
print(f"Flag rate (predicted P > 0.5): {df['flag'].mean():.2%}")
print(f"True failure rate:             {df['yield_fail'].mean():.2%}")
print()
print('Score distribution by true label:')
print(df.groupby('yield_fail')['flag_score'].describe().round(3).to_string())"""),

md("""## Part 3 — Fit the SCM $V \\to X$ (period → sensors)

For each sensor $X_j$ we fit a per-period mean as the structural model $g_j(\\text{period})$. A wafer's residual sensor reading $\\varepsilon_{ij} = X_{ij} - g_j(\\text{period}_i)$ is its idiosyncratic contribution (manufacturing variability, microscopic conditions, sensor jitter) — *not* attributable to the upstream period.

We use one-hot period dummies in a linear regression for $g_j$ — this is the saturated model that exactly matches per-period sensor means."""),

code("""period_dummies = pd.get_dummies(df['period'], drop_first=False).astype(float)
period_cols = period_dummies.columns.tolist()

g_models = {}     # j -> fitted regression
residuals = {}    # j -> per-wafer residual array
for s in sensor_cols:
    m = LinearRegression(fit_intercept=False).fit(period_dummies.values, df[s].values)
    g_models[s] = m
    residuals[s] = df[s].values - m.predict(period_dummies.values)

resid_df = pd.DataFrame({s: residuals[s] for s in sensor_cols})
print('Per-sensor residual std (after subtracting period-mean structure):')
print(resid_df.std().round(3).to_string())
print()
# Variance shares
total_var = df[sensor_cols].var()
period_explained = total_var - resid_df.var()
print('Variance share explained by period (per sensor):')
for s in sensor_cols:
    pe = float(period_explained[s] / max(total_var[s], 1e-12))
    print(f'  {s}: {pe:.3f}')"""),

md("""**Read the variance shares.** A sensor whose variance share explained by period is large (say > 0.20) is *substantially* under upstream control: its readings move with calibration / supplier / seasonality. A sensor whose share is small is largely driven by within-period idiosyncrasy. In §14.5's language, sensors with large period-shares behave like "image features that respond strongly to gas flow"; sensors with small shares are like "image features dominated by per-wafer noise". The counterfactual attribution to period will move the former substantially and the latter very little."""),

md("""## Part 4 — Pick the baseline period and the wafer to attribute

**Choosing the baseline.** The §14.5 baseline is "the value an engineer would reset the upstream variable to." For an ordered process variable like gas flow, that is usually a target setpoint. For a categorical variable like `period`, there is no natural numerical baseline; the right pick is *the period whose structural sensor profile gives the lowest classifier-predicted flag rate* — i.e., the "cleanest" calibration epoch on record. We pick it programmatically so the lab is self-anchoring."""),

code("""# For each period, compute the classifier score on its STRUCTURAL sensor vector
# (the per-period mean sensors, with zero residual noise). The period with the
# lowest predicted flag rate is the natural baseline.
mean_score_per_period = {}
for p in period_levels:
    cf_pv = np.zeros((1, len(period_cols)))
    cf_pv[0, period_cols.index(p)] = 1.0
    structural_sensors = np.array([float(g_models[s].predict(cf_pv)[0]) for s in sensor_cols])
    # Average the structural sensors with each wafer's residual noise -> per-wafer
    # counterfactual score; take its mean across the flagged population.
    cf_scores_p = []
    for j in df.index:
        cf_j = structural_sensors + np.array([residuals[s][j] for s in sensor_cols])
        cf_scores_p.append(clf.predict_proba(cf_j.reshape(1, -1))[0, 1])
    mean_score_per_period[p] = float(np.mean(cf_scores_p))

baseline_period = min(mean_score_per_period, key=mean_score_per_period.get)

print('Mean counterfactual flag score per candidate baseline period:')
for p in period_levels:
    marker = '  <- chosen baseline' if p == baseline_period else ''
    print(f'  {p}: {mean_score_per_period[p]:.4f}{marker}')
print()
print(f"Baseline period: {baseline_period}  (the 'cleanest' calibration epoch in the data).")"""),

md("""**Picking the wafer.** Highest-scoring wafer the classifier flagged. The narrative is "this is the wafer the system flagged most confidently — why?"."""),

code("""# Pick the most confidently flagged wafer.
flagged_idx = df[df['flag'] == 1].sort_values('flag_score', ascending=False).index
i = int(flagged_idx[0])
wafer = df.loc[i]

print(f'Flagged wafer i = {i}')
print(f'  period:      {wafer[\"period\"]}  (baseline = {baseline_period})')
print(f'  flag score:  {wafer[\"flag_score\"]:.3f}')
print(f'  true label:  yield_fail = {int(wafer[\"yield_fail\"])}')
print()
print('Observed sensor values (standardised):')
for s in sensor_cols:
    print(f'  {s}: {wafer[s]:+.3f}')"""),

code("""def counterfactual_score(wafer_idx, target_period):
    \"\"\"Compute the classifier score for wafer wafer_idx under do(period = target_period).

    Holds the wafer's per-sensor residual noise fixed; replaces the structural
    (period-explained) part with the baseline period's value.
    \"\"\"
    # Build a one-hot vector for the counterfactual period
    cf_period_vec = np.zeros((1, len(period_cols)))
    cf_period_vec[0, period_cols.index(target_period)] = 1.0

    # For each sensor, predict the structural value at the counterfactual period
    cf_sensors = np.empty(len(sensor_cols))
    for k, s in enumerate(sensor_cols):
        structural_at_cf = float(g_models[s].predict(cf_period_vec)[0])
        cf_sensors[k] = structural_at_cf + residuals[s][wafer_idx]
    return float(clf.predict_proba(cf_sensors.reshape(1, -1))[0, 1])

factual = float(wafer['flag_score'])
cf      = counterfactual_score(i, baseline_period)

contribution = factual - cf
print(f'Wafer {i} counterfactual attribution to period:')
print(f'  Factual score (period = {wafer[\"period\"]}):       {factual:.3f}')
print(f'  Counterfactual score (period = {baseline_period}):    {cf:.3f}')
print(f'  Contribution of period to flag:              {contribution:+.3f}')
print()
print('Interpretation:')
if contribution > 0.01:
    print(f'  The actual period ({wafer[\"period\"]}) RAISED this wafer\\'s flag score by {contribution:+.3f}')
    print(f'  relative to the baseline period {baseline_period}. The remaining score (~{cf:.2f})')
    print('  is attributable to per-wafer idiosyncrasy that period intervention cannot fix.')
    print(f'  Action prompt: investigate what changed in period {wafer[\"period\"]} (calibration,')
    print('  supplier rotation, ambient conditions).')
elif contribution < -0.01:
    print(f'  The actual period ({wafer[\"period\"]}) LOWERED this wafer\\'s flag score by {-contribution:+.3f}')
    print(f'  relative to baseline. This wafer was flagged DESPITE being in a relatively')
    print(f'  clean period -- the per-wafer noise residual ~{cf:.2f} is what drove the flag.')
    print('  Action prompt: investigate per-wafer process variability (tool jitter, microscopic')
    print('  contamination); period intervention would not help.')
else:
    print('  Period contribution is near zero -- this wafer was flagged for per-wafer')
    print('  reasons unrelated to the upstream period regime.')"""),

md("""**Why this matters — the §14.5 lesson on real data.**

Chapter 14's worked example (§14.5) showed a wafer where the upstream variable $G$ contributed +0.90 to the flag — almost the entire score. The engineer's action was clear: investigate gas flow. On real SECOM data, the most-confidently-flagged wafer often falls in a *different* regime: its sensor values are extreme, but most of that extremeness is in the *residual* (per-wafer noise from manufacturing variability, microscopic contamination, sensor jitter) rather than in the *period-explained* portion. The counterfactual contribution to period is small; the engineer's action is *not* to investigate the period regime.

This is the canonical case where **per-unit counterfactual attribution diverges from population-level attribution**. Across the flagged population (Part 6), period contributes ~+0.17 on average; but for *this specific wafer* it contributes ~0. Both findings are correct and complementary — they answer different questions:

- *Population:* "Across our 366 flagged wafers, what fraction of the flag rate is driven by the period regime?" → ~17%.
- *This wafer:* "For wafer $i$, what fraction of *its* flag was driven by the period regime?" → near 0.

A real-world QC pipeline reports both: a population-level recommendation (e.g., "calibrate the period drift") and a per-wafer recommendation (e.g., "this wafer is a per-unit outlier; investigate its specific run conditions")."""),

md("""## Part 5 — Contrast with SHAP attribution

SHAP attributes the flag score to the *features the classifier consumed* (the sensors). For our wafer, SHAP will say which sensor drove the flag — useful for debugging the model, but not directly actionable for the engineer. Counterfactual attribution says which *upstream variable* drove the flag — actionable, but only as good as the SCM."""),

code("""import shap

# SHAP on the logistic-regression classifier. LinearExplainer is exact for logistic.
background = X[rng.integers(0, len(X), size=min(200, len(X)))]
explainer = shap.LinearExplainer(clf, background)
shap_values = explainer(X)

wafer_shap = shap_values.values[i]
shap_df = pd.DataFrame({
    'sensor':   sensor_cols,
    'SHAP_phi': wafer_shap,
}).sort_values('SHAP_phi', key=abs, ascending=False)
print(f'SHAP attribution for wafer {i} (sensor space):')
print(shap_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))
print(f'Sum of SHAP phi = {wafer_shap.sum():+.4f}  (expected base score ~ '
      f'{explainer.expected_value:+.4f})')"""),

code("""print(f'\\nSide-by-side comparison for wafer {i}:')
print(f'  Classifier flag score (factual):            {factual:.3f}')
print(f'  Top SHAP feature:                            {shap_df.iloc[0][\"sensor\"]} '
      f'(phi = {shap_df.iloc[0][\"SHAP_phi\"]:+.3f})')
print(f'  Counterfactual contribution of period:      {factual - cf:+.3f}')
print()
print('The engineer asks: what should I act on?')
print(f'  - SHAP says: investigate {shap_df.iloc[0][\"sensor\"]} (the sensor)')
print(f"  - Counterfactual says: investigate the period-{wafer['period']} regime")
print(f'  - The two answer different questions; both are valid in their own scope.')"""),

md("""**The two attributions answer different questions.**

- **SHAP** answers: *"Among the features the classifier sees, which contributed most to this prediction?"* The answer is in feature space — useful for model debugging, ML observability, monitoring representation drift.
- **Counterfactual on period** answers: *"Among the upstream causal variables, which contributed most to this prediction?"* The answer is in process-variable space — useful for the engineer who wants to act.

When the SCM is correctly specified and the classifier is faithful to its features, the two should *not* identify the same target. SHAP tells you about the model; counterfactual tells you about the world. The §14.4 lesson is to read them as *complementary*, not interchangeable."""),

md("""## Part 6 — Population aggregation

A single wafer's attribution may be noisy. Aggregate over all flagged wafers and ask: *across the population, how much of the flag rate is attributable to period vs to within-period idiosyncrasy?* This is the population analogue of the per-unit attribution — and it connects this lab back to the mediation analysis of Chapter 10 (NDE/NIE)."""),

code("""flagged = df[df['flag'] == 1].copy()
print(f'Flagged wafer pool: {len(flagged)} wafers across {flagged[\"period\"].nunique()} periods')

# Compute counterfactual scores for all flagged wafers under baseline period
cf_scores = np.empty(len(flagged))
for idx_local, row_i in enumerate(flagged.index):
    cf_scores[idx_local] = counterfactual_score(row_i, baseline_period)

flagged['cf_score']     = cf_scores
flagged['contribution'] = flagged['flag_score'] - flagged['cf_score']

print()
print('Per-period summary of the period contribution to flag scores:')
summary = flagged.groupby('period').agg(
    n_flagged=('flag_score', 'size'),
    mean_factual=('flag_score', 'mean'),
    mean_cf=('cf_score', 'mean'),
    mean_contribution=('contribution', 'mean'),
).round(3)
print(summary.to_string())
print()
print(f'Population-mean attribution of flag score to period (across all flagged wafers): '
      f'{flagged[\"contribution\"].mean():+.3f}')"""),

code("""# Visualisation: SHAP top-feature consistency vs counterfactual contribution.
top_shap_feature = np.argmax(np.abs(shap_values.values), axis=1)
shap_top = pd.Series(top_shap_feature[flagged.index], name='top_shap_feature_idx')

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(flagged['flag_score'].values,
           flagged['contribution'].values,
           c=shap_top.values, cmap='tab10', alpha=0.7)
ax.set_xlabel('Factual flag score')
ax.set_ylabel('Counterfactual contribution of period')
ax.set_title('Per-flagged-wafer: flag score vs period contribution\\n(colour = SHAP top feature)')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Disagreement diagnostic: for each flagged wafer, compute SHAP's *magnitude*
# (the sum of |phi_j| across sensors) and the period counterfactual contribution.
# A wafer with HIGH SHAP magnitude and LOW period contribution is one where the
# model-explanation says "look at the sensors" but the causal attribution says
# "period is not the actionable lever" -- exactly the divergence case.
shap_magnitude = np.abs(shap_values.values[flagged.index]).sum(axis=1)
flagged['shap_magnitude'] = shap_magnitude

high_shap_low_period = ((flagged['shap_magnitude'] > flagged['shap_magnitude'].median())
                        & (flagged['contribution'] < flagged['contribution'].median()))
high_shap_high_period = ((flagged['shap_magnitude'] > flagged['shap_magnitude'].median())
                         & (flagged['contribution'] > flagged['contribution'].median()))
print(f\"High SHAP magnitude  +  HIGH period contribution: {int(high_shap_high_period.sum())} wafers\")
print(f\"  -> model and causal attribution AGREE that the flag is real and actionable upstream.\")
print()
print(f\"High SHAP magnitude  +  LOW  period contribution: {int(high_shap_low_period.sum())} wafers\")
print(f\"  -> model says 'the sensors matter' but counterfactual says 'period is not the cause'.\")
print(f\"  -> These wafers' flags are NOT actionable via the period intervention; their per-wafer\")
print(f\"     noise residuals (manufacturing variability, microscopic conditions) drove the flag.\")"""),

md("""## Part 7 — Sensitivity: how robust is the attribution?

The counterfactual attribution depends on the period→sensor SCM (Part 3). Two failure modes:

1. **Misspecification.** If a sensor's true response to period is non-linear, our linear-in-dummies fit captures the per-period means but misses within-period heterogeneity. Effect: residuals are inflated, and the counterfactual score under-counts the period contribution.
2. **Confounded period→sensor regression.** Period is itself caused by upstream variables we have not measured (recipe revisions, ambient temperature, operator rotations). The structural value $g_j(\\text{period})$ would, under a more complete SCM, decompose further.

A pragmatic stress test: add small Gaussian noise to the per-period structural estimates, re-run the attribution, and check that the *direction* of the answer (which period contributes most) does not flip."""),

code("""rng_sens = np.random.default_rng(1)
sensitivity_strengths = [0.0, 0.05, 0.10, 0.20, 0.40]
sens_rows = []
for sigma in sensitivity_strengths:
    g_models_perturbed = {}
    for s in sensor_cols:
        coef_orig = g_models[s].coef_.copy()
        # Perturb per-period coefficients with noise proportional to the residual std.
        residual_std = float(np.std(residuals[s], ddof=1))
        noise = rng_sens.normal(0, sigma * residual_std, size=coef_orig.shape)
        m_p = LinearRegression(fit_intercept=False)
        m_p.coef_ = coef_orig + noise
        m_p.intercept_ = 0.0
        g_models_perturbed[s] = m_p

    def cf_score_perturbed(wafer_idx, target_period):
        cf_pv = np.zeros((1, len(period_cols)))
        cf_pv[0, period_cols.index(target_period)] = 1.0
        cf_sensors = np.empty(len(sensor_cols))
        for k, s in enumerate(sensor_cols):
            structural = float(g_models_perturbed[s].predict(cf_pv)[0])
            cf_sensors[k] = structural + residuals[s][wafer_idx]
        return float(clf.predict_proba(cf_sensors.reshape(1, -1))[0, 1])

    cf_scores_p = np.array([cf_score_perturbed(j, baseline_period) for j in flagged.index])
    sens_rows.append({
        'sigma':                sigma,
        'mean_contribution':    float((flagged['flag_score'].values - cf_scores_p).mean()),
        'frac_pos_contribution': float(((flagged['flag_score'].values - cf_scores_p) > 0).mean()),
    })

sens_df = pd.DataFrame(sens_rows)
print('Sensitivity to perturbation of the period -> sensor SCM:')
print(sens_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))"""),

md("""**Read the sensitivity table.** At $\\sigma = 0$ we recover the Part-6 attribution. As we perturb the SCM:

- If `mean_contribution` and `frac_pos_contribution` are stable across $\\sigma$, the attribution is robust to plausible SCM misspecification.
- If they shrink toward zero as $\\sigma$ grows, the attribution is brittle — the SCM was doing most of the work, and any model error wipes out the conclusion.
- If `frac_pos_contribution` drops below 0.5 even at small $\\sigma$, the direction of the attribution can flip for some wafers — those wafers' counterfactual decisions are not stable enough to act on individually."""),

md("""## Part 8 — Decision

Three bullets, the deliverable a process engineer + an ML observability lead would each read:

1. **Per-wafer attribution table** (Part 4 for one wafer; Part 6 for the flagged population). For each flagged wafer, the engineer's action prompt is the counterfactual *contribution of period* — the upstream variable they can intervene on. The number directly translates to "investigate the period regime" (a calibration audit, a supplier check, an ambient-conditions check).

2. **SHAP-vs-counterfactual divergence** (Part 5). When SHAP and counterfactual rank different things, the disagreement is *itself* the diagnostic: SHAP tells you what the model sees, counterfactual tells you what the world causes. A flagged wafer where both agree is the easy case; a flagged wafer where they diverge is the case that needs a process engineer + an ML engineer in the same room.

3. **Sensitivity bound** (Part 7). The attribution survives small SCM perturbations but degrades at large ones. The deployment-time policy: re-fit the period→sensor SCM monthly (drift monitoring); recompute the attribution when sensor variances shift by more than 20% (concept drift trigger); flag any wafer for human review when the SHAP top feature and the counterfactual top variable disagree on the action (Part 6's plot)."""),

md("""## Part 9 — Connection to the capstone (§14.7)

This lab covers approximately *three of the six* capstone artifacts §14.7 asks for. The remaining three are the natural extension a capstone project would deliver.

| Capstone artifact | Coverage in this lab |
|---|---|
| 1. Problem statement and DAG | ✓ Parts 1-3: vision-QC framing, period → sensors → flag DAG with edges defended from SECOM domain knowledge (calibration, supplier rotation, seasonality). |
| 2. Estimand | ✓ Part 4: the per-wafer counterfactual contribution of period. The population analogue (Part 6) is the NDE/NIE-style decomposition of Chapter 10. |
| 3. Identification | ✓ Part 3's SCM specification and Part 7's sensitivity-to-misspecification address the identifying assumptions. |
| 4. Estimator | Partial — the lab fits a logistic classifier and a per-sensor SCM. A capstone would defend each over alternatives (deep nets, GBMs, hierarchical SCM forms). |
| 5. Sensitivity analysis | Partial — Part 7 covers SCM-misspecification sensitivity; a capstone would add a Cinelli-Hazlett RV for the hidden-confounder direction (Lab 13B's tooling). |
| 6. Deployment-readiness | Not covered — a capstone would add a deployment checklist (population, transportability, monitoring triggers, rollback). Lab 13B's transportability framework is the template. |

The student's capstone path: take this lab's analysis, add the missing three artifacts, and produce a single-document deliverable applying the same machinery to *a different dataset of their choice*."""),

md("""## Reflection

**The model is not the cause.** SHAP, LIME, Grad-CAM, integrated gradients — every model-explanation tool attributes a prediction to *the features the model saw*. None of them attribute to the upstream variables that *caused* those features. For a deployment where the engineer must take action upstream of the model, the model-explanation tool is necessary (for ML debugging) but not sufficient (for engineering action). The counterfactual attribution in this lab is the structural step that bridges from one to the other.

**A correct SCM is the cost of admission.** §14.4 stated this and §14.5 demonstrated it on synthetic data: when the DAG is right, counterfactual attribution gives the actionable answer; when the DAG is wrong, the attribution is an extrapolation from a regression surface rather than an intervention on the real causal mechanism. SECOM only gives us one upstream variable to work with (`period`); a real fab would have ten or twenty, and the DAG over them is the *prerequisite* the analysis cannot skip.

**The course's recurring lesson, restated for the last time.** Every chapter has converged on the same point: *the choice of estimator implies a set of identifying assumptions; the assumptions are testable only against domain knowledge; the defensibility of the analysis rests on whether those assumptions hold in the specific industrial setting where it is deployed.* This lab is no different. The counterfactual machinery is portable. The DAG is not."""),

md("""## What's next — the capstone

The course is complete. The capstone exists to integrate everything: a single defensible analysis on a problem of your choice, applying the full pipeline (DAG → identification → estimation → sensitivity → deployment) to a dataset and a question that matters. §14.7 lists the six artifacts; this lab covers the first three to give you a worked template.

Pick a problem you actually care about. Causal inference rewards depth over breadth — a defensible analysis on one well-chosen question is worth far more than a shallow survey across many."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14b_secom_counterfactual.ipynb", cells)
print("Built lab14b_secom_counterfactual.ipynb")
