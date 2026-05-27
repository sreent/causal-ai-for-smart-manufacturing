"""Build labs/ch05/lab05b.ipynb — DML on SECOM with four-estimator comparison."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 5B — Double Machine Learning on Real SECOM Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch05/lab05b.ipynb)

**Companion to Lab 5A.** Lab 5A built G-comp, IPW, AIPW, and DML on a synthetic line where we knew the true ATE and could verify each estimator recovered it. **Lab 5B applies the same four estimators to SECOM, where no oracle exists.**

What replaces the oracle here is **estimator agreement**. When G-comp, IPW, AIPW, and DML converge to the same number under the same assumed DAG, we have triangulating evidence the estimate is not an artefact of a single nuisance model. When they disagree, the disagreement diagnoses which assumption is failing.

**Dataset.** SECOM. Same five sensors as Lab 1B. Here we pick the top sensor as the *treatment* (binarized at its median), use the other four as continuous controls, and estimate the ATE on `yield_fail`."""),

md("""## What this lab is *not* doing

- **Feature selection.** Five sensors are pre-selected in `secom_prep.py` (same as Lab 1B). The treatment is the top sensor by |corr with yield| — the one a naive ML pipeline would point at.
- **Binarization choice.** The treatment is dichotomised at the *sample median*. A real engineering analysis would dichotomise at a process-meaningful threshold (e.g., a control-chart specification limit). Median split is the chapter's pedagogical convention.
- **Period adjustment.** Period was the back-door confounder in Lab 1B. Here we run DML without period in the control set as the baseline (matches the chapter's worked example), then add period as a robustness check.
- **SUTVA debate.** Wafer-level treatment assumes one wafer's sensor reading doesn't affect another wafer's yield. Plausible for sensor-level treatment; less so for tool-level interventions.
- **Hyperparameter tuning.** Defaults throughout."""),

code("""# Install causal-inference libraries used in this lab.
%pip install -q ucimlrepo econml"""),

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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load the focused slice

The Ch 5 slice returns the same 5 sensors as Ch 1, plus a `period` stratifier and `yield_fail`. The slice's `attrs` carries the convention: the top sensor by |corr with yield| is the designated treatment; the others are controls."""),

code("""df = load_secom(chapter=5)
treatment_sensor = df.attrs["treatment"]
control_sensors  = df.attrs["controls"]

print(f"Shape:           {df.shape}")
print(f"Treatment X:     {treatment_sensor}")
print(f"Controls Z:      {control_sensors}")
print(f"Outcome Y:       yield_fail")
print(f"Failure rate:    {df['yield_fail'].mean():.3%}")"""),

md("""## Part 2 — The estimand and the DAG

We frame the question as: *if we shifted this sensor from \"low\" (below its median) to \"high\" (at or above its median), by how much would the failure rate change, holding the other four sensors fixed?*

**Binarize the treatment.** Let `X = 1` iff the treatment sensor is at or above its median, else `X = 0`. This is the chapter's convention; the median split gives roughly equal exposure groups.

**Assumed DAG.**

```
   Z (four other sensors) ──┬──► X (treatment sensor, binarized)
                            └──► Y (yield_fail)

   X ────────────────────────────► Y
```

The four other sensors are observed covariates that plausibly cause both the treatment sensor's reading and yield (correlated production conditions, calibration state). Under this DAG, the back-door criterion is satisfied by `Z`. The four estimators of §5.3–5.7 all target the ATE of X on Y under this DAG.

**What this DAG does *not* claim.** It does not say all confounders are in `Z`. Period (Lab 1B's confounder) is excluded here — by design, so we can study whether the four estimators agree in the simplest case, then add period in the robustness step."""),

code("""# Binarize the treatment at its sample median.
x_continuous = df[treatment_sensor].values
x_threshold  = np.median(x_continuous)
X = (x_continuous >= x_threshold).astype(int)
Z = df[control_sensors].values
Y = df["yield_fail"].values

print(f"Treatment threshold (median of {treatment_sensor}): {x_threshold:.3f}")
print(f"P(X=1):       {X.mean():.3f}   (target ~0.5 from median split)")
print(f"P(Y=1|X=1):   {Y[X==1].mean():.3%}")
print(f"P(Y=1|X=0):   {Y[X==0].mean():.3%}")
print(f"Naive ATE:    {Y[X==1].mean() - Y[X==0].mean():+.3%}   (no covariate adjustment)")"""),

md("""**Read this naive ATE before continuing.** The unadjusted difference is what a process engineer would compute from a contingency table — \"high X correlates with this much extra failure rate.\" That number is the back-door-confounded estimate. The four estimators in Part 3 ask: under the assumed DAG, what is the *causal* ATE after blocking the back-door through `Z`?"""),

md("""## Part 3 — Four estimators under the same DAG

We compute each estimator with **5-fold cross-fitting** so the comparison is apples-to-apples. The nuisance models are gradient-boosted classifiers (the chapter's default ML choice).

- **G-computation**: fit $\\hat{\\mu}(x, z) = E[Y \\mid X = x, Z = z]$, average $\\hat{\\mu}(1, Z_i) - \\hat{\\mu}(0, Z_i)$ over the sample.
- **IPW**: fit $\\hat{e}(z) = P(X = 1 \\mid Z = z)$, trim to $[0.05, 0.95]$, apply the IPW formula.
- **AIPW**: combine both nuisances using the doubly-robust score.
- **DML**: AIPW score under cross-fitting (this implementation does that already)."""),

code("""def cross_fit_nuisances(X, Z, Y, K=5, seed=0):
    \"\"\"Return out-of-fold predictions for mu0, mu1, e (propensity).\"\"\"
    n = len(Y)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    e_hat   = np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(Z):
        # Outcome models, one per treatment arm.
        m0 = GradientBoostingClassifier(random_state=seed)
        m1 = GradientBoostingClassifier(random_state=seed)
        if (X[tr] == 0).sum() > 1 and len(np.unique(Y[tr][X[tr] == 0])) > 1:
            m0.fit(Z[tr][X[tr] == 0], Y[tr][X[tr] == 0])
            mu0_hat[te] = m0.predict_proba(Z[te])[:, 1]
        if (X[tr] == 1).sum() > 1 and len(np.unique(Y[tr][X[tr] == 1])) > 1:
            m1.fit(Z[tr][X[tr] == 1], Y[tr][X[tr] == 1])
            mu1_hat[te] = m1.predict_proba(Z[te])[:, 1]
        # Propensity model.
        ep = GradientBoostingClassifier(random_state=seed)
        ep.fit(Z[tr], X[tr])
        e_hat[te] = ep.predict_proba(Z[te])[:, 1]
    return mu0_hat, mu1_hat, e_hat

mu0_hat, mu1_hat, e_hat = cross_fit_nuisances(X, Z, Y, K=5, seed=0)

print(f"Propensity range: [{e_hat.min():.3f}, {e_hat.max():.3f}]")
print(f"Propensity quartiles: {np.quantile(e_hat, [0.25, 0.5, 0.75]).round(3).tolist()}")"""),

code("""def g_comp(mu0, mu1):
    return float(np.mean(mu1 - mu0))

def ipw(X, Y, e, trim=(0.05, 0.95)):
    e_clipped = np.clip(e, *trim)
    w1 = X / e_clipped
    w0 = (1 - X) / (1 - e_clipped)
    # Hajek (self-normalized) form, more stable than Horvitz-Thompson.
    return float((w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum())

def aipw(X, Y, mu0, mu1, e, trim=(0.05, 0.95)):
    e_clipped = np.clip(e, *trim)
    score = (mu1 - mu0
             + X * (Y - mu1) / e_clipped
             - (1 - X) * (Y - mu0) / (1 - e_clipped))
    return float(np.mean(score)), float(np.std(score, ddof=1) / np.sqrt(len(score)))

tau_g  = g_comp(mu0_hat, mu1_hat)
tau_ipw = ipw(X, Y, e_hat)
tau_aipw, se_aipw = aipw(X, Y, mu0_hat, mu1_hat, e_hat)

# DML via EconML, with the same nuisance choice.
from econml.dml import LinearDML
dml = LinearDML(
    model_y=GradientBoostingRegressor(random_state=0),
    model_t=GradientBoostingClassifier(random_state=0),
    discrete_treatment=True,
    cv=5,
    random_state=0,
)
dml.fit(Y=Y, T=X, X=None, W=Z)
tau_dml = float(np.atleast_1d(dml.const_marginal_effect()).ravel()[0])
lo, hi  = dml.const_marginal_effect_interval(alpha=0.05)
ci_dml  = (float(np.atleast_1d(lo).ravel()[0]), float(np.atleast_1d(hi).ravel()[0]))

print()
print(f"{'Estimator':<22}{'ATE':>10}{'95% CI':>20}")
print("-" * 52)
print(f"{'G-computation':<22}{tau_g:+.4f}")
print(f"{'IPW (trimmed)':<22}{tau_ipw:+.4f}")
print(f"{'AIPW':<22}{tau_aipw:+.4f}    [{tau_aipw - 1.96*se_aipw:+.4f}, {tau_aipw + 1.96*se_aipw:+.4f}]")
print(f"{'DML (EconML)':<22}{tau_dml:+.4f}    [{ci_dml[0]:+.4f}, {ci_dml[1]:+.4f}]")
print(f"{'Naive (no Z)':<22}{Y[X==1].mean() - Y[X==0].mean():+.4f}    (reference, ignores DAG)")"""),

md("""**How to read the table.**

1. *Direction.* If all four estimators sign the same way, the direction of the effect is consistent across nuisance choices.
2. *Magnitude.* If they cluster within roughly one standard error, the magnitude is consistent — the DML CI is the most defensible single number.
3. *Disagreement.* If G-comp differs from IPW substantially, suspect a misspecified outcome or propensity model. AIPW and DML *should* be close to each other in this binary-treatment, sufficient-sample regime; if they aren't, suspect cross-fit instability."""),

md("""## Part 4 — Diagnostics: overlap and positivity

Positivity says $0 < e(z) < 1$ everywhere in the support. The propensity histogram is the first-line diagnostic."""),

code("""fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(e_hat[X == 1], bins=30, alpha=0.6, label="Treated (X=1)",  density=True)
ax.hist(e_hat[X == 0], bins=30, alpha=0.6, label="Controls (X=0)", density=True)
ax.set_xlabel("Estimated propensity P(X=1 | Z)")
ax.set_ylabel("Density")
ax.set_title(f"Overlap diagnostic for treatment={treatment_sensor}")
ax.legend()
plt.tight_layout()
plt.show()

n_trim = int(((e_hat < 0.05) | (e_hat > 0.95)).sum())
print(f"Wafers in poor-overlap region (e < 0.05 or > 0.95): {n_trim} / {len(e_hat)} ({100*n_trim/len(e_hat):.1f}%)")"""),

md("""**Read the histogram.** Healthy overlap means both treated and control density spans most of $[0,1]$. A bimodal pattern (most mass near 0 for controls and near 1 for treated) flags positivity violations: at extreme propensities, there are few comparison units, and IPW weights blow up.

If the poor-overlap fraction is small (say <5%), the trimmed IPW estimate is defensible. If it's large (>15%), report the estimate *restricted to the common-support region* and say so clearly — the ATE on the original population is no longer identified."""),

md("""## Part 5 — Sensitivity: positivity stress

Drop the wafers in the poor-overlap region and re-run the four estimators. If the estimates move substantially, positivity was binding. If they barely move, the positivity violation was minor and the original ATE stands."""),

code("""mask = (e_hat >= 0.05) & (e_hat <= 0.95)
print(f"Retained {mask.sum()} / {len(mask)} wafers after trimming.")

if mask.sum() < 50 or mask.sum() == len(mask):
    print("Trimming removes too few or too many wafers; skipping the stress run.")
else:
    Xs, Ys, Zs = X[mask], Y[mask], Z[mask]
    mu0_s, mu1_s, e_s = cross_fit_nuisances(Xs, Zs, Ys, K=5, seed=0)
    tau_g_s    = g_comp(mu0_s, mu1_s)
    tau_ipw_s  = ipw(Xs, Ys, e_s)
    tau_aipw_s, _ = aipw(Xs, Ys, mu0_s, mu1_s, e_s)
    print()
    print(f"{'Estimator':<22}{'Full':>10}{'Trimmed':>12}{'Shift':>10}")
    print("-" * 54)
    print(f"{'G-computation':<22}{tau_g:+.4f}{tau_g_s:+.4f}{tau_g_s - tau_g:+.4f}")
    print(f"{'IPW':<22}{tau_ipw:+.4f}{tau_ipw_s:+.4f}{tau_ipw_s - tau_ipw:+.4f}")
    print(f"{'AIPW':<22}{tau_aipw:+.4f}{tau_aipw_s:+.4f}{tau_aipw_s - tau_aipw:+.4f}")"""),

md("""## Part 6 — Robustness: add period to the control set

Lab 1B established that `period` is a back-door confounder. The baseline DML above excluded it. As a robustness check, re-fit DML with period dummies appended to Z. If the estimate is stable, the original DAG was adequate. If it shifts substantially, period was non-trivial — re-anchor the analysis on a control set that includes it."""),

code("""period_dummies = pd.get_dummies(df["period"], drop_first=True).astype(float).values
Z_aug = np.hstack([Z, period_dummies])

dml_aug = LinearDML(
    model_y=GradientBoostingRegressor(random_state=0),
    model_t=GradientBoostingClassifier(random_state=0),
    discrete_treatment=True,
    cv=5,
    random_state=0,
)
dml_aug.fit(Y=Y, T=X, X=None, W=Z_aug)
tau_dml_aug = float(np.atleast_1d(dml_aug.const_marginal_effect()).ravel()[0])
lo_a, hi_a  = dml_aug.const_marginal_effect_interval(alpha=0.05)
ci_dml_aug  = (float(np.atleast_1d(lo_a).ravel()[0]), float(np.atleast_1d(hi_a).ravel()[0]))

print(f"DML on Z              : {tau_dml:+.4f}    [{ci_dml[0]:+.4f}, {ci_dml[1]:+.4f}]")
print(f"DML on Z + period     : {tau_dml_aug:+.4f}    [{ci_dml_aug[0]:+.4f}, {ci_dml_aug[1]:+.4f}]")
print(f"Shift from adding period: {tau_dml_aug - tau_dml:+.4f}")"""),

md("""## Part 7 — Decision

Three bullets, the deliverable an engineer would read:

1. **Estimator agreement.** Read the table from Part 3: if G-comp, IPW, AIPW, and DML cluster within roughly one CI width, the ATE under the assumed DAG is *internally consistent* — switching nuisance models doesn't switch the answer. The reported number is the DML estimate with its 95% CI.

2. **Positivity and overlap.** Part 4's histogram and Part 5's trimming check tell us whether the estimate generalises to the full population or only to the common-support subset. If trimming barely moved the estimate (Part 5 \"shift\" near zero), the original ATE is valid; if it moved substantially, restrict the recommendation to the overlap region.

3. **Period robustness.** If Part 6's period-augmented estimate matches the baseline within CI, the assumed DAG was adequate. If it doesn't, the period adjustment from Lab 1B was binding — re-anchor on the augmented DAG and treat the baseline estimate as misleading.

The point estimate alone is never the deliverable; the deliverable is the table + the diagnostics + the bullet that names the binding assumption."""),

md("""## Reflection

**The four-estimator gauntlet replaces the synthetic oracle.** Lab 5A told us that under correct nuisance specification all four estimators target the same ATE. Lab 5B uses that fact in reverse: under real data with unknown nuisance specifications, *agreement across the four* is the strongest non-oracular evidence we can produce that our estimate is not an artefact of any single modelling choice. Disagreement is informative — it tells us *which* nuisance is misspecified.

**Sensitivity bounds replace certainty.** The positivity-stress run and the period-augmentation run quantify how much the estimate moves under specific deviations from the baseline DAG. A robust estimate is not one that you trust *because* of these checks; it is one that *survives* these checks. The report names the deviation, the diagnostic, and the result — that is what makes the analysis defensible."""),

md("""## What's next

Lab 6B uses the same DML machinery but generalises from one ATE to a *function* of effects — CATE per station, on Bosch Production Line Performance data. The diagnostic tools (overlap, sensitivity, robustness) carry over; the deliverable becomes a heterogeneity report instead of a single number."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch05" / "lab05b.ipynb", cells)
print("Built lab05b.ipynb")
