"""Build labs/ch04/lab04_iv_did_rdd.ipynb — IV, DID, and RDD."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 4 — Quasi-Experimental Designs: IV, DID, and RDD

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch04/lab04_iv_did_rdd.ipynb)

**Companion lab to Chapter 4.** Three identification strategies for when neither back-door nor front-door applies: an *instrumental variable* design, a *difference-in-differences* design, and a *regression-discontinuity* design. Each is built from a manufacturing scenario and verified against a known SCM.

Chapters 1–3 dealt with identification when an adjustment set is available. Chapter 4 turns to situations where the back-door is blocked from view but a *design feature* of the data — a quasi-randomization — lets you identify the effect anyway. Three classical patterns recur in manufacturing operations: a randomly-assigned routing decision (IV), a phased rollout that creates pre/post + treated/control structure (DID), and an administrative threshold that randomly-ish separates units (RDD)."""),

md("""## What you'll do

1. **IV**: A randomly-assigned rework lane is an instrument for whether a part actually goes through rework. We will fit the Wald estimator and 2SLS and recover the LATE.
2. **DID**: A new etch recipe rolls out to Tool Group A in Q3 while Group B stays on the old recipe. We will fit the 2×2 DID and inspect parallel-trends.
3. **RDD**: Parts with a measurement above a fixed cutoff are passed to assembly; parts below are reworked. We will use local linear regression near the cutoff to estimate the treatment effect at the threshold.

Each section has its own SCM, its own truth, and its own estimator. The same lab pattern applies — generate data from an SCM with known coefficients, apply the estimator, verify against the analytical truth."""),

md("""## Setup"""),

code("""# Colab: install linearmodels (for IV via IV2SLS) and CausalPy (for DID/RDD).
%pip install --quiet linearmodels causalpy 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — Instrumental Variables

A measurement station tests parts coming off a stamping line. Operationally, parts can go to one of two rework lanes; for capacity reasons, a *random* lane assignment determines which lane the part is queued for. The two lanes have different compliance rates: in lane A, 80% of queued parts actually get reworked; in lane B, only 30%. The actual rework decision depends on lane assignment *and* on a latent quality factor (some parts are nominally borderline and the inspector overrides the queue). The latent quality also affects final yield directly. So we have:

- $Z$ — lane assignment (random; A or B). This is the **instrument**.
- $D$ — actual rework status (1 = reworked, 0 = not).
- $U$ — latent quality factor (unobserved).
- $Y$ — downstream yield outcome.

DAG:

```
   Z ──► D ──► Y
         ▲     ▲
         │     │
         U ────┘
```

The instrument $Z$ affects $Y$ *only* through $D$ (exclusion restriction). $U$ confounds the $D \\to Y$ relationship. The IV identifying assumption — exclusion + relevance + monotonicity — says we can recover the LATE (effect on compliers) from the ratio of $\\text{cov}(Y, Z) / \\text{cov}(D, Z)$, or equivalently, the 2SLS estimator."""),

code("""# IV SCM — step 1: instrument and latent confounder
n = 10_000

Z = rng.binomial(1, 0.5, n)         # Lane assignment, random (the instrument): 0 = lane B, 1 = lane A
U = rng.normal(0, 1, n)             # Latent quality (unobserved confounder)
print(f"P(Z = lane A): {Z.mean():.3f}   (should be ~0.5, random assignment)")"""),

md("""$Z$ is the *exogenous* instrument — a randomized lane assignment that the lab data can audit. $U$ is the latent quality that we cannot measure but that confounds the $D \\to Y$ relationship."""),

code("""# IV SCM — step 2: actual rework decision (treatment)
# Compliance rate depends on lane assignment; latent quality nudges some borderline cases.
# Lane A: 80% baseline compliance.  Lane B: 30%.  +0.1 * U adjusts at the margin.
prob_rework = np.where(Z == 1, 0.8, 0.3) + 0.1 * U
prob_rework = np.clip(prob_rework, 0.01, 0.99)
D = rng.binomial(1, prob_rework)
print(f"P(D=1 | Z=lane B): {D[Z==0].mean():.3f}")
print(f"P(D=1 | Z=lane A): {D[Z==1].mean():.3f}")
print(f"  Relevance gap:   {D[Z==1].mean() - D[Z==0].mean():+.3f}   (large gap = strong instrument)")"""),

md("""The lane assignment shifts rework probability by ~50 percentage points — this is what the IV will exploit."""),

code("""# IV SCM — step 3: outcome (yield)
# True effect of rework on yield: +2.0.  U also affects yield directly (the confounding path).
true_effect = 2.0
Y = 70 + true_effect * D + 1.5 * U + rng.normal(0, 1, n)

iv_data = pd.DataFrame({"Z": Z, "D": D, "U": U, "Y": Y})
print(f"True ATE of rework on yield: {true_effect:+.3f}")
print(f"Dataset shape: {iv_data.shape}")"""),

md("""The instrument is *relevant*: lane A has ~50 percentage points higher rework rate than lane B. We can use this differential exposure to identify the effect."""),

code("""# Wald estimator: cov(Y, Z) / cov(D, Z), or equivalently
#   [E[Y | Z=1] - E[Y | Z=0]] / [E[D | Z=1] - E[D | Z=0]]
num = iv_data.loc[Z == 1, "Y"].mean() - iv_data.loc[Z == 0, "Y"].mean()
den = iv_data.loc[Z == 1, "D"].mean() - iv_data.loc[Z == 0, "D"].mean()
wald = num / den

# Naive OLS: Y ~ D, ignoring confounding by U
naive_iv = LinearRegression().fit(iv_data[["D"]], iv_data["Y"]).coef_[0]

print(f"Naive OLS (Y ~ D):       {naive_iv:+.3f}  (biased by U)")
print(f"Wald estimator:          {wald:+.3f}")
print(f"True LATE (= ATE here):  {true_effect:+.3f}")"""),

md("""The Wald estimator recovers the truth. The naive OLS is biased because $U$ (latent quality) affects both $D$ and $Y$. The IV's exclusion restriction — $Z$ affects $Y$ only via $D$ — is what makes the estimator work. In this synthetic case the LATE equals the ATE because rework's effect is homogeneous; with heterogeneous effects, the IV identifies the *compliers'* average effect, which is a sub-population estimate."""),

code("""# Two-stage least squares (2SLS): equivalent to Wald for binary Z, generalizes
# to continuous Z or with covariates.
# Stage 1: D ~ Z
stage1 = LinearRegression().fit(iv_data[["Z"]], iv_data["D"])
D_hat = stage1.predict(iv_data[["Z"]])
# Stage 2: Y ~ D_hat
stage2 = LinearRegression().fit(D_hat.reshape(-1, 1), iv_data["Y"])
two_sls = stage2.coef_[0]
print(f"2SLS estimate:           {two_sls:+.3f}  (matches Wald exactly for binary Z)")"""),

md("""**Diagnostic: weak-instrument first-stage F-statistic.** When the instrument has a small effect on the treatment ($Z \\to D$), the Wald denominator is small and the estimator is highly noisy. The conventional diagnostic is the first-stage F-statistic; rule-of-thumb threshold is F > 10 (Stock-Yogo)."""),

code("""# First-stage F-statistic. With one instrument, F = t^2.
slope = stage1.coef_[0]
y_resid = iv_data["D"] - stage1.predict(iv_data[["Z"]])
mse = (y_resid ** 2).sum() / (n - 2)
se_slope = np.sqrt(mse / ((iv_data["Z"] - iv_data["Z"].mean()) ** 2).sum())
F = (slope / se_slope) ** 2
print(f"First-stage F-statistic: {F:.1f}  (Stock-Yogo threshold: 10)")"""),

md("""F is far above 10 — this is a strong instrument. With a real-world instrument that nudges the treatment by only a few percentage points, F can drop to single digits and the IV estimate becomes unreliable.

**Library cross-check with `linearmodels.IV2SLS`.** The same estimate via the standard IV library:"""),

code("""try:
    from linearmodels.iv import IV2SLS
    have_lm = True
except ImportError:
    have_lm = False
    print("linearmodels not installed. The %pip install at the top should have fixed this in Colab.")

if have_lm:
    iv_data["const"] = 1.0
    res = IV2SLS(dependent=iv_data["Y"],
                 exog=iv_data[["const"]],
                 endog=iv_data[["D"]],
                 instruments=iv_data[["Z"]]).fit(cov_type="robust")
    print(res.summary)
    print(f"\\nManual Wald: {wald:+.3f}    linearmodels IV2SLS: {res.params['D']:+.3f}")"""),

md("""## Part 2 — Difference-in-Differences

A fab rolls out a new etch recipe to *Tool Group A* in Q3, but Group B stays on the old recipe through Q3. We observe yield in Q2 (pre) and Q3 (post) for both groups. DID compares the Q2→Q3 yield change for Group A against the same change for Group B.

The DID identifying assumption is **parallel trends**: in the absence of the recipe change, both groups would have followed the same time trend. We can't test this directly (Group A's Q3 yield under the old recipe is the counterfactual), but pre-period inspection helps."""),

code("""# DID SCM
n_tools_per_group = 50
periods = [0, 1]  # 0 = pre (Q2), 1 = post (Q3)
groups  = ["A", "B"]

records = []
# Tool-level fixed effects (different baseline yields per tool)
tool_id = 0
for g in groups:
    for k in range(n_tools_per_group):
        # Tool-level baseline yield, with Group A slightly lower on average
        baseline = 75 + (-0.5 if g == "A" else 0.0) + rng.normal(0, 1)
        # Time trend: yield drifts up by 0.8 per period for all tools
        for t in periods:
            yield_value = baseline + 0.8 * t
            # The treatment: Group A in post-period gets the new recipe
            if g == "A" and t == 1:
                yield_value += 1.5  # the true treatment effect
            yield_value += rng.normal(0, 0.5)
            records.append({"tool": tool_id, "group": g, "period": t, "Y": yield_value})
        tool_id += 1

did_data = pd.DataFrame(records)
did_data.head()"""),

code("""# Compute means in each (group, period) cell
means = did_data.groupby(["group", "period"])["Y"].mean().unstack()
print("Mean Y by group and period:")
print(means.round(3))
print()

did_estimate = (means.loc["A", 1] - means.loc["A", 0]) - (means.loc["B", 1] - means.loc["B", 0])
naive_diff = means.loc["A", 1] - means.loc["B", 1]   # post-period difference, ignores baselines
print(f"Naive Q3 difference (A - B):           {naive_diff:+.3f}")
print(f"DID estimate ((A1 - A0) - (B1 - B0)):  {did_estimate:+.3f}")
print(f"True treatment effect:                 +1.500")"""),

md("""The naive Q3 difference understates the effect because Group A had a slightly lower baseline. DID accounts for the baseline level (by differencing within group) and the time trend (by differencing across periods). The remaining estimate is the treatment effect.

Equivalently, DID can be fit as a regression: $Y = \\alpha + \\beta_{\\text{group}} \\cdot \\mathbf 1[\\text{A}] + \\gamma_{\\text{period}} \\cdot \\mathbf 1[\\text{post}] + \\tau \\cdot \\mathbf 1[\\text{A, post}] + \\varepsilon$. The coefficient $\\tau$ on the interaction is the DID estimate."""),

code("""# Regression form of DID
did_data["A"]    = (did_data["group"] == "A").astype(int)
did_data["post"] = did_data["period"].astype(int)
did_data["D"]    = did_data["A"] * did_data["post"]

X = did_data[["A", "post", "D"]].values
reg = LinearRegression().fit(X, did_data["Y"].values)
print(f"Regression DID:    tau (D coef) = {reg.coef_[2]:+.3f}")
print(f"Difference form:   tau          = {did_estimate:+.3f}")"""),

md("""**Visual parallel-trends check.** Plot the (group, period) means. If the line connecting Group A's pre-mean to post-mean is parallel to Group B's line in the *absence* of the intervention, parallel trends is plausible. A non-parallel pre-period trend is evidence against the assumption (and a reason to use more sophisticated estimators — Callaway-Sant'Anna, etc., as discussed in Chapter 4)."""),

code("""fig, ax = plt.subplots()
for g, marker in [("A", "o"), ("B", "s")]:
    sub = did_data[did_data["group"] == g].groupby("period")["Y"].mean()
    ax.plot(sub.index, sub.values, marker=marker, label=f"Group {g}",
            linewidth=2, markersize=10)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Q2 (pre)", "Q3 (post)"])
ax.set_ylabel("Mean yield")
ax.set_title("DID: Group A receives new recipe in Q3")
ax.legend()
plt.show()"""),

md("""**Library cross-check with CausalPy.** `causalpy.DifferenceInDifferences` is a PyMC-backed implementation that returns a full posterior over the treatment effect rather than a point estimate."""),

code("""try:
    import causalpy as cp
    have_cp = True
except ImportError:
    have_cp = False
    print("CausalPy not installed. %pip install causalpy in Colab.")

if have_cp:
    df_did = did_data.copy()
    df_did = df_did.rename(columns={"period": "t", "tool": "unit"})
    df_did["group"] = df_did["A"]
    df_did["post_treatment"] = df_did["t"]
    res = cp.DifferenceInDifferences(
        df_did, formula="Y ~ 1 + group*post_treatment",
        time_variable_name="t", group_variable_name="group",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={"chains": 2, "draws": 500, "tune": 200,
                           "progressbar": False, "random_seed": 0},
        ),
    )
    cp_mean = float(np.asarray(res.causal_impact).mean())
    cp_q05  = float(np.quantile(np.asarray(res.causal_impact), 0.05))
    cp_q95  = float(np.quantile(np.asarray(res.causal_impact), 0.95))
    print(f"Manual DID:        tau = {did_estimate:+.3f}")
    print(f"CausalPy posterior: mean = {cp_mean:+.3f}, 90% CI = [{cp_q05:+.3f}, {cp_q95:+.3f}]")"""),

md("""## Part 3 — Regression Discontinuity

A measurement station inspects parts: a continuous *measurement* $Z$ is recorded for each part. If $Z \\geq c$, the part is reworked; if $Z < c$, it passes directly to assembly. Downstream yield is the outcome. Parts just above and just below the cutoff are nearly identical in unobserved characteristics (by continuity), but differ in their treatment assignment. RDD recovers the treatment effect *at the cutoff*."""),

code("""# RDD SCM
n_rdd = 2000
cutoff = 50.0

# Forcing variable Z, with some U-correlation (a latent quality)
U = rng.normal(0, 1, n_rdd)
Z = 50 + 5 * U + rng.normal(0, 3, n_rdd)

# Treatment: rework if Z >= cutoff
D = (Z >= cutoff).astype(int)

# Outcome: yield depends smoothly on Z (continuity assumption) and on rework treatment
true_rdd_effect = 3.0
Y = 60 + 0.3 * (Z - cutoff) + true_rdd_effect * D + 0.5 * U + rng.normal(0, 1, n_rdd)

rdd_data = pd.DataFrame({"Z": Z, "D": D, "Y": Y})
rdd_data.head()"""),

code("""# Estimator: local linear regression on each side of the cutoff within a bandwidth h.
def rdd_local_linear(data, c, h):
    left  = data[(data["Z"] < c) & (data["Z"] >= c - h)]
    right = data[(data["Z"] >= c) & (data["Z"] <= c + h)]
    # Linear regression of Y on (Z - c) in each window
    lr_left  = LinearRegression().fit(left[["Z"]].values - c, left["Y"].values)
    lr_right = LinearRegression().fit(right[["Z"]].values - c, right["Y"].values)
    # Intercepts at Z = c give the limits from below and above
    y_lim_minus = lr_left.predict(np.array([[0.0]]))[0]
    y_lim_plus  = lr_right.predict(np.array([[0.0]]))[0]
    return y_lim_plus - y_lim_minus

# Estimates at three bandwidths
for h in [2, 5, 10]:
    est = rdd_local_linear(rdd_data, cutoff, h)
    print(f"Bandwidth h = {h:2d}:  RDD estimate = {est:+.3f}")
print(f"True effect at cutoff:        {true_rdd_effect:+.3f}")"""),

md("""All three bandwidths recover the truth (within sampling noise). Narrow bandwidths are less biased but noisier; wide bandwidths are smoother but include more units further from the cutoff (where the smooth-relationship-in-Z assumption may break down). A standard diagnostic: report the estimate at the CCT-optimal bandwidth, and at $h/2$ and $2h$ — wildly different estimates across bandwidths is a warning sign."""),

code("""# Visualize the discontinuity
fig, ax = plt.subplots()
ax.scatter(rdd_data["Z"], rdd_data["Y"], alpha=0.2, s=8, label="Data")

# Local linear fits on each side
for side in ["below", "above"]:
    mask = (rdd_data["Z"] < cutoff) if side == "below" else (rdd_data["Z"] >= cutoff)
    sub = rdd_data[mask & (np.abs(rdd_data["Z"] - cutoff) <= 5)]
    lr = LinearRegression().fit(sub[["Z"]] - cutoff, sub["Y"])
    z_range = np.linspace(sub["Z"].min(), sub["Z"].max(), 50)
    ax.plot(z_range, lr.predict((z_range - cutoff).reshape(-1, 1)),
            linewidth=2.5, label=f"Local linear ({side} cutoff)")

ax.axvline(cutoff, linestyle="--", color="black", alpha=0.5, label="Cutoff")
ax.set_xlabel("Forcing variable Z (measurement)")
ax.set_ylabel("Outcome Y (yield)")
ax.set_title(f"RDD: discontinuity at Z = {cutoff}")
ax.legend()
plt.show()"""),

md("""The visible jump at $Z = 50$ is the treatment effect at the cutoff. The continuity of the data around the threshold (no spike in density, no shift in covariates) is what makes the design credible.

**Library cross-check with CausalPy.** `causalpy.RegressionDiscontinuity` runs a PyMC-based fit and returns a posterior on the discontinuity."""),

code("""if have_cp:
    # CausalPy expects the running variable named 'x' and outcome 'y'.
    df_rdd = rdd_data.rename(columns={"Z": "x", "Y": "y"}).copy()
    df_rdd["treated"] = (df_rdd["x"] >= cutoff)
    res = cp.RegressionDiscontinuity(
        df_rdd, formula="y ~ 1 + x + treated",
        treatment_threshold=cutoff,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={"chains": 2, "draws": 500, "tune": 200,
                           "target_accept": 0.95,
                           "progressbar": False, "random_seed": 0},
        ),
    )
    cp_rdd = float(np.asarray(res.discontinuity_at_threshold).mean())
    print(f"Manual RDD (h=5):            {rdd_local_linear(rdd_data, cutoff, 5):+.3f}")
    print(f"CausalPy RDD posterior mean: {cp_rdd:+.3f}")
    print(f"True effect at cutoff:        {true_rdd_effect:+.3f}")"""),

md("""**McCrary density test.** A diagnostic: check that the density of $Z$ is smooth across the cutoff. A spike just above the threshold (parts nudged above to get reworked, or just below to avoid rework) is a sign of *manipulation* that defeats RDD's exchangeability assumption."""),

code("""# Histogram of Z near the cutoff, with cutoff marked
fig, ax = plt.subplots()
ax.hist(rdd_data["Z"], bins=50, color="steelblue", alpha=0.7)
ax.axvline(cutoff, linestyle="--", color="red", label="Cutoff")
ax.set_xlabel("Z"); ax.set_ylabel("Count")
ax.set_title("Density of forcing variable (McCrary diagnostic)")
ax.legend()
plt.show()"""),

md("""No spike — the density is smooth across the cutoff. (A formal McCrary test would fit a discontinuous density estimator and test the jump; the visual inspection here suffices.)"""),

md("""## Part 4 — Choosing among the designs

Each design has a specific data structure that makes it possible:

| Design | Required data structure | What it identifies | Typical manufacturing setting |
|---|---|---|---|
| **IV** | An exogenous variable that affects treatment but not outcome | LATE (effect on compliers) | Randomized routing, queue assignment, lottery |
| **DID** | Pre/post + treated/control panel | ATT (effect on the treated) | Phased recipe rollout, tool-by-tool deployment |
| **RDD** | A threshold-based assignment | Treatment effect at the cutoff | Inspection thresholds, ranking-based prioritization |

The choice is not "which estimator is best" but "which design feature is present in the data". IV needs a credible exclusion restriction. DID needs parallel pre-trends. RDD needs a sharp threshold with no manipulation. In the field, you scout the data for these structures rather than picking an estimator first."""),

md("""## Reflection

**Identification comes from the design, not the estimator.** Each section's estimator is mechanically simple — Wald is a ratio, DID is a 2×2 mean comparison, RDD is two local regressions. The *identifying power* comes from the structural assumption: random assignment for IV, parallel trends for DID, continuity for RDD.

**Each design has a falsifiable diagnostic.** IV: first-stage F (relevance). DID: pre-trends plot (parallel trends). RDD: McCrary density (no manipulation). Failing the diagnostic is a reason to abandon the design, not to ignore it.

**These designs identify *local* effects.** IV gives the LATE (compliers). DID gives the ATT (post-treatment Group A). RDD gives the effect at the cutoff. Extrapolating to the full population requires additional assumptions about effect heterogeneity."""),

md("""## Exercises

1. **Weak instrument.** Reduce the IV relevance: change the lane compliance from 80%/30% to 40%/35%. Recompute the Wald estimator and first-stage F. What happens to the estimate's noise and the F-statistic? At what F-value does the estimate become unreliable?

   <details><summary>Solution</summary>

   With compliance 40%/35%, the relevance gap shrinks from 0.5 to ~0.05. The Wald denominator becomes tiny and the estimate explodes (high variance, often heavily biased). First-stage F drops from ~3300 to single digits. Rule of thumb (Stock-Yogo): F < 10 → weak instrument, IV unreliable. CIs widen dramatically; reported point estimates can be far from truth even with large $n$.
   </details>

2. **Pre-trend violation.** Modify the DID SCM so Group A's baseline drifts down by 0.5 per period (a non-parallel pre-trend). Refit the 2×2 DID. How biased is the estimate? Plot the trends to see why.

   <details><summary>Solution</summary>

   Group A's pre-trend gets absorbed into the DID estimate as if it were treatment effect. With Group A drifting down by 0.5/period independently of treatment, the DID estimator returns ~`+1.0` instead of the true `+1.5` — a `-0.5` bias from the differential pre-trend. The diagnostic is an event-study specification with pre-period coefficients; non-zero pre-period coefficients are the smoking gun.
   </details>

3. **Manipulation at the cutoff.** Modify the RDD SCM so 30% of parts just above the cutoff have their measurement nudged down to just below (to avoid rework). Refit the local linear estimator. Plot the McCrary density. Diagnose the bias.

   <details><summary>Solution</summary>

   Nudging creates a density spike just below the cutoff and a corresponding drop just above. The McCrary density test rejects (clear discontinuity in $f(Z)$ at $c$). The RDD estimator now compares manipulated units below the cutoff against non-manipulated units above — biased toward zero because the "below" group is enriched with would-be-treated units. Always run the McCrary test before reporting RDD.
   </details>

4. **Combine designs.** Imagine a fab rolls out a new recipe to Group A in Q3, *and* there's a measurement-threshold-based rework rule that applies to all tools. The RDD identifies the rework effect; the DID identifies the recipe effect. Sketch how you would estimate both effects from the same dataset.

   <details><summary>Solution</summary>

   Two separate identification strategies on the same data: (a) RDD on the rework-threshold variable, ignoring the recipe rollout (the rework decision is local around the cutoff and orthogonal to recipe in expectation); (b) DID on the recipe rollout, ignoring the rework rule (parallel-trend assumption applies on average over the rework decision). Report both. If recipe interacts with rework — e.g., Recipe B + rework gives a different outcome than Recipe A + rework — fit a triple-difference (DiDiD) or a CATE estimator (Chapter 6) that respects both designs.
   </details>"""),

md("""## What's next

Lab 5 turns from identification to *estimation*: when the back-door is identifiable, what's the best estimator? We will compare matching, IPW, doubly-robust, and double machine learning on the same data, with a focus on how each handles complex outcome surfaces and imbalanced treatment groups."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch04" / "lab04_iv_did_rdd.ipynb", cells)
print("Built lab04_iv_did_rdd.ipynb")
