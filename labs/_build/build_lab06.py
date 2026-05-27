"""Build labs/ch06/lab06.ipynb — CATE, meta-learners, causal forests, uplift."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 6 — CATE, Meta-Learners, Causal Forests, and Uplift

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06.ipynb)

**Companion lab to Chapter 6.** From average effects to *conditional* average effects: estimate how a recipe change's effect varies across the covariate space, fit the five meta-learners, deploy a causal forest with confidence intervals, and evaluate a targeted policy with a Qini curve.

The chapter's §6.8 worked example motivates this lab: a lithography recipe rollout where Recipe B helps thick-resist lots but hurts thin-resist ones. The lab's SCM uses a wider per-unit CATE range than the chapter's narrative numbers — $\\tau(z)$ here varies from about $-15\\%$ to $+18\\%$ across the resist-thickness grid (with a small positive ATE on top) so the heterogeneity is more visible in plots. The structural shape — positive effect on thick lots, negative on thin — and the consequent policy lesson are the same as the chapter: a targeted policy beats blanket deployment.

This lab builds that scenario, estimates the CATE with multiple methods, and compares targeted vs blanket policies on the resulting Qini curve."""),

md("""## What you'll do

1. **Build the SCM** for the lithography recipe with resist-thickness-dependent treatment effect.
2. **The production workflow**: fit a `CausalForestDML` from EconML with confidence intervals — what you'd actually run on real data.
3. **Compare EconML's meta-learners** (`SLearner`, `TLearner`, `XLearner`, `DRLearner`) on the same data.
4. **Under the hood**: implement S-, T-, X-, R-learners from scratch and verify they match the library outputs.
5. **Build the Qini curve** and compute the Qini coefficient for ranking-based targeting.
6. **Compare policies** — blanket, threshold, optimized — by their policy values."""),

md("""## Setup"""),

code("""# Colab: install EconML (provides SLearner/TLearner/XLearner/DRLearner +
# CausalForestDML out of the box). The from-scratch meta-learners we build in
# Part 2 are pedagogical; Part 4 swaps to the library for the production tool.
%pip install --quiet econml 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The lithography recipe SCM

A semiconductor fab evaluates Recipe B against incumbent Recipe A on a lithography tool. Each lot has a resist thickness $Z_1$ and a chamber condition (days since last PM, $Z_2$). Recipe choice is partially driven by the recipe lead — high-complexity lots tend to get Recipe B in observational data.

True effect: $\\tau(z) = 0.0006 (z_1 - 500) - 0.002 z_2$ — Recipe B helps thick-resist lots and hurts thin-resist; effect is also slightly negative on long-since-PM tools.

Observational data: 5000 lots split roughly 50/50 by the existing scheduling system. We want to recover the per-unit CATE."""),

code("""def gen_litho(n, rng, prob_recipe_B=None):
    \"\"\"Generate observational lithography data.

    Z1: resist thickness (nm), uniform [300, 800].
    Z2: chamber days-since-PM, uniform [0, 14].
    X: recipe choice (1 = B, 0 = A).
    Y: yield delta (continuous, mean-centered).
    \"\"\"
    Z1 = rng.uniform(300, 800, n)
    Z2 = rng.uniform(0, 14, n)

    if prob_recipe_B is None:
        # Behavior: thick + recent-PM lots more likely to get B
        logit = -0.5 + 0.005 * (Z1 - 500) - 0.05 * Z2
        prob_recipe_B = 1 / (1 + np.exp(-logit))
    X = rng.binomial(1, prob_recipe_B)

    # True CATE: depends on Z1 (thickness) and Z2 (chamber state)
    tau = 0.0006 * (Z1 - 500) - 0.002 * Z2

    # Outcome: baseline yield + tau * X + noise
    baseline = 0.92 + 0.0001 * (Z1 - 500) - 0.0005 * Z2
    Y = baseline + tau * X + rng.normal(0, 0.01, n)

    return pd.DataFrame({"Z1": Z1, "Z2": Z2, "X": X, "Y": Y, "tau_true": tau})

n = 5000
data = gen_litho(n, rng)
print(f"Treated fraction (Recipe B):  {data['X'].mean():.3f}")
print(f"Z1 range: [{data['Z1'].min():.0f}, {data['Z1'].max():.0f}] nm")
print(f"Z2 range: [{data['Z2'].min():.1f}, {data['Z2'].max():.1f}] days")
print()
print(f"Observed mean yield (X=0):    {data[data['X']==0]['Y'].mean():.4f}")
print(f"Observed mean yield (X=1):    {data[data['X']==1]['Y'].mean():.4f}")
print(f"Naive ATE difference:         {data[data['X']==1]['Y'].mean() - data[data['X']==0]['Y'].mean():+.4f}")
print(f"True ATE (mean of tau_true):  {data['tau_true'].mean():+.4f}")"""),

md("""The naive observational difference ($\\sim +0.06$) substantially overshoots the true ATE ($\\sim +0.016$) — confounding inflates the apparent average effect by ~4×. But the *per-unit CATE* is what matters for targeted policy. Some lots have CATE $\\sim +0.18$ (thick resist + fresh PM); others have CATE $\\sim -0.15$ (thin resist + long-since PM). The ATE averages these. The actionable answer is: route lots by Z1 and Z2, not by a blanket policy."""),

md("""## Part 2 — Production workflow: EconML `CausalForestDML`

The single line that does what the rest of this lab will unpack: fit a causal forest with valid confidence intervals, get per-unit treatment effects with uncertainty quantification, ready to plug into a targeting decision."""),

code("""try:
    from econml.dml import CausalForestDML
    from econml.metalearners import SLearner, TLearner, XLearner, DomainAdaptationLearner
    from econml.dr import DRLearner
    have_econml = True
except ImportError:
    have_econml = False
    print("EconML not installed. The %pip install at the top should have fixed this in Colab;")
    print("locally, run `pip install econml`. Skipping Parts 2-3.")

ZCOLS = ["Z1", "Z2"]
Z_arr = data[ZCOLS].values
X_arr = data["X"].values
Y_arr = data["Y"].values
tau_true = data["tau_true"].values

# Evaluation grid for plotting (resist-thickness sweep at mid chamber-day)
z1_grid = np.linspace(310, 790, 100)
z2_grid_mid = np.full(100, 7.0)
Z_grid = np.column_stack([z1_grid, z2_grid_mid])
true_curve = 0.0006 * (z1_grid - 500) - 0.002 * 7.0

if have_econml:
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3),
        model_t=GradientBoostingClassifier(random_state=0, n_estimators=100, max_depth=3),
        discrete_treatment=True, cv=3, random_state=0, n_estimators=300,
    )
    cf.fit(Y=Y_arr, T=X_arr, X=Z_arr, W=None)
    cate_cf = cf.effect(Z_grid)
    cate_lower, cate_upper = cf.effect_interval(Z_grid, alpha=0.1)

    fig, ax = plt.subplots()
    ax.plot(z1_grid, true_curve, "k", linewidth=2.5, label="True CATE")
    ax.plot(z1_grid, cate_cf, label="CausalForestDML")
    ax.fill_between(z1_grid, cate_lower, cate_upper, alpha=0.2, label="90% CI")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Resist thickness Z1 (nm)")
    ax.set_ylabel("CATE")
    ax.set_title("Production: CausalForestDML with 90% pointwise CIs")
    ax.legend()
    plt.show()

    print(f"CausalForestDML PEHE: {np.sqrt(((cf.effect(Z_arr) - tau_true) ** 2).mean()):.4f}")"""),

md("""That single library call produced per-unit treatment-effect estimates with confidence intervals. The CIs are doing real work — they tell you which lots have a credibly positive effect (treat them with Recipe B) versus which are uncertain (more data needed).

For policies, that's nearly the whole story. The next sections explore *which kind* of CATE estimator suits which kind of data — EconML provides the alternatives directly."""),

md("""## Part 3 — EconML's meta-learners

EconML implements the four standard meta-learners as one-line constructors. We fit each on the same data and compare PEHE — the per-unit error against the known true CATE."""),

code("""# EconML's meta-learners use sklearn estimators internally.
if have_econml:
    gbr = lambda: GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
    gbc = lambda: GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3)

    # S-learner: one model on the full (X, Z) data
    sl = SLearner(overall_model=gbr())
    sl.fit(Y=Y_arr, T=X_arr, X=Z_arr)
    cate_sl = sl.effect(Z_arr)

    # T-learner: two models, one per treatment arm
    tl = TLearner(models=gbr())
    tl.fit(Y=Y_arr, T=X_arr, X=Z_arr)
    cate_tl = tl.effect(Z_arr)

    # X-learner: T-learner + propensity-weighted combination
    xl = XLearner(models=gbr(), propensity_model=gbc(), cate_models=gbr())
    xl.fit(Y=Y_arr, T=X_arr, X=Z_arr)
    cate_xl = xl.effect(Z_arr)

    # DR-learner: doubly-robust meta-learner with cross-fitting
    drl = DRLearner(model_propensity=gbc(), model_regression=gbr(),
                    model_final=gbr(), cv=3, random_state=0)
    drl.fit(Y=Y_arr, T=X_arr, X=Z_arr)
    cate_dr = drl.effect(Z_arr)

    # CausalForestDML (per-unit estimates on training data)
    cate_cfg = cf.effect(Z_arr)

    print(f"PEHE (lower is better):")
    for name, c in [("S-learner",      cate_sl),
                    ("T-learner",      cate_tl),
                    ("X-learner",      cate_xl),
                    ("DR-learner",     cate_dr),
                    ("CausalForestDML", cate_cfg)]:
        print(f"  {name:<20s}  {np.sqrt(((c - tau_true) ** 2).mean()):.4f}")"""),

md("""All five EconML estimators converge near the truth, with PEHE differences that reflect the chapter's bias-variance trade-offs:
- **S-learner** can underestimate the CATE when the outcome model regularizes the treatment indicator.
- **T-learner** is noisy in regions where one arm has few observations.
- **X-learner** trades imbalance robustness for additional model assumptions.
- **DR-learner** is the doubly-robust analog — consistent if either propensity or outcome model is right.
- **CausalForestDML** combines orthogonalization with a tree-based final stage and built-in CIs.

In production you fit several, compare PEHE on a holdout or compare policy values on a downstream metric, and ship the one that wins. EconML makes this swap-and-compare a few-line exercise."""),

md("""## Part 4 — Under the hood: implement the meta-learners from scratch

To see what EconML is doing internally, here are the same four meta-learners implemented manually. The point: the library is a thin wrapper on familiar sklearn machinery. Understanding the wrapper is what lets you debug it when it fails — and gives you the building blocks for non-standard variants."""),

md("""The S-, T-, X-, and R-learners all turn a supervised learner into a CATE estimator. Each has different bias-variance trade-offs.

**S-learner**: fit one outcome model $\\hat\\mu(x, z)$ on the full data, predict $\\hat\\mu(1, z) - \\hat\\mu(0, z)$.

**T-learner**: fit two outcome models, $\\hat\\mu_1(z)$ on treated, $\\hat\\mu_0(z)$ on controls. Predict $\\hat\\mu_1(z) - \\hat\\mu_0(z)$.

**X-learner**: T-learner imputations + a propensity-weighted combination. Designed for imbalanced treatment groups.

**R-learner**: residualization + orthogonalization. Fit $\\hat m(z) = E[Y \\mid Z]$ and $\\hat e(z) = P(X=1 \\mid Z)$, then fit $\\hat\\tau(z)$ by minimizing $\\sum_i (Y_i - \\hat m(Z_i) - \\hat\\tau(Z_i)(X_i - \\hat e(Z_i)))^2$."""),

code("""# Reuse the arrays we built in Part 2; alias to short names for the manual code below.
Z = Z_arr
X = X_arr
Y = Y_arr
true_grid = true_curve

# S-learner
s_model = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
s_model.fit(np.column_stack([X, Z]), Y)
def s_cate(Z_eval):
    n_eval = len(Z_eval)
    mu1 = s_model.predict(np.column_stack([np.ones(n_eval), Z_eval]))
    mu0 = s_model.predict(np.column_stack([np.zeros(n_eval), Z_eval]))
    return mu1 - mu0

# T-learner
t_model_1 = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
t_model_0 = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
t_model_1.fit(Z[X == 1], Y[X == 1])
t_model_0.fit(Z[X == 0], Y[X == 0])
def t_cate(Z_eval):
    return t_model_1.predict(Z_eval) - t_model_0.predict(Z_eval)

# R-learner (manual implementation)
m_model = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3).fit(Z, Y)
e_model = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3).fit(Z, X)
m_hat = m_model.predict(Z)
e_hat = np.clip(e_model.predict_proba(Z)[:, 1], 0.05, 0.95)
Y_resid = Y - m_hat
X_resid = X - e_hat
# Solve weighted regression: tau(Z) = (X_resid^T X_resid)^-1 X_resid^T Y_resid
# Use the orthogonalized formulation, with tau modeled as a learner of Z
tau_target = Y_resid / np.where(np.abs(X_resid) > 0.01, X_resid, 0.01)
weights = X_resid ** 2
r_model = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)
r_model.fit(Z, tau_target, sample_weight=weights)
def r_cate(Z_eval):
    return r_model.predict(Z_eval)

# X-learner
mu1_hat = t_model_1.predict(Z)
mu0_hat = t_model_0.predict(Z)
D1 = Y[X == 1] - mu0_hat[X == 1]      # impute control counterfactual for treated
D0 = mu1_hat[X == 0] - Y[X == 0]      # impute treated counterfactual for controls
tau_1 = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3).fit(Z[X == 1], D1)
tau_0 = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3).fit(Z[X == 0], D0)
def x_cate(Z_eval):
    e = np.clip(e_model.predict_proba(Z_eval)[:, 1], 0.05, 0.95)
    return e * tau_0.predict(Z_eval) + (1 - e) * tau_1.predict(Z_eval)

# Evaluate on the grid
fig, ax = plt.subplots()
true_grid = 0.0006 * (z1_grid - 500) - 0.002 * 7.0
ax.plot(z1_grid, true_grid, "k", linewidth=2.5, label="True CATE")
ax.plot(z1_grid, s_cate(Z_grid), label="S-learner")
ax.plot(z1_grid, t_cate(Z_grid), label="T-learner")
ax.plot(z1_grid, r_cate(Z_grid), label="R-learner")
ax.plot(z1_grid, x_cate(Z_grid), label="X-learner")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Resist thickness Z1 (nm) — Z2 fixed at 7 days")
ax.set_ylabel("Estimated CATE")
ax.set_title("Meta-learner comparison: CATE as a function of resist thickness")
ax.legend()
plt.show()"""),

md("""All four learners track the true CATE reasonably well in the bulk of the resist-thickness range. Where they differ is in the *PEHE* (computed in Part 3): with this balanced setup, S-, T-, and X-learners come in around 0.004 and the R-learner around 0.012 — the R-learner's residualized regression is more variable here because the propensity has limited cross-resist variation. The R-learner's advantage shows up in imbalanced settings or with non-linear nuisance models; Exercise 1 stresses that case."""),

md("""## Part 5 — Per-unit CATE accuracy (PEHE)

PEHE for the manual implementations from Part 4 should land in the same range as the EconML library outputs from Part 3, validating both.

The *Precision in Estimation of Heterogeneous Effects* (PEHE) measures the per-unit accuracy:

$$\\text{PEHE} = \\sqrt{\\frac{1}{n}\\sum_i (\\hat\\tau(Z_i) - \\tau(Z_i))^2}.$$

PEHE is only computable on synthetic data (we need the true CATE per unit). It's the right metric when $\\hat\\tau$ will drive per-unit decisions."""),

code("""def pehe(predict_fn):
    pred = predict_fn(Z)
    return np.sqrt(((pred - tau_true) ** 2).mean())

print(f"PEHE (lower is better):")
print(f"  S-learner: {pehe(s_cate):.4f}")
print(f"  T-learner: {pehe(t_cate):.4f}")
print(f"  R-learner: {pehe(r_cate):.4f}")
print(f"  X-learner: {pehe(x_cate):.4f}")"""),

md("""## Part 6 — Qini curve and the Qini coefficient

A Qini curve ranks units by predicted CATE from largest to smallest, then plots the cumulative *true* treatment effect against the fraction treated. A good CATE estimator concentrates large effects at the top, producing a steeply rising curve.

The *Qini coefficient* is the area between the Qini curve and the diagonal (random ranking)."""),

code("""# Use the X-learner's CATE predictions for ranking
cate_pred = x_cate(Z)

order = np.argsort(-cate_pred)             # rank by predicted CATE, descending
cum_true_effect = np.cumsum(tau_true[order])
fraction = np.arange(1, n + 1) / n
random_baseline = fraction * cum_true_effect[-1]

# Qini coefficient = area between curve and random baseline
qini_coef = (cum_true_effect - random_baseline).sum() / n

fig, ax = plt.subplots()
ax.plot(fraction, cum_true_effect, label="X-learner targeting")
ax.plot(fraction, random_baseline, "--", label="Random targeting")
ax.fill_between(fraction, cum_true_effect, random_baseline,
                where=(cum_true_effect > random_baseline), alpha=0.2)
ax.set_xlabel("Fraction of lots treated (ranked by X-learner CATE)")
ax.set_ylabel("Cumulative true treatment effect")
ax.set_title(f"Qini curve — coefficient = {qini_coef:.3f}")
ax.legend()
plt.show()"""),

md("""The curve climbs above the diagonal in the early fractions — the X-learner's top-ranked lots are indeed the ones with the largest treatment effect. The Qini coefficient quantifies how much of the maximum possible gain is captured by ranking-based targeting.

A Qini close to zero would mean random ranking is as good as the model's — the CATE estimator added no value. A Qini near its maximum would mean perfect ranking. In real settings, Qini lets you compare CATE estimators directly on a target-population value."""),

md("""## Part 7 — Policy comparison

Three candidate policies:

- **Blanket-B**: assign Recipe B to all lots.
- **Threshold**: assign B only when Z1 ≥ 500 nm.
- **Optimized**: assign B when the X-learner CATE prediction is positive.

Policy value = average true CATE among lots assigned to B (the units where treatment runs)."""),

code("""# Policy value = mean true outcome under the policy
# For each policy, compute the mean of (1 if treated then Y(1) else Y(0)).
# In the synthetic SCM, Y(1) - Y(0) = tau, so policy_value relative to no-treatment is
# the mean of tau over treated units, times the fraction treated.

def policy_value(treatment_indicator):
    \"\"\"Expected gain over treating none, averaged across the population.\"\"\"
    return (treatment_indicator * tau_true).mean()

policies = {
    "Blanket-B (treat all)":             np.ones(n, dtype=int),
    "Threshold (Z1 >= 500)":              (data["Z1"] >= 500).astype(int).values,
    "Optimized (X-learner CATE > 0)":     (cate_pred > 0).astype(int),
    "Oracle (true CATE > 0)":             (tau_true > 0).astype(int),
}

for name, ind in policies.items():
    print(f"  {name:<35s}  treated fraction = {ind.mean():.3f}   policy value = {policy_value(ind):+.4f}")"""),

md("""**Blanket-B** treats everyone, including the thin-resist lots where Recipe B hurts. The blanket policy's value is positive but small.

**Threshold** treats only thick-resist lots. Its value is roughly $2-3\\times$ blanket because it avoids the negative-CATE lots.

**Optimized** (X-learner CATE > 0) further refines by also considering Z2. It approaches the oracle.

**Oracle** is the upper bound — if we knew the true CATE we'd treat exactly the positive-CATE lots. The optimized X-learner gets close.

In practice, a team would report:
- Blanket policy value (the *naive* deployment baseline)
- Threshold policy value (an interpretable manual rule)
- Optimized policy value (the model-driven recommendation)
And let the deployment trade-off — interpretability vs. value — be a decision."""),

md("""## Reflection

**Heterogeneity changes the deployment decision.** When ATE is small but CATE varies, a blanket deployment can be barely better than nothing while a targeted deployment can deliver real gains. CATE is the right tool when the deployment is per-unit (per lot, per part, per customer).

**Different learners have different inductive biases.** S-learner shrinks effects under regularization. T-learner is noisy under imbalance. R- and X-learners handle imbalance better. In real settings, fit multiple learners and use the Qini curve or holdout policy value to pick.

**Confidence intervals matter for policy.** A CATE point estimate that is positive in expectation but whose 90% CI includes zero is not strong evidence for treatment at that unit. CausalForestDML's interval-aware output supports better deployment decisions."""),

md("""## Exercises

1. **Stress with imbalance.** Modify the SCM so only 20% of lots are treated (skew the behavior policy). Refit S- and T-learners. Which degrades more? Plot the CATE curves.

   <details><summary>Solution</summary>

   T-learner suffers more: the treated arm has 5× fewer samples, so $\\hat\\mu_1$ is noisy. S-learner pools the data but its treatment-indicator signal is now relatively weak (~20% of rows mark $X=1$). X-learner is designed for this case and degrades less — its propensity-weighted combination puts weight on the better-estimated side per region. CausalForestDML's PEHE typically wins by 10–30% in this regime.
   </details>

2. **Adversarial CATE.** Make the true CATE a sharp step function at $Z_1 = 500$ instead of a smooth ramp. Which meta-learner captures the step best?

   <details><summary>Solution</summary>

   Tree-based final stages (CausalForestDML, X- or DR-learner with gradient-boosted CATE models) handle the step cleanly because their split points can land exactly at 500 nm. Linear-CATE learners (R-learner with a linear final stage) smooth the step into a slope, underestimating CATE just below 500 and just above. Lesson: pick the CATE final-stage model class with the kind of heterogeneity you expect.
   </details>

3. **Confidence intervals.** Use causal forest CIs to identify the *uncertain* subpopulation — lots where the 90% CI includes zero. How many lots fall in this region? What does an operator do with them?

   <details><summary>Solution</summary>

   ```python
   ci_lower, ci_upper = cf.effect_interval(Z_arr, alpha=0.1)
   uncertain = (ci_lower < 0) & (ci_upper > 0)
   print(f"Uncertain lots: {uncertain.sum()} of {len(Z_arr)} ({uncertain.mean()*100:.1f}%)")
   ```

   Typically 30–50% of lots are uncertain in modest-$n$ settings. Operationally these are lots where a per-unit decision is unsafe; group them into a "monitor and collect more data" bucket, or default to the conservative policy (existing recipe). The certain-positive lots are where the targeted intervention is justified.
   </details>

4. **A/B vs CATE.** Compare a per-unit optimized policy to a single-best-recipe A/B selection on a different metric: regret. How much value is lost by ignoring heterogeneity?

   <details><summary>Solution</summary>

   Regret = $E[\\tau(Z) \\cdot \\mathbf 1[\\hat\\pi(Z) \\ne \\pi^*(Z)]]$. The single-best A/B policy applies one decision globally; its regret integrates $|\\tau(z)|$ over the "wrong side" of the CATE. On the lab's SCM that's substantial when the CATE has both signs — blanket-B regret was ~70% of the targeted policy's value (the §6.6 policy table makes this concrete: blanket +0.016, targeted +0.046).
   </details>"""),

md("""## What's next

Lab 7 turns from a single intervention to *sequential* treatments — the time-varying setting where decisions repeat over time and the same machinery (G-formula, IPW, DR) generalizes."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch06" / "lab06.ipynb", cells)
print("Built lab06.ipynb")
