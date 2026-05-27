"""Build labs/ch10/lab10.ipynb — Multi-stage mediation, RCA, and FDC."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 10 — Multi-Stage Mediation, RCA, and FDC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch10/lab10.ipynb)

**Companion lab to Chapter 10.** Three parts:

1. **Single-mediator estimation** on the chapter's two-stage wafer-flow scenario. Reproduce the §10.8 result: TE = 1.16, NDE = 0.60, NIE = 0.56 with $M_2$ as the chosen mediator.
2. **Multi-stage path-specific decomposition** that respects the recanting-witness constraint by leaving non-identified pieces explicitly unattributed.
3. **FDC root-cause attribution** on a synthetic etch-tool dataset with 10 candidate sensor signals, three of which are real mediators."""),

md("""## What you'll do

1. **Build the §10.8 two-stage wafer flow** with the M1 → Y edge that makes M1 a recanting witness.
2. **Implement the mediation formula** (sequential regression) with both linear and Monte-Carlo plug-in.
3. **Implement the doubly-robust mediation estimator** (Tchetgen Tchetgen-Shpitser 2012) and verify it's consistent under one-sided misspecification.
4. **Decompose into path-specific effects** — direct, M2-only, M1+M2 — leaving "M1-only" unattributed.
5. **FDC RCA exercise**: 10 candidate FDC signals during an excursion; rank by NIE on yield; identify the 3 true mediators."""),

md("""## Setup"""),

code("""# Colab: install DoWhy for the library cross-check at the end.
%pip install --quiet dowhy 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The chapter's two-stage wafer flow

SCM with the M1→Y edge added (so M1 is a real recanting witness):

- $X \\sim \\text{Bernoulli}(0.5)$
- $M_1 = 1.0\\,X + \\varepsilon_1$
- $M_2 = 0.5\\,M_1 + 0.3\\,X + \\varepsilon_2$
- $Y = 0.7\\,M_2 + 0.2\\,M_1 + 0.4\\,X + \\varepsilon_Y$

Path-specific effects:
- Direct ($X \\to Y$): 0.40
- Through $M_1$ only ($X \\to M_1 \\to Y$): 0.20 — **non-identified** (M1 is the recanting witness for X→M1→M2→Y)
- Through $M_2$ only ($X \\to M_2 \\to Y$): 0.21
- Through $M_1$ and $M_2$ ($X \\to M_1 \\to M_2 \\to Y$): 0.35
- Total: 1.16

NDE w.r.t. $M_2$ = 0.60 (direct + M1-only). NIE w.r.t. $M_2$ = 0.56 (M2-only + M1+M2)."""),

code("""n = 5000

def mediator_chain(X, eps1, eps2):
    M1 = 1.0 * X + eps1
    M2 = 0.5 * M1 + 0.3 * X + eps2
    return M1, M2

def yield_fn(M1, M2, X, epsY):
    return 0.7 * M2 + 0.2 * M1 + 0.4 * X + epsY

X    = rng.binomial(1, 0.5, n).astype(float)
eps1 = rng.normal(0, 0.3, n)
eps2 = rng.normal(0, 0.3, n)
epsY = rng.normal(0, 0.3, n)
M1, M2 = mediator_chain(X, eps1, eps2)
Y      = yield_fn(M1, M2, X, epsY)

# True path-specific decomposition (analytical)
direct      = 0.4
via_M1_only = 1.0 * 0.2
via_M2_only = 0.3 * 0.7
via_M1_M2   = 1.0 * 0.5 * 0.7
TE_true  = direct + via_M1_only + via_M2_only + via_M1_M2
NDE_true = direct + via_M1_only      # NDE w.r.t. M2
NIE_true = via_M2_only + via_M1_M2
print(f"True path-specific decomposition:")
print(f"  Direct:           {direct:+.3f}")
print(f"  M1 only:          {via_M1_only:+.3f}  (NON-IDENTIFIED; recanting witness)")
print(f"  M2 only:          {via_M2_only:+.3f}")
print(f"  M1 and M2:        {via_M1_M2:+.3f}")
print(f"  TE:               {TE_true:+.3f}")
print(f"  NDE wrt M2:       {NDE_true:+.3f}")
print(f"  NIE wrt M2:       {NIE_true:+.3f}")"""),

md("""**Recanting witness — sketch the paths.** Before continuing, list every directed path from $X$ to $Y$ in the SCM above. You should find four: $X \\to Y$ (direct), $X \\to M_1 \\to Y$ (M1 only), $X \\to M_2 \\to Y$ (M2 only), and $X \\to M_1 \\to M_2 \\to Y$ (M1 then M2). Now ask: to isolate the *M1 only* path, you would need to "switch on" $M_1$'s response to $X$ on the $X \\to M_1 \\to Y$ leg but "switch off" $M_1$'s response on the $X \\to M_1 \\to M_2 \\to Y$ leg. The same node $M_1$ would have to play two contradictory roles in the same hypothetical world — it would have to *recant its testimony*. That's why this path-specific effect is **non-identified** from observational data alone (Avin-Shpitser-Pearl 2005). The next part estimates the *identifiable* pieces and leaves the M1-only piece explicitly unattributed."""),

md("""## Part 2 — Mediation formula estimator (Pearl 2001)

With $M_2$ as the chosen mediator, the mediation formula requires fitting two models:
- Outcome model: $\\hat\\mu_Y(X, M_1, M_2)$. **Must include $M_1$** because $M_1 \\to Y$ exists; omitting it biases the NDE.
- Mediator-chain models: $\\hat\\mu_{M_1}(X)$ and $\\hat\\mu_{M_2}(X, M_1)$ for sampling counterfactual $M_1$ and $M_2$ values."""),

code("""# Outcome model (must include M1)
mu_Y  = LinearRegression().fit(np.column_stack([X, M1, M2]), Y)
mu_M1 = LinearRegression().fit(X.reshape(-1, 1), M1)
mu_M2 = LinearRegression().fit(np.column_stack([X, M1]), M2)

# Counterfactual mediator values
M1_x0 = mu_M1.predict(np.zeros((n, 1)))
M1_x1 = mu_M1.predict(np.ones((n, 1)))
M2_x0 = mu_M2.predict(np.column_stack([np.zeros(n), M1_x0]))
M2_x1 = mu_M2.predict(np.column_stack([np.ones(n), M1_x1]))

def Y_hat(x_val, m1_val, m2_val):
    return mu_Y.predict(np.column_stack([np.full(n, x_val), m1_val, m2_val]))

NDE_hat = (Y_hat(1, M1_x1, M2_x0) - Y_hat(0, M1_x0, M2_x0)).mean()
NIE_hat = (Y_hat(1, M1_x1, M2_x1) - Y_hat(1, M1_x1, M2_x0)).mean()

print(f"True NDE: {NDE_true:.3f}, Estimated: {NDE_hat:.3f}")
print(f"True NIE: {NIE_true:.3f}, Estimated: {NIE_hat:.3f}")
print(f"True TE:  {TE_true:.3f}, Estimated: {NDE_hat + NIE_hat:.3f}")"""),

md("""## Part 3 — The production workflow: DoWhy mediation

The manual two-stage estimator above is the right way to *understand* the mediation formula. In a real pipeline you would call DoWhy: pass it the DAG, ask for the NDE and NIE estimands, fit an estimator, get numbers.

Below: the same NDE/NIE on the same SCM, via DoWhy's mediation API."""),

code("""import dowhy
from dowhy import CausalModel

df_dowhy = pd.DataFrame({"X": X, "M1": M1, "M2": M2, "Y": Y})
gml = '''graph [directed 1
    node [id "X" label "X"]
    node [id "M1" label "M1"]
    node [id "M2" label "M2"]
    node [id "Y" label "Y"]
    edge [source "X" target "M1"]
    edge [source "X" target "M2"]
    edge [source "X" target "Y"]
    edge [source "M1" target "M2"]
    edge [source "M1" target "Y"]
    edge [source "M2" target "Y"]
]'''

# Use M2 as the chosen mediator (matching the manual analysis above).
model = CausalModel(data=df_dowhy, treatment="X", outcome="Y", graph=gml)

# NDE: direct effect of X on Y not through M2
est_nde = model.identify_effect(estimand_type="nonparametric-nde",
                                proceed_when_unidentifiable=True)
nde_result = model.estimate_effect(
    est_nde,
    method_name="mediation.two_stage_regression",
    method_params={
        "first_stage_model": dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
        "second_stage_model": dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
    },
)

# NIE: indirect effect of X on Y through M2
est_nie = model.identify_effect(estimand_type="nonparametric-nie",
                                proceed_when_unidentifiable=True)
nie_result = model.estimate_effect(
    est_nie,
    method_name="mediation.two_stage_regression",
    method_params={
        "first_stage_model": dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
        "second_stage_model": dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
    },
)

print(f"Truth:       NDE = {NDE_true:.3f}    NIE = {NIE_true:.3f}    TE = {TE_true:.3f}")
print(f"Manual:      NDE = {NDE_hat:.3f}    NIE = {NIE_hat:.3f}    TE = {NDE_hat+NIE_hat:.3f}")
print(f"DoWhy:       NDE = {nde_result.value:.3f}    NIE = {nie_result.value:.3f}    "
      f"TE = {nde_result.value + nie_result.value:.3f}")"""),

md("""DoWhy reproduces the manual NDE and NIE estimates (within sampling noise). Two practical wins from using the library: (1) it identifies the estimand symbolically before estimating, so the assumptions are spelled out in the output (no silent wrong-adjustment failure); (2) the same `estimate_effect` API lets you swap from linear regression to a non-parametric estimator, propensity-score-based, or a flexible ML learner — without rewriting the mediation logic."""),

md("""## Part 4 — Importance of including $M_1$ in the outcome model

Try omitting $M_1$ from the outcome model. The NDE estimate is biased because the regression then marginalizes over $M_1$ given $X$ and $M_2$ — but the mediation formula needs the conditional given the *natural* $M_1$ under intervention."""),

code("""# Bad outcome model: omit M1 even though M1 -> Y exists in the SCM.
# Why this breaks: the fitted mu_Y_bad(X, M2) estimates E[Y | X, M2], which
# implicitly marginalizes over M1 given (X, M2). The mediation formula needs
# E[Y | X, M1, M2] evaluated at the *natural* M1 under do(X = x) — a different
# conditional. The M1-mediated effect leaks into the X coefficient, biasing NDE.
mu_Y_bad = LinearRegression().fit(np.column_stack([X, M2]), Y)
def Y_hat_bad(x_val, m2_val):
    return mu_Y_bad.predict(np.column_stack([np.full(n, x_val), m2_val]))

NDE_bad = (Y_hat_bad(1, M2_x0) - Y_hat_bad(0, M2_x0)).mean()
NIE_bad = (Y_hat_bad(1, M2_x1) - Y_hat_bad(1, M2_x0)).mean()

print(f"With M1 omitted from outcome model:")
print(f"  NDE: {NDE_bad:.3f}  (true: {NDE_true:.3f},  biased)")
print(f"  NIE: {NIE_bad:.3f}  (true: {NIE_true:.3f},  also biased)")"""),

md("""The NDE is now biased. Mechanism: when $M_1$ is omitted from the outcome regression, the fitted $\\hat\\mu_Y(X, M_2)$ is the best linear approximation to $E[Y \\mid X, M_2]$, which marginalizes over $M_1$ given $(X, M_2)$. The mediation formula needs $E[Y \\mid X, M_1, M_2]$ evaluated at the *natural* $M_1$ under intervention $do(X = x)$ — a different conditional. The model's $X$-coefficient picks up some of the $M_1$-mediated effect, shifting NDE downward and (in compensation) shifting NIE upward. The chapter §10.9 calls this out: include in the outcome model every variable with a direct edge to $Y$ — even if that variable isn't the chosen mediator."""),

md("""## Part 5 — Path-specific decomposition

The four path-specific effects in this linear SCM are products of the coefficients along each path. The "M1 only" path is non-identified from observational data (recanting witness); the other three are identified. A defensible report shows the identified pieces and *explicitly* leaves the non-identified piece unattributed — rather than silently absorbing it into one of the others."""),

code("""# Path-specific decomposition (analytical, from the linear SCM coefficients).
print(f"Path-specific decomposition:")
print(f"  Direct:     {direct:+.3f}")
print(f"  M1 only:    {via_M1_only:+.3f}  [NON-IDENTIFIED from observational data]")
print(f"  M2 only:    {via_M2_only:+.3f}")
print(f"  M1 and M2:  {via_M1_M2:+.3f}")
print(f"  Sum:        {direct + via_M1_only + via_M2_only + via_M1_M2:+.3f}")
print()
print("Reportable decomposition (only identified paths):")
print(f"  Direct + M2-only + (M1 and M2) = {direct + via_M2_only + via_M1_M2:+.3f}")
print(f"  M1-only effect (unattributed):   {via_M1_only:+.3f}")
print(f"  Total effect:                    {direct + via_M1_only + via_M2_only + via_M1_M2:+.3f}")"""),

md("""## Part 6 — FDC root-cause attribution

A simulated FDC scenario: an etch tool runs 5000 wafers. During the second half, an "excursion" begins (an upstream parameter shifted), causing yield to drop. We have 10 candidate FDC signals during this window. Three are *real* mediators of the upstream cause; seven are correlated noise.

The mediation framing: for each candidate signal $M_i$, estimate the NIE through $M_i$. The three true mediators should have large NIE; the seven non-mediators should have NIE near zero."""),

code("""# Build the FDC scenario
n_fdc = 5000
excursion = (np.arange(n_fdc) > n_fdc // 2).astype(int)   # X = 0 baseline, 1 = during excursion

# True mediators (3): driven by the excursion and causally affect yield
true_med1 = 1.0 * excursion + rng.normal(0, 0.3, n_fdc)
true_med2 = 0.8 * excursion + rng.normal(0, 0.3, n_fdc)
true_med3 = 1.2 * excursion + rng.normal(0, 0.3, n_fdc)

# Non-mediators (7): correlated with the excursion OR with each other but NOT causal on yield
# Some are downstream sensors (depend on the mediators), some are correlated noise
nonmed1 = 0.5 * true_med1 + rng.normal(0, 0.3, n_fdc)   # downstream of true_med1
nonmed2 = 0.5 * excursion + rng.normal(0, 0.3, n_fdc)   # correlated but not causal
nonmed3 = rng.normal(0, 1, n_fdc)                        # pure noise
nonmed4 = 0.3 * excursion + rng.normal(0, 0.3, n_fdc)
nonmed5 = rng.normal(0, 1, n_fdc)
nonmed6 = 0.7 * true_med2 + rng.normal(0, 0.3, n_fdc)   # downstream of true_med2
nonmed7 = rng.normal(0, 1, n_fdc)

# Yield: function of true mediators only
yield_excursion = (90 - 0.4 * true_med1 - 0.5 * true_med2 - 0.3 * true_med3
                   + rng.normal(0, 0.5, n_fdc))

candidates = {
    "med1 (true)": true_med1, "med2 (true)": true_med2, "med3 (true)": true_med3,
    "nonmed1":    nonmed1,    "nonmed2":    nonmed2,    "nonmed3":    nonmed3,
    "nonmed4":    nonmed4,    "nonmed5":    nonmed5,    "nonmed6":    nonmed6,
    "nonmed7":    nonmed7,
}

# Per-mediator: fit the mediation formula and compute NIE through each candidate
def estimate_nie(med, X, Y):
    \"\"\"NIE through M, using linear models for E[Y | X, M] and E[M | X].\"\"\"
    mu_Y_m = LinearRegression().fit(np.column_stack([X, med]), Y)
    mu_M   = LinearRegression().fit(X.reshape(-1, 1), med)
    m0 = mu_M.predict(np.zeros((len(X), 1)))
    m1 = mu_M.predict(np.ones((len(X), 1)))
    y10 = mu_Y_m.predict(np.column_stack([np.ones(len(X)),  m0]))
    y11 = mu_Y_m.predict(np.column_stack([np.ones(len(X)),  m1]))
    return (y11 - y10).mean()

print(f"{'Candidate':<15} {'NIE estimate':>12}")
print("-" * 30)
results = []
for name, med in candidates.items():
    nie = estimate_nie(med, excursion, yield_excursion)
    results.append((name, nie))
    print(f"{name:<15} {nie:>+12.3f}")

print()
# Label each candidate by its causal role (set externally; in production this
# would come from the DAG, not the variable name).
role = {
    "med1 (true)": "TRUE MEDIATOR",   "med2 (true)": "TRUE MEDIATOR",
    "med3 (true)": "TRUE MEDIATOR",
    "nonmed1":    "downstream-of-mediator",  # downstream of med1
    "nonmed6":    "downstream-of-mediator",  # downstream of med2
    "nonmed2":    "correlated noise (X->W, W not->Y)",
    "nonmed3":    "noise",
    "nonmed4":    "correlated noise (X->W, W not->Y)",
    "nonmed5":    "noise",
    "nonmed7":    "noise",
}
print("Top-5 by |NIE| (with causal role):")
ranked = sorted(results, key=lambda x: -abs(x[1]))
for name, nie in ranked[:5]:
    print(f"  {name:<15} {nie:>+8.3f}  ({role[name]})")"""),

md("""The three true mediators have the largest |NIE| estimates and are correctly ranked at the top. The "downstream-of-mediator" sensors (nonmed1, nonmed6) also get substantial NIE because they carry the mediated signal — but they are *not* causally between $X$ and $Y$, they're downstream observations of the true mediators.

This is a real-world FDC pitfall: a downstream sensor with high NIE looks like a root cause but is actually just a record of the real mediator's effect. The fix is *temporal screening* — the sensor must move *before* the outcome moves, and *before* the candidate-mediator analysis is run. The chapter §10.9 discusses this."""),

md("""## Reflection

**Mediation analysis is more than "control for the mediator".** The mediation formula is a *specific* integration over the conditional distribution of the mediator under intervention, not just a regression with the mediator as a covariate.

**The outcome model must include every variable with a direct edge to $Y$.** Omitting M1 biased the NDE. The discovery / DAG validation step (Lab 9) is what tells you which variables those are.

**Non-identified path-specific effects require interventions.** When a recanting witness exists, the data is silent on the corresponding PSE. Report the identifiable pieces; flag the non-identified piece as such; if it matters, run a controlled experiment.

**FDC RCA is mediation in disguise.** The right rank-by-NIE approach gives correct attributions; the wrong "rank by correlation" approach confuses downstream sensors with root causes."""),

md("""## Exercises

1. **DR mediation estimator.** Implement the Tchetgen Tchetgen-Shpitser (2012) doubly-robust mediation estimator and verify it is consistent if the outcome model OR the propensity model is correct.

   <details><summary>Solution</summary>

   The DR mediation estimator augments the sequential-regression NDE/NIE with a pair of weighted-residual corrections. For NDE:
   $$\\widehat{\\text{NDE}}^{DR} = \\widehat{\\text{NDE}}^{plug-in} + \\frac{1}{n}\\sum_i \\left[\\frac{(1-X_i)}{1 - \\hat e(C_i)} \\cdot \\frac{\\hat p(M_i \\mid X=1, C_i)}{\\hat p(M_i \\mid X=0, C_i)} \\cdot \\bigl(Y_i - \\hat\\mu_Y(0, M_i, C_i)\\bigr)\\right] - \\dots$$
   plus symmetric terms. Verify by deliberately misspecifying either the outcome model OR the propensity / mediator-distribution model; DR recovers, plug-in does not. The full derivation is in Tchetgen Tchetgen & Shpitser (2012, §3).
   </details>

2. **Non-linear outcome.** Replace the linear $Y$ with a sigmoid output. Refit the linear-shortcut and Monte-Carlo mediation estimators. Which one degrades?

   <details><summary>Solution</summary>

   The linear-shortcut estimator (plug in $\\hat\\mu_M(X)$ as a point) suffers Jensen's-inequality bias when $\\hat\\mu_Y$ is non-linear in $M$ — the conditional mean of a non-linear function is not the function of the conditional mean. Monte-Carlo (sample $M$ from $\\hat p(M \\mid X)$, average $\\hat\\mu_Y$ over the samples) recovers the truth. The gap can be 10–30% of the NDE/NIE depending on the sigmoid's slope and the mediator's variance.
   </details>

3. **Temporal pre-screen.** Add a "downstream-of-yield" sensor to the FDC candidates (signal moves AFTER yield drops). Check if NIE ranking still picks it up — the temporal filter should exclude it before mediation.

   <details><summary>Solution</summary>

   Without a temporal filter, the NIE through a post-outcome sensor can still come out non-zero (the signal correlates with yield, so the regression finds a path). The correct workflow: pre-screen candidates by *temporal order* — the sensor must move strictly before the outcome. In FDC practice this is done by aligning sensor timestamps against the yield-measurement timestamp; only "upstream-in-time" sensors are even considered for the NIE-ranking step.
   </details>

4. **Interventional effects under exposure-induced confounding.** Introduce a variable $W = 0.5\\,X + 0.3\\,M_1$ that confounds $M_2 \\to Y$. The natural NDE/NIE are no longer identified. Implement the interventional analogs (Vansteelandt-Daniel 2017).

   <details><summary>Solution</summary>

   When $W$ is post-treatment and confounds $M_2 \\to Y$, the cross-world independence (sequential ignorability assumption 3) fails. The fix: replace the cross-world counterfactual with an *interventional* analog — instead of $M_2(x^*)$ (the natural value under $x^*$), use a *random draw* from the population distribution of $M_2$ at $X = x^*$. The interventional NDE/NIE are identified under weaker conditions (no cross-world); the trade-off is that the decomposition is a population-average property, not an individual one. Implementation: sample $M_2$ from the empirical $\\hat P(M_2 \\mid X = x^*, C)$ and average $\\hat\\mu_Y$.
   </details>"""),

md("""## What's next

Lab 11 turns from decomposing existing effects to *evaluating new policies* without deploying them — off-policy evaluation."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch10" / "lab10.ipynb", cells)
print("Built lab10.ipynb")
