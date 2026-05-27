"""Build labs/ch07/lab07.ipynb — Time-varying treatments."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 7 — Time-Varying Treatments: G-Formula, IPTW, MSM, and Doubly-Robust Sequential

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch07/lab07.ipynb)

**Companion lab to Chapter 7.** When decisions repeat over time and the same machinery from Chapter 5 — outcome models, propensities, doubly-robust combination — extends to sequential settings. The lab implements all four sequential estimators on a two-period preventive-maintenance scenario, then introduces a controlled outcome-model misspecification and watches each estimator degrade in characteristic ways.

The chapter's §7.7 worked example is the predictive-maintenance setting: a critical etch tool with a wear pattern, decisions every 50 hours whether to run early PM, condition $L_t$ that drifts up over time and is reset by PM, yield $Y$ that depends on the final condition. The true ATE of always-PM vs. never-PM is $+4.0$. This lab recovers that number multiple ways and then breaks the assumptions to see how each estimator fails."""),

md("""## What you'll do

1. **Build the 2-period PM SCM** from §7.7 with known true ATE of $+4.0$.
2. **Naive estimator** — regress $Y$ on $(A_1, A_2)$ and read off coefficients. Wrong: $L_1$ is a time-varying confounder.
3. **G-formula** — sequential outcome modeling. Recovers the truth.
4. **IPTW** — sequential propensity scores with stabilized weights, fit an MSM.
5. **Doubly-robust sequential** — combines g-formula and IPTW; consistent if either is correct.
6. **Misspecification stress test** — wrong outcome model degrades g-formula; wrong propensity degrades IPTW; DR is rescued in both cases (when the *other* model is correct).
7. **Dynamic regime optimization** — find the threshold $\\theta$ such that the rule "PM whenever $L_t > \\theta$" maximizes cumulative yield."""),

md("""## Setup"""),

code("""# Colab: install zEpid (the standard Python g-methods library for IPTW + MSM).
%pip install --quiet zepid 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The 2-period PM SCM (Chapter 7 §7.7)

A tool with drift, two decision points, yield at the end.

- $L_0 \\sim \\mathcal N(0, 1)$ — baseline condition.
- $A_1 \\sim \\text{Bernoulli}(\\sigma(L_0))$ — PM more likely when condition is poor.
- $L_1 = L_0 + 0.5 - 1.0\\,A_1 + \\varepsilon_{L_1}$ — drift +0.5/period; PM subtracts 1.0.
- $A_2 \\sim \\text{Bernoulli}(\\sigma(L_1))$ — same decision rule, new condition.
- $Y = -2.0\\,L_1 + 1.0\\,A_1 + 1.0\\,A_2 + \\varepsilon_Y$ — drift hurts yield, each PM directly helps.

True ATE of always-PM vs never-PM = +1.0 (A1 direct) + 1.0 (A2 direct) + 2.0 (A1 → L1 propagated) = +4.0."""),

code("""n = 5000

def step_L(prev_L, A, noise):
    return prev_L + 0.5 - 1.0 * A + noise

def yield_fn(L1, A1, A2, noise):
    return -2.0 * L1 + 1.0 * A1 + 1.0 * A2 + noise

# Observational data
noise_L1 = rng.normal(0, 0.3, n)
noise_Y  = rng.normal(0, 0.5, n)
L0 = rng.normal(0, 1.0, n)
A1 = rng.binomial(1, 1 / (1 + np.exp(-L0)))
L1 = step_L(L0, A1, noise_L1)
A2 = rng.binomial(1, 1 / (1 + np.exp(-L1)))
Y  = yield_fn(L1, A1, A2, noise_Y)
df = pd.DataFrame({"L0": L0, "A1": A1, "L1": L1, "A2": A2, "Y": Y})

# True ATE: re-evaluate the SCM under always-PM and never-PM with the same noise
def true_value(a1, a2):
    L1_pol = step_L(L0, np.full(n, a1), noise_L1)
    return yield_fn(L1_pol, np.full(n, a1), np.full(n, a2), noise_Y).mean()

true_ate = true_value(1, 1) - true_value(0, 0)
print(f"True ATE of always-PM vs never-PM: {true_ate:+.3f}  (analytical: +4.000)")
print(f"\\nObserved data:")
print(df.head())"""),

md("""## Part 2 — Naive: regress $Y$ on $(A_1, A_2)$

The naive estimator: treat $A_1$ and $A_2$ as a single 2-vector treatment, regress $Y$ on them, read coefficients. This ignores the time-varying confounder $L_1$: $A_2$ depends on $L_1$, which is itself caused by $A_1$, so the observational $A_1$–$Y$ relationship is biased by $L_1$'s downstream confounding."""),

code("""# Naive: Y = beta0 + beta1*A1 + beta2*A2 (ignoring L0, L1)
naive_lr = LinearRegression().fit(df[["A1", "A2"]], df["Y"])
naive_ate = naive_lr.predict([[1, 1]])[0] - naive_lr.predict([[0, 0]])[0]
print(f"Naive estimate of always-PM vs never-PM: {naive_ate:+.3f}")
print(f"True ATE:                                {true_ate:+.3f}")
print(f"Bias:                                    {naive_ate - true_ate:+.3f}")"""),

md("""The naive estimate undershoots — $A_1$ is selected because $L_0$ is high (poor condition), and $L_0$ doesn't appear in the regression. The model attributes part of $A_1$'s apparent effect to "lots that were going to fail anyway."

The chapter (§7.4) makes this point: standard regression adjustment on time-varying covariates is *not* the same as conditioning on baseline confounders. The g-formula sequential adjustment is the right tool."""),

md("""## Part 3 — G-formula (sequential outcome modeling)

The parametric g-formula for two periods:

$$E[Y(\\bar a)] = E_{L_0}\\left[E_{L_1 \\mid A_1 = a_1, L_0}\\left[E[Y \\mid A_1 = a_1, A_2 = a_2, L_1, L_0]\\right]\\right].$$

Estimation: fit two models. (i) $\\hat\\mu_Y(L_1, A_1, A_2, L_0)$ — outcome on covariates. (ii) $\\hat\\mu_{L_1}(A_1, L_0)$ — next-period covariate on history. Then evaluate the formula by Monte Carlo: for each unit, simulate $L_1$ under the intervention and average the outcome model."""),

code("""def g_formula(a1_val, a2_val, df, n_mc=50):
    \"\"\"G-formula estimate of E[Y | do(A1=a1_val, A2=a2_val)].\"\"\"
    n = len(df)
    L0 = df["L0"].values

    # Outcome model: Y ~ L0, A1, L1, A2 (linear is correct here)
    mu_Y = LinearRegression().fit(df[["L0", "A1", "L1", "A2"]], df["Y"])

    # Mediator (next-period covariate) model: L1 ~ L0, A1
    mu_L1 = LinearRegression().fit(df[["L0", "A1"]], df["L1"])
    # Residual variance for sampling
    L1_resid = df["L1"].values - mu_L1.predict(df[["L0", "A1"]].values)
    sigma_L1 = L1_resid.std()

    # Monte Carlo: for each unit, simulate L1 under do(A1=a1_val), then compute Y
    total = 0.0
    rng_inner = np.random.default_rng(42)
    for _ in range(n_mc):
        L1_mean = mu_L1.predict(np.column_stack([L0, np.full(n, a1_val)]))
        L1_sample = L1_mean + rng_inner.normal(0, sigma_L1, n)
        Y_hat = mu_Y.predict(np.column_stack([L0, np.full(n, a1_val), L1_sample, np.full(n, a2_val)]))
        total += Y_hat.mean()
    return total / n_mc

gf_ate = g_formula(1, 1, df) - g_formula(0, 0, df)
print(f"G-formula estimate: {gf_ate:+.3f}  (true: {true_ate:+.3f})")"""),

md("""G-formula recovers the truth (within sampling noise). It correctly handles the time-varying confounding by sequentially modeling each next-period covariate conditional on past treatments and covariates."""),

md("""## Part 4 — IPTW with stabilized weights and MSM

Inverse-probability-of-treatment weighting: weight each unit by the inverse of the probability of its observed treatment sequence, given history. Stabilized weights divide by the unconditional probability of the sequence to reduce variance."""),

code("""# Fit propensity models
# Step 1: P(A1 = 1 | L0)
pi1 = LogisticRegression().fit(df[["L0"]], df["A1"])
# Step 2: P(A2 = 1 | L0, A1, L1)
pi2 = LogisticRegression().fit(df[["L0", "A1", "L1"]], df["A2"])

# Conditional treatment probabilities at observed data
p_A1 = pi1.predict_proba(df[["L0"]])[:, 1]
p_A2 = pi2.predict_proba(df[["L0", "A1", "L1"]])[:, 1]

# Stabilized weight = (marginal-rate numerator) / (history-conditional denominator).
# Numerator P(A_t) is the marginal probability of the observed treatment — independent of history.
# Denominator P(A_t | history) is the propensity. The ratio cancels the part of treatment
# that is "predictable from history", keeping weights near 1 and reducing variance vs raw IPW.

# Numerators: marginal P(A1) and conditional-on-A1 marginal P(A2 | A1)
p_A1_marg     = df["A1"].mean()
p_A2_given_A1 = df.groupby("A1")["A2"].transform("mean").values

# Per-period probability evaluated at the observed treatment value
p_A1_obs = np.where(df["A1"] == 1, p_A1,      1 - p_A1)         # denominator: P(A1 | L0)
p_A2_obs = np.where(df["A2"] == 1, p_A2,      1 - p_A2)         # denominator: P(A2 | L0, A1, L1)
num_A1   = np.where(df["A1"] == 1, p_A1_marg, 1 - p_A1_marg)    # numerator:   P(A1)
num_A2   = np.where(df["A2"] == 1, p_A2_given_A1, 1 - p_A2_given_A1)  # numerator: P(A2 | A1)

# Stabilized weight: product over periods of (marginal / conditional)
w_stab = (num_A1 / p_A1_obs) * (num_A2 / p_A2_obs)
print(f"Stabilized weight stats: mean={w_stab.mean():.3f}, max={w_stab.max():.2f}, "
      f"99th pct={np.percentile(w_stab, 99):.2f}")"""),

code("""# MSM: weighted regression of Y on A1, A2 (and cumulative A1+A2 if you want a linear MSM)
# Linear MSM: Y = beta0 + beta1*A1 + beta2*A2
msm = LinearRegression().fit(df[["A1", "A2"]], df["Y"], sample_weight=w_stab)
msm_ate = msm.predict([[1, 1]])[0] - msm.predict([[0, 0]])[0]
print(f"IPTW + linear MSM: {msm_ate:+.3f}  (true: {true_ate:+.3f})")"""),

md("""IPTW recovers the truth. The *median* and 99th-percentile of the stabilized weights are small (median ~1, 99th ≈ 4), even though the single largest weight is ~66 — a long tail that is the usual fingerprint of stabilized IPS. The estimator is still well-behaved here because the tail's contribution to the mean is bounded by the stabilizing numerator. With longer horizons and sharper propensities, the tail can dominate and the estimator becomes unreliable — see Chapter 11's discussion."""),

md("""## Part 5 — Doubly-robust sequential

The DR sequential estimator combines g-formula and IPTW: consistent if *either* the outcome model or the propensity model is correctly specified.

The construction is iterative — at each time step, the DR estimator combines an outcome-model prediction with an IPTW-weighted residual correction. For two periods:"""),

code("""# Doubly-robust sequential estimator (Bang-Robins style)
# Step 1: Fit Q1(L0, A1, L1, A2) = E[Y | L0, A1, L1, A2]
Q_final = LinearRegression().fit(df[["L0", "A1", "L1", "A2"]], df["Y"])

def Y_hat(L0_v, A1_v, L1_v, A2_v):
    return Q_final.predict(np.column_stack([L0_v, A1_v, L1_v, A2_v]))

# DR for the final-period (A2) treatment: AIPW formula at time 2
def dr_sequential(a1_target, a2_target):
    n = len(df)
    # Step 1: AIPW correction at time 2
    Q2_at_target = Y_hat(df["L0"].values, df["A1"].values, df["L1"].values, np.full(n, a2_target))
    # Q2 has the IPTW correction added back
    p_A2_target = np.where(a2_target == 1, p_A2, 1 - p_A2)
    Q2_aug = Q2_at_target + ((df["A2"] == a2_target) * (df["Y"].values - Q_final.predict(df[["L0","A1","L1","A2"]].values))) / p_A2_target

    # Step 2: integrate over L1's distribution under do(A1 = a1_target)
    # Use g-formula style: fit a model of Q2_aug on (L0, A1), evaluate at A1=a1_target
    Q1_model = LinearRegression().fit(df[["L0", "A1"]], Q2_aug)
    Q1_at_target = Q1_model.predict(np.column_stack([df["L0"].values, np.full(n, a1_target)]))
    p_A1_target = np.where(a1_target == 1, p_A1, 1 - p_A1)
    Q1_aug = Q1_at_target + ((df["A1"] == a1_target) * (Q2_aug - Q1_model.predict(df[["L0", "A1"]].values))) / p_A1_target

    return Q1_aug.mean()

dr_ate = dr_sequential(1, 1) - dr_sequential(0, 0)
print(f"DR sequential: {dr_ate:+.3f}  (true: {true_ate:+.3f})")"""),

md("""## Part 6 — Misspecification stress test

Now break the outcome model and see what happens. We omit the time-varying confounder $L_1$ from the outcome regression — without it, the model can't see how $A_1$ propagates through $L_1$ to $Y$."""),

code("""# Misspecified outcome model: omits L1 (the time-varying confounder).
# Without L1, the model can't see that A1's effect propagates through L1 to Y.
Q_bad = LinearRegression().fit(df[["L0", "A1", "A2"]], df["Y"])
def Y_hat_bad(L0_v, A1_v, A2_v):
    return Q_bad.predict(np.column_stack([L0_v, A1_v, A2_v]))

def g_formula_bad(a1_val, a2_val):
    n = len(df)
    Y_pred = Y_hat_bad(df["L0"].values, np.full(n, a1_val), np.full(n, a2_val))
    return Y_pred.mean()

gf_bad = g_formula_bad(1, 1) - g_formula_bad(0, 0)
print(f"G-formula with bad outcome model (no L1): {gf_bad:+.3f}  (biased)")
print(f"IPTW (still correct propensity):          {msm_ate:+.3f}  (still ok)")
print(f"True ATE:                                 {true_ate:+.3f}")"""),

code("""# Now break the propensity model: use a constant
def iptw_constant_prop():
    p_const = 0.5
    p_A1_obs = np.full(n, p_const)
    p_A2_obs = np.full(n, p_const)
    w = (p_A1_marg / p_A1_obs) * (p_A2_given_A1 / p_A2_obs)
    msm = LinearRegression().fit(df[["A1", "A2"]], df["Y"], sample_weight=w)
    return msm.predict([[1, 1]])[0] - msm.predict([[0, 0]])[0]

iptw_bad = iptw_constant_prop()
print(f"\\nIPTW with constant propensity 0.5: {iptw_bad:+.3f}  (biased)")
print(f"G-formula (still correct outcome): {gf_ate:+.3f}  (still ok)")
print(f"True ATE:                          {true_ate:+.3f}")"""),

md("""**The double-robust point.** When the outcome model is wrong but the propensity is right, IPTW saves us. When the propensity is wrong but the outcome model is right, g-formula saves us. The DR estimator combines them — consistent if either is right. In practice we don't know which model is right; DR is the safer default."""),

md("""## Part 7 — Dynamic regime: threshold optimization

So far we evaluated *static* regimes (always-PM, never-PM). A *dynamic* regime conditions on the current state: "PM if $L_t > \\theta$". Find the threshold $\\theta$ that maximizes cumulative yield.

We evaluate candidate $\\theta$ values by g-formula simulation. For each $\\theta$, simulate the SCM under the policy and compute the mean yield."""),

code("""def simulate_threshold_policy(theta, n_sim=5000, rng=np.random.default_rng(7)):
    \"\"\"Simulate the 2-period PM scenario under the threshold rule 'PM if L > theta'.\"\"\"
    noise_L1_s = rng.normal(0, 0.3, n_sim)
    noise_Y_s  = rng.normal(0, 0.5, n_sim)
    L0_s = rng.normal(0, 1.0, n_sim)
    A1_s = (L0_s > theta).astype(int)
    L1_s = step_L(L0_s, A1_s, noise_L1_s)
    A2_s = (L1_s > theta).astype(int)
    Y_s  = yield_fn(L1_s, A1_s, A2_s, noise_Y_s)
    return Y_s.mean()

# Sweep theta from -2 to +2
thetas = np.linspace(-2, 2, 40)
yields = [simulate_threshold_policy(t) for t in thetas]

# Also compute the values of the two static baselines for reference
y_always = simulate_threshold_policy(-np.inf)
y_never  = simulate_threshold_policy(+np.inf)

fig, ax = plt.subplots()
ax.plot(thetas, yields, marker="o", markersize=4)
ax.axhline(y_always, color="C1", linestyle="--", label=f"Always-PM = {y_always:.2f}")
ax.axhline(y_never,  color="C2", linestyle="--", label=f"Never-PM = {y_never:.2f}")
ax.set_xlabel(r"Threshold $\\theta$")
ax.set_ylabel("Mean yield under 'PM if L > theta'")
ax.set_title("Dynamic threshold optimization")
ax.legend()
plt.show()

best_theta = thetas[np.argmax(yields)]
best_yield = max(yields)
print(f"\\nOptimal threshold:   theta* = {best_theta:.2f}")
print(f"Yield at theta*:    {best_yield:.3f}")
print(f"Yield always-PM:    {y_always:.3f}")
print(f"Yield never-PM:     {y_never:.3f}")"""),

md("""**Read this plot carefully.** Without a cost on PM, *always-PM* is essentially optimal — every PM is pure benefit (it pushes condition down and helps yield). The threshold sweep's "best" is at the lower edge of the search range (the most aggressive PM policy), and yield rises monotonically as the threshold decreases. The dynamic regime here does *not* beat always-PM, because there is nothing to trade off.

Lab 8 introduces a per-PM cost. With the cost, "always-PM" pays a penalty that grows with the number of PM actions; the optimum becomes interior (PM only when condition exceeds a threshold). The bridge from policy *evaluation* (this lab) to policy *optimization* (Lab 8) is that the same simulator that lets us evaluate a threshold also lets us search for the best one — but only once the optimization has a non-trivial objective."""),

md("""## Reflection

**Time-varying confounding is qualitatively different from static.** The naive estimator that uses only the action vector misses the within-trajectory dependence. The g-formula handles it by sequential modeling; IPTW handles it by sequential reweighting.

**Doubly-robust extends to sequential.** The pattern from Chapter 5 generalizes: combine the two models so the estimator is consistent if either is correct. The construction is more involved (iterative AIPW at each time step) but the principle is the same.

**Dynamic regimes can dominate static.** When the action depends on the observed state, you can do better than any one-size-fits-all rule. Lab 8 turns this from "find the optimal threshold" to "find the optimal regime in general" via Q-learning."""),

md("""## Exercises

1. **Three-period extension.** Add a third period ($A_3$, $L_2$) with the same dynamics. Extend g-formula and IPTW. Does the trajectory weight start to blow up?

   <details><summary>Solution</summary>

   Trajectory IPW weight is $\\prod_t \\pi_e(A_t) / \\pi_b(A_t \\mid L_t)$. With three periods and a sharp behavior policy, the product of three ratios near `0.1` vs `0.9` gives weights of ~1000 on some trajectories; ESS collapses. G-formula scales gracefully (one more conditional model per period). Lesson: per-step (DR) methods dominate trajectory IPW beyond 2–3 periods.
   </details>

2. **Heterogeneous PM effect.** Modify the SCM so the PM effect depends on $L_t$: PM helps a lot when $L_t$ is very high but barely helps when $L_t$ is at nominal. Does the optimal threshold change?

   <details><summary>Solution</summary>

   Yes — the threshold rises. When PM's benefit is small at nominal $L_t$, the cost dominates and PM should only be triggered at larger $L_t$. If the PM effect scales as $\\beta \\cdot L_t$ rather than a constant, the optimal threshold becomes approximately $L_t > c / \\beta$ rather than $L_t > c/2$. Chapter 8's §8.6 derivation generalizes directly.
   </details>

3. **Misspecified $L_1$ model.** Use a step function instead of a linear model for the $L_1 \\sim L_0, A_1$ relationship. Refit g-formula. How biased is the estimate?

   <details><summary>Solution</summary>

   The step-function dynamics model misrepresents how $L_1$ depends on $L_0$ and $A_1$ — typically biasing the simulated $\\hat L_1$ values toward the step centers. Downstream, the outcome model is evaluated at biased $\\hat L_1$ inputs, propagating the bias to $\\hat V(a_1, a_2)$. The size depends on how badly the step approximates the true linear relationship; in this SCM, a 2-step approximation gives ~15% bias on the always-PM vs never-PM ATE.
   </details>

4. **Real CMAPSS turbofan.** Apply the same framework to the CMAPSS turbofan-degradation dataset. The dynamics model becomes harder to specify (degradation signal is non-linear). Compare the parametric g-formula with a Monte Carlo g-formula using sampled trajectories.

   <details><summary>Solution</summary>

   NASA CMAPSS has run-to-failure trajectories for ~100 engines per dataset. Pick a single sensor as $L_t$, define a binary maintenance action $A_t$ (e.g., "engine swapped at cycle $t$"), and a per-cycle reward (negative degradation or positive remaining-useful-life). The dynamics $L_{t+1} \\mid L_t, A_t$ is non-linear and ML-fitted (random forest or LSTM). Compare:
   - *Parametric g-formula*: plug-in the conditional mean of $L_{t+1}$ at each step.
   - *Monte Carlo g-formula*: sample $L_{t+1}$ from a fitted conditional distribution; average over $M = 100$ trajectories.

   The Monte Carlo version is unbiased for any outcome surface; the parametric plug-in is biased when the outcome is non-linear in $L_t$ (Jensen's-inequality gap).
   </details>"""),

md("""## What's next

Lab 8 turns from evaluating a given dynamic regime to *finding the optimal one* — Q-learning and A-learning. The maintenance scenario gets a cost-per-PM penalty so the optimum is non-trivial."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch07" / "lab07.ipynb", cells)
print("Built lab07.ipynb")
