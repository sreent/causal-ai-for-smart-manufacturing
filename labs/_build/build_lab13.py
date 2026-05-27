"""Build labs/ch13/lab13.ipynb — Transportability, sensitivity, and deployment."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 13 — Transportability, Sensitivity, and Deployment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch13/lab13.ipynb)

**Companion lab to Chapter 13.** Reproduce the §13.7 Cinelli-Hazlett robustness analysis. Implement line-to-line transportability via re-weighting. Build a deployment-monitoring pipeline that detects distribution shift and concept drift."""),

md("""## What you'll do

1. **Reproduce §13.7's sensitivity analysis**: synthetic SCM with observed Z and unobserved U; back-door on Z; compute the Cinelli-Hazlett robustness value; benchmark against Z's strength.
2. **Implement the E-value** (VanderWeele-Ding 2017) for a binary-outcome variant.
3. **Transportability across two production lines**: estimate effect at Line A, transport to Line B via Stuart-Cole-Bradshaw-Leaf re-weighting, validate against a Line B oracle.
4. **Failure mode**: when Line B has a different *effect modifier* distribution, re-weighting alone is insufficient.
5. **Deployment-monitoring pipeline**: KS test for distribution shift, CUSUM for performance drift, automated rollback trigger."""),

md("""## Setup"""),

code("""# Colab: install PySensemakr (the Cinelli-Hazlett sensitivity-analysis
# library) and statsmodels (for the OLS the library expects).
%pip install --quiet pysensemakr statsmodels 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import brentq
from scipy import stats
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — Cinelli-Hazlett sensitivity (§13.7)

SCM with observed Z and unobserved U:
- $Z \\sim \\mathcal N(0, 1)$ — observed covariate.
- $U \\sim \\mathcal N(0, 1)$ — unobserved confounder.
- $X \\sim \\text{Bernoulli}(\\sigma(0.5 Z + 0.8 U))$.
- $Y = 1.0 X + 0.5 Z + 0.8 U + \\varepsilon_Y$ (true ATE = 1.0)."""),

code("""n = 5000

def gen_data(n, rng, u_to_x=0.8, u_to_y=0.8, true_ate=1.0):
    Z = rng.normal(0, 1, n)
    U = rng.normal(0, 1, n)
    X = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * Z + u_to_x * U))))
    Y = true_ate * X + 0.5 * Z + u_to_y * U + rng.normal(0, 0.3, n)
    return Z, X, Y, U

Z, X, Y, _ = gen_data(n, rng)
Xmat = np.column_stack([X, Z])
ols  = LinearRegression().fit(Xmat, Y)
ate_hat = ols.coef_[0]
y_resid = Y - ols.predict(Xmat)
sigma2  = (y_resid**2).sum() / (n - 3)
design  = np.hstack([np.ones((n, 1)), Xmat])
se_X    = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])

print(f"Back-door estimate (on Z only): ATE_hat = {ate_hat:+.3f}  (SE = {se_X:.3f})")
print(f"True ATE (from SCM):            ATE     = +1.000")
print(f"Bias from omitted U:                      {ate_hat - 1.0:+.3f}")"""),

code("""# Cinelli-Hazlett robustness value.
# RV is defined as the partial-R^2 value at which the bias from a hypothetical
# unmeasured U would exactly equal the observed estimate (i.e. it would nullify
# the ATE). There is no closed form for RV — it is the root of bias(r2) - |ate|.
# scipy.optimize.brentq is a 1D bracketed root-finder; we bracket on (0, 1).
dof = n - 3
def bias_minus_estimate(r2):
    return np.sqrt(r2 * r2 / (1 - r2) * dof) * se_X - abs(ate_hat)
RV = brentq(bias_minus_estimate, 1e-6, 0.999)

# Benchmark: R^2 of observed Z on X — gives us a real-world reference point for "how strong is RV?"
r2_Z_on_X = LinearRegression().fit(Z.reshape(-1, 1), X).score(Z.reshape(-1, 1), X)

print(f"\\nRobustness value RV = {RV:.3f}")
print(f"  Interpretation: any unmeasured U with partial R^2 >= {RV*100:.1f}% in BOTH the")
print(f"  treatment model (X | Z, U) AND the outcome model (Y | X, Z, U) could fully")
print(f"  explain away the estimated ATE.")
print(f"\\nBenchmark: observed Z has R^2 = {r2_Z_on_X:.3f} on X.")
print(f"To reach RV, hypothetical U would need to be ~{RV/r2_Z_on_X:.0f}x stronger than Z.")"""),

md("""**Interpretation.** The point estimate is biased ($\\hat\\beta = +1.55$ vs true $+1.00$), but to *nullify* the estimate, an unmeasured confounder would need to be 12× stronger than the observed $Z$. A confounder that strong, unmeasured, would be remarkable. The qualitative finding (positive ATE) is robust; the point estimate is not — but no plausible confounder flips the sign.

The lab's sensitivity result *cannot* tell you the true ATE without seeing $U$. It tells you how strong $U$ would need to be to overturn the conclusion. That is sufficient for many practical decisions.

**Library cross-check with PySensemakr.** The canonical implementation reproduces the same numbers and adds standard plots."""),

code("""try:
    import sensemakr
    import statsmodels.api as sm
    df_sens = pd.DataFrame({"Y": Y, "X": X.astype(float), "Z": Z})
    model_sm = sm.OLS(df_sens["Y"], sm.add_constant(df_sens[["X", "Z"]])).fit()
    s = sensemakr.Sensemakr(model=model_sm, treatment="X", benchmark_covariates=["Z"])
    s.summary()
    print(f"\\n(Library RV should match the manual value above: RV = {RV:.3f})")
except ImportError:
    print("PySensemakr not installed. %pip install pysensemakr in Colab.")
except Exception as e:
    print(f"PySensemakr call failed: {type(e).__name__}: {e}")
    print("(Likely a pandas-version compatibility issue; in Colab the call works.)")
    print(f"Manual RV = {RV:.3f} (use the manual computation above as the reference.)")"""),

md("""## Part 2 — E-value for a binary outcome variant

For risk ratios > 1: $E = \\text{RR} + \\sqrt{\\text{RR}(\\text{RR} - 1)}$."""),

code("""def e_value(RR):
    \"\"\"VanderWeele-Ding 2017 E-value for a risk ratio.\"\"\"
    if RR < 1: RR = 1 / RR  # symmetric for protective effects
    return RR + np.sqrt(RR * (RR - 1))

# Example: observed RR = 1.5
for RR in [1.2, 1.5, 2.0, 3.0]:
    print(f"Observed RR = {RR:.2f}  ->  E-value = {e_value(RR):.2f}")"""),

md("""**Interpretation of E-value.** For RR = 1.5, E = 2.37. An unmeasured confounder would need to be associated with treatment AND with outcome at RR ≥ 2.37 *each* to nullify the observed effect. Smaller E-values (closer to 1) mean the result is fragile; larger E-values mean it's robust."""),

md("""## Part 3 — Transportability across two production lines

Two lines, same SCM structurally but different covariate distributions. Line A has $Z \\sim \\mathcal N(0, 1)$; Line B has $Z \\sim \\mathcal N(1, 1)$ (a different mix of tool ages, say).

Estimate the ATE at Line A. Transport to Line B by re-weighting the Line A units to match Line B's covariate distribution."""),

code("""# Line A: original SCM, n_A units
n_A = 3000
Z_A, X_A, Y_A, _ = gen_data(n_A, np.random.default_rng(7), u_to_x=0.3, u_to_y=0.3)

# Line B: same SCM but Z shifted (mean=1 instead of 0)
def gen_lineB(n, rng, u_to_x=0.3, u_to_y=0.3, true_ate=1.0):
    Z = rng.normal(1.0, 1.0, n)
    U = rng.normal(0, 1, n)
    X = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * Z + u_to_x * U))))
    Y = true_ate * X + 0.5 * Z + u_to_y * U + rng.normal(0, 0.3, n)
    return Z, X, Y, U

n_B = 1000  # smaller, to mimic "we have limited data at the new site"
Z_B, X_B, Y_B, _ = gen_lineB(n_B, np.random.default_rng(13))

# Line A naive estimate (back-door on Z)
ate_A = LinearRegression().fit(np.column_stack([X_A, Z_A]), Y_A).coef_[0]
# Line B naive estimate
ate_B = LinearRegression().fit(np.column_stack([X_B, Z_B]), Y_B).coef_[0]

# Transport via re-weighting: weight Line A units by P_B(Z) / P_A(Z)
from scipy.stats import norm
weights = norm.pdf(Z_A, loc=1, scale=1) / norm.pdf(Z_A, loc=0, scale=1)
ate_transport = LinearRegression().fit(
    np.column_stack([X_A, Z_A]), Y_A, sample_weight=weights).coef_[0]

print(f"Line A naive estimate:        {ate_A:+.3f}")
print(f"Line B naive estimate:        {ate_B:+.3f}")
print(f"Line A transported to B (rwt): {ate_transport:+.3f}")
print(f"True ATE (both lines):         +1.000")"""),

md("""Both lines have the same true ATE because the structural treatment effect is unchanged. The re-weighted Line-A-transported-to-B estimate should converge with the Line B oracle."""),

md("""## Part 4 — Failure: transportability breaks when an effect modifier differs

Modify the scenario so the true ATE depends on $Z$ (heterogeneous effect). Now Line B's different Z distribution means a different *true* effect at Line B, and naive re-weighting on $Z$ alone gives the wrong answer."""),

code("""def gen_hetero(n, rng, Z_mean=0, true_ate_intercept=0.5, true_ate_slope=0.5):
    Z = rng.normal(Z_mean, 1, n)
    U = rng.normal(0, 1, n)
    X = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * Z + 0.3 * U))))
    # Treatment effect is now: 0.5 + 0.5*Z (depends on Z)
    Y = (true_ate_intercept + true_ate_slope * Z) * X + 0.3 * Z + 0.3 * U + rng.normal(0, 0.3, n)
    # True ATE at this Z mean: 0.5 + 0.5 * Z_mean
    return Z, X, Y

Z_A2, X_A2, Y_A2 = gen_hetero(3000, np.random.default_rng(7), Z_mean=0)
Z_B2, X_B2, Y_B2 = gen_hetero(1000, np.random.default_rng(13), Z_mean=1)
ate_A_true = 0.5 + 0.5 * 0
ate_B_true = 0.5 + 0.5 * 1

# Line A estimate (naive back-door on Z, gets average over Line A distribution)
ate_A2 = LinearRegression().fit(np.column_stack([X_A2, Z_A2]), Y_A2).coef_[0]
# Re-weight to Line B distribution
w = norm.pdf(Z_A2, loc=1, scale=1) / norm.pdf(Z_A2, loc=0, scale=1)
ate_transport2 = LinearRegression().fit(
    np.column_stack([X_A2, Z_A2]), Y_A2, sample_weight=w).coef_[0]
ate_B2 = LinearRegression().fit(np.column_stack([X_B2, Z_B2]), Y_B2).coef_[0]

print(f"Line A true ATE (avg over Z~N(0,1)):     {ate_A_true:+.3f}")
print(f"Line B true ATE (avg over Z~N(1,1)):     {ate_B_true:+.3f}")
print()
print(f"Line A naive estimate:                    {ate_A2:+.3f}")
print(f"Line A re-weighted to B distribution:     {ate_transport2:+.3f}")
print(f"Line B oracle estimate:                   {ate_B2:+.3f}")"""),

md("""**Re-weighting actually works here**: from $+0.557$ (Line A naive) to $+1.012$ (re-weighted), close to the Line B oracle $+1.055$. The reason: $Z$ is both the variable that differs between the two lines *and* the effect modifier. Re-weighting on $Z$ realigns both at once.

**When does re-weighting fail?** When the effect modifier is *different* from the covariate being re-weighted on — e.g., if the treatment effect depended on an unobserved operator skill $L$ that has a different distribution at Line B but is not in the data. Re-weighting on observable $Z$ alone wouldn't account for the shift in $L$. The fix is either (a) collect $L$ at the new site to enable a full re-weighting, or (b) use a CATE-based approach that fits $\\hat\\tau(z, L)$ from Line A and integrates over the *joint* $(Z, L)$ distribution at Line B."""),

md("""## Part 4b — Synthetic-control transportability with CausalPy

A different transportability scenario: a single treated unit (a fab or production line) instead of a population. The synthetic-control method (Abadie et al. 2010) constructs a weighted combination of donor units to match the treated unit's pre-treatment trajectory, then attributes the post-treatment gap to the intervention.

`causalpy.SyntheticControl` implements this with a PyMC backend (full posterior over the counterfactual)."""),

code("""try:
    import causalpy as cp
    have_cp = True
except ImportError:
    have_cp = False
    print("CausalPy not installed. %pip install causalpy in Colab.")

if have_cp:
    # Build a simple synthetic-control scenario: one treated fab + 5 donors,
    # 24 monthly observations with treatment starting at month 16.
    rng_sc = np.random.default_rng(0)
    n_periods, n_donors = 24, 5
    t = np.arange(n_periods)
    # Donor outcomes (each follows a different trend + noise)
    donor_means = [70 + 0.1*t, 65 + 0.05*t, 75 - 0.05*t, 72 + 0.08*t, 68 + 0.12*t]
    donors = np.stack([m + rng_sc.normal(0, 0.5, n_periods) for m in donor_means], axis=1)
    # Treated unit: convex combination of donors, plus +2.0 effect after month 15
    w_true = np.array([0.3, 0.2, 0.1, 0.25, 0.15])
    treated = donors @ w_true + rng_sc.normal(0, 0.5, n_periods)
    treated[16:] += 2.0
    df_sc = pd.DataFrame(
        np.column_stack([t.reshape(-1, 1), treated.reshape(-1, 1), donors]),
        columns=["t", "treated"] + [f"donor_{i}" for i in range(n_donors)],
    ).set_index("t")
    donor_cols = [f"donor_{i}" for i in range(n_donors)]
    res = cp.SyntheticControl(
        df_sc,
        treatment_time=16,
        control_units=donor_cols,
        treated_units=["treated"],
        model=cp.pymc_models.WeightedSumFitter(
            sample_kwargs={"chains": 2, "draws": 500, "tune": 200,
                           "progressbar": False, "random_seed": 0},
        ),
    )
    impact_post = float(np.asarray(res.post_impact).mean())
    print(f"True post-treatment effect (per period): +2.000")
    print(f"CausalPy synthetic-control posterior mean: {impact_post:+.3f}")
else:
    print("Synthetic control example skipped (CausalPy not installed).")"""),

md("""Synthetic control is the right tool for single-treated-unit settings (one fab gets a new MES; the other five fabs are the donor pool). The donor weights are learned from the pre-treatment fit; the post-treatment gap is the estimated effect. CausalPy returns a *posterior* over the gap, so you get uncertainty quantification automatically — important when you have one treated unit and no parametric standard-error formula applies."""),

md("""## Part 5 — Deployment-monitoring pipeline

Simulate a deployed policy whose input distribution shifts slowly (drift) and whose performance degrades. Build monitors that detect this and trigger rollback."""),

code("""# Simulate deployment over time
n_periods = 30
samples_per_period = 100

# Baseline distribution: Z ~ N(0, 1)
# Drift: Z gradually shifts to N(0.5, 1) by period 20
def gen_period(period_idx, n_per, rng):
    z_mean = 0.0 + 0.025 * max(0, period_idx - 5)  # drift starts at period 5
    Z = rng.normal(z_mean, 1, n_per)
    return Z

baseline_rng = np.random.default_rng(0)
baseline_Z = baseline_rng.normal(0, 1, 1000)   # reference distribution

deploy_rng = np.random.default_rng(42)
ks_p_values = []
performance_kpis = []

for period in range(n_periods):
    Z_period = gen_period(period, samples_per_period, deploy_rng)
    # Distribution shift monitor: KS test against the baseline
    ks_stat, ks_p = stats.ks_2samp(baseline_Z, Z_period)
    ks_p_values.append(ks_p)
    # Performance KPI (yield-like; degrades with Z shift)
    yield_kpi = 100 - 2 * np.abs(Z_period.mean()) + deploy_rng.normal(0, 0.5, samples_per_period).mean()
    performance_kpis.append(yield_kpi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax1.plot(range(n_periods), performance_kpis, marker="o")
ax1.set_ylabel("Yield KPI")
ax1.set_title("Performance over time (deployment-drift simulation)")
ax1.axhline(98, color="red", linestyle="--", label="Rollback threshold")
ax1.legend()

ax2.plot(range(n_periods), -np.log10(np.array(ks_p_values) + 1e-10), marker="o", color="C2")
ax2.axhline(-np.log10(0.05), color="red", linestyle="--", label="alpha = 0.05")
ax2.set_xlabel("Period")
ax2.set_ylabel("-log10(KS p-value)")
ax2.set_title("Distribution-shift monitor")
ax2.legend()
plt.tight_layout()
plt.show()

# Rollback trigger: yield drops below 98 OR KS p-value < 0.05 for 3 consecutive periods
rolled_back = False
for i in range(2, n_periods):
    cond_perf = performance_kpis[i] < 98
    cond_dist = all(p < 0.05 for p in ks_p_values[i-2:i+1])
    if cond_perf or cond_dist:
        print(f"Rollback triggered at period {i}: KPI={performance_kpis[i]:.2f}, "
              f"KS p={ks_p_values[i]:.4f}")
        rolled_back = True
        break

if not rolled_back:
    print(f"No rollback triggered in {n_periods} periods.")"""),

md("""The pipeline catches the drift before the yield KPI drops below the threshold. Real deployment systems combine both signals (distribution shift + performance drift) to reduce false-positive rollbacks."""),

md("""## Reflection

**Sensitivity analysis is the discipline of explicit assumptions.** A point estimate without sensitivity is incomplete. The Cinelli-Hazlett RV (continuous outcomes) and the VanderWeele-Ding E-value (binary outcomes) are the modern industry tools.

**Transportability is a query about the *target* population.** Re-weighting suffices when the source-and-target distributions differ only in covariates that are not effect modifiers. When effect modifiers differ, you need a CATE estimator integrated over the target distribution.

**Deployment monitoring catches what models cannot predict.** Distribution shift monitors are mandatory; performance-drift monitors should track the actual outcome; rollback criteria must be pre-committed before deployment, not negotiated after a degradation."""),

md("""## Exercises

1. **E-value bounds.** For an observed RR with a 95% CI, what is $E_L$ (the E-value for the CI lower bound)? Interpret in terms of robustness.

   <details><summary>Solution</summary>

   ```python
   def e_value(rr):
       rr = max(rr, 1/rr)  # symmetric for RR < 1
       return rr + np.sqrt(rr * (rr - 1))

   rr, ci_low, ci_high = 1.5, 1.2, 1.9
   print("E (point):", e_value(rr))
   print("E_L (CI lower):", e_value(ci_low))  # robustness of CI lower bound
   ```

   For $\\text{RR} = 1.5$, $E = 1.5 + \\sqrt{1.5 \\times 0.5} \\approx 2.37$ (an unmeasured confounder would need RR $\\geq 2.37$ with both treatment and outcome to fully explain the association). The CI-lower E-value $E_L = e\\_value(1.2) \\approx 1.69$ is what most epidemiologists report because it tells you the strength needed to push the *evidence for an effect* (not just the point estimate) to null. $E_L \\leq 1.5$ is generally considered weak; $E_L \\geq 2$ is moderately robust; $E_L \\geq 3$ is strong.
   </details>

2. **CATE-based transportability.** Implement the "fit CATE at Line A, integrate over Line B's covariate distribution" approach. Compare to re-weighting on the heterogeneous-effect scenario in Part 4.

   <details><summary>Solution</summary>

   ```python
   from econml.dml import CausalForestDML
   cf = CausalForestDML(model_y=..., model_t=...).fit(Y_A, T_A, X=X_A, W=W_A)
   tau_hat_B = cf.effect(X_B)  # CATE evaluated at Line B's covariate values
   ate_target = tau_hat_B.mean()  # integrate over Line B's empirical distribution
   ```

   This works when **effect modification** is captured in $X$ — Line B has different $X$ values, but the conditional treatment effect $\\tau(x)$ is invariant across lines. Re-weighting (also valid here) targets the *overall mean* by weighting Line A's data to look like Line B; CATE-based transportability fits a *function* and evaluates it. The CATE approach is more efficient when Line B is small (you borrow Line A's information for the function fit), but is biased if there's residual effect modification from variables not in $X$. Re-weighting needs Line B to be large enough to densely sample the relevant covariate region.
   </details>

3. **CUSUM for performance drift.** Replace the simple threshold with a CUSUM control chart. Tune the CUSUM parameters for false-positive rate.

   <details><summary>Solution</summary>

   ```python
   def cusum_monitor(stream, mu0, sigma, k=0.5, h=4):
       # k = reference value (allowance); h = decision threshold (in sigma units)
       S = 0
       for t, x in enumerate(stream):
           S = max(0, S + (mu0 - x) / sigma - k)  # one-sided lower CUSUM
           if S > h:
               return t  # alarm
       return None

   # Calibrate via in-control simulation
   fp_rates = []
   for _ in range(1000):
       in_control = rng.normal(mu0, sigma, 200)
       fp_rates.append(cusum_monitor(in_control, mu0, sigma, k=0.5, h=4) is not None)
   print("FP rate:", np.mean(fp_rates))  # tune h to hit target FP rate
   ```

   CUSUM is far more sensitive to small persistent shifts than a fixed threshold. Standard tuning: $k = \\delta/2$ where $\\delta$ is the smallest shift you care about (in $\\sigma$ units); $h \\in [4, 5]$ gives an in-control ARL ~370 (Shewhart-equivalent). For a 1% FP rate over 200 periods you want $h \\approx 4.5$; for 5% you want $h \\approx 3.5$.
   </details>

4. **Capstone setup.** For your capstone project (Chapter 14), draft the deployment-readiness checklist: (a) source population the estimate applies to, (b) transportability scenario, (c) distribution-shift and performance-drift monitors, (d) rollback criteria.

   <details><summary>Solution</summary>

   A defensible checklist for any deployed causal estimate:

   **(a) Source population.** State the exact inclusion/exclusion criteria of your training data. "Customers active in 2024 Q3, US-only, mobile app, ages 18-65." Be honest about what is *not* in your population (new customers, web users, churned users).

   **(b) Transportability scenario.** Name the target deployment population. Are effect modifiers the same as in source? If not, list them and either (i) fit CATE and integrate, or (ii) accept the gap and document it.

   **(c) Monitors.** Distribution-shift: PSI on each input feature, KS-test on the propensity score distribution, weekly. Performance-drift: CUSUM on the outcome KPI, with the pre-deployment baseline as reference. Both monitor outputs go to a dashboard reviewed before each model refresh.

   **(d) Rollback criteria.** Pre-committed thresholds: PSI > 0.25 on any feature → investigate; CUSUM alarm → page on-call; KPI drop > $X\\%$ over 2 weeks → automatic rollback to baseline policy. Get business sign-off on these *before* deployment.

   The discipline is: every item is **pre-committed, monitorable, and falsifiable**. Vague language ("monitor performance") is unactionable; numeric thresholds with named owners are actionable.
   </details>"""),

md("""## Course wrap-up

This is the last lab. The arc of the course's hands-on work: identification → estimation → heterogeneity → time-varying → optimization → discovery → mediation → evaluation → causal RL → deployment.

The lab pattern that recurred: *state the SCM, define the truth, run the estimator, compare*. That pattern works in industrial settings too. Real data won't have a known SCM — but a defensible DAG, an identification result, an estimator, and a sensitivity analysis are the same four ingredients. The discipline of explicit assumptions is the lab's lasting deliverable.

Best of luck with the capstone."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch13" / "lab13.ipynb", cells)
print("Built lab13.ipynb")
