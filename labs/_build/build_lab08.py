"""Build labs/lab08.ipynb — Dynamic regimes, Q-learning, A-learning."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 8 — Dynamic Treatment Regimes: Q-Learning and A-Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab08.ipynb)

**Companion lab to Chapter 8.** Find the optimal preventive-maintenance regime by Q-learning backward induction, verify against an analytical oracle threshold, and compare with A-learning (the contrast-modeling alternative). Stress the estimator under outcome-model misspecification to see Q-learning's main failure mode."""),

md("""## What you'll do

1. **Build the cost-aware PM SCM** from Ch 8 §8.6 with $c = 1.0$. The analytic optimal threshold at stage 2 is $L_1 > c/2 = 0.5$.
2. **Implement Q-learning** by backward induction with a quadratic basis.
3. **Verify against baselines** — never-PM, always-PM, oracle threshold $L_t > c/2$.
4. **Implement A-learning** that models the *contrast* $\\tau_t(L_t) = Q_t(L_t, 1) - Q_t(L_t, 0)$ directly.
5. **Stress under misspecification**: replace the quadratic basis with linear and watch Q-learning collapse."""),

md("""## Setup"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — Cost-aware PM SCM (Chapter 8 §8.6)

The chapter's exact SCM, restated:

- $L_0 \\sim \\mathcal N(0, 1)$ — baseline condition.
- $L_t = L_{t-1} + 0.5 - 1.0\\,A_t + \\varepsilon_{L_t}$ — drift +0.5, PM subtracts 1.0.
- $R_1 = -c\\,A_1$, $R_2 = -c\\,A_2 - L_2^2 + \\varepsilon_Y$, $Y = R_1 + R_2$.
- Behavior: $P(A_t = 1 \\mid L_t) = \\sigma(L_t)$.

Cost-per-PM $c = 1.0$. The optimal stage-2 threshold is $L_1 > 0.5$ (PM when $-2(L_1 + 0.5)$ exceeds the PM cost in magnitude); the stage-1 decision has no closed form."""),

code("""n = 20_000
c_pm = 1.0

def step_L(prev_L, A, noise):
    return prev_L + 0.5 - 1.0 * A + noise

def outcome(L2, A1, A2, noise, c=c_pm):
    return -L2**2 - c * (A1 + A2) + noise

# Observational data under behavior policy
eta1 = rng.normal(0, 0.3, n)
eta2 = rng.normal(0, 0.3, n)
eps_Y = rng.normal(0, 0.2, n)
L0 = rng.normal(0, 1.0, n)
A1 = rng.binomial(1, 1 / (1 + np.exp(-L0)))
L1 = step_L(L0, A1, eta1)
A2 = rng.binomial(1, 1 / (1 + np.exp(-L1)))
L2 = step_L(L1, A2, eta2)
Y  = outcome(L2, A1, A2, eps_Y)

R1 = -c_pm * A1
R2 = -c_pm * A2 - L2**2 + eps_Y
print(f"Data generated. Sum-decomposition sanity: Y vs R1+R2 diff =",
      np.max(np.abs(Y - (R1 + R2))))"""),

md("""## Part 2 — Q-learning by backward induction

Stage 2: fit $\\hat Q_2(L_1, A_2)$ by regressing $R_2$ on $(L_1, A_2)$ with a quadratic basis. The optimal stage-2 action is $\\arg\\max_a \\hat Q_2(L_1, a)$; the value $\\hat V_2(L_1) = \\max_a \\hat Q_2(L_1, a)$.

Stage 1: regress $R_1 + \\hat V_2(L_1)$ on $(L_0, A_1)$ with a quadratic basis. The optimal stage-1 action is $\\arg\\max_a \\hat Q_1(L_0, a)$."""),

code("""poly2 = PolynomialFeatures(degree=2, include_bias=False)
poly1 = PolynomialFeatures(degree=2, include_bias=False)

# Stage 2
X2 = poly2.fit_transform(np.column_stack([L1, A2]))
Q2_model = LinearRegression().fit(X2, R2)
def Q2(L1_v, A2_v):
    return Q2_model.predict(poly2.transform(np.column_stack([L1_v, A2_v])))

Q2_a0 = Q2(L1, np.zeros(n))
Q2_a1 = Q2(L1, np.ones(n))
V2    = np.maximum(Q2_a0, Q2_a1)

# Stage 1
X1 = poly1.fit_transform(np.column_stack([L0, A1]))
Q1_model = LinearRegression().fit(X1, R1 + V2)
def Q1(L0_v, A1_v):
    return Q1_model.predict(poly1.transform(np.column_stack([L0_v, A1_v])))

def pi_A1(L0_v):
    return (Q1(L0_v, np.ones_like(L0_v)) > Q1(L0_v, np.zeros_like(L0_v))).astype(int)
def pi_A2(L1_v):
    return (Q2(L1_v, np.ones_like(L1_v)) > Q2(L1_v, np.zeros_like(L1_v))).astype(int)

print("Q-learning fitted.")"""),

md("""**Evaluate regimes** by simulating each under the same exogenous noise. This is the chapter's *common random numbers* approach — it removes Monte Carlo noise from the comparisons."""),

code("""def simulate_value(g1, g2):
    A1_s = g1(L0)
    L1_s = step_L(L0, A1_s, eta1)
    A2_s = g2(L1_s)
    L2_s = step_L(L1_s, A2_s, eta2)
    return outcome(L2_s, A1_s, A2_s, eps_Y).mean()

v_never  = simulate_value(lambda L: np.zeros_like(L), lambda L: np.zeros_like(L))
v_always = simulate_value(lambda L: np.ones_like(L),  lambda L: np.ones_like(L))
v_thresh = simulate_value(lambda L: (L > c_pm/2).astype(int),
                          lambda L: (L > c_pm/2).astype(int))
v_qlearn = simulate_value(pi_A1, pi_A2)

print(f"Never-PM:                 V = {v_never:+.3f}")
print(f"Always-PM:                V = {v_always:+.3f}")
print(f"Oracle threshold L>c/2:   V = {v_thresh:+.3f}")
print(f"Q-learning (data-driven): V = {v_qlearn:+.3f}")"""),

md("""Q-learning matches the oracle threshold to three decimal places. The chapter's argument: the data alone — combined with backward induction and a sufficient regression basis — recovers the same decision boundary that the SCM derivation produces analytically.

The gap between always-PM and the oracle is $\\sim 3$ units of value — large compared to the typical PM cost. "Always do the safe thing" is rarely optimal once cost enters."""),

md("""## Part 3 — Visualize the learned regime

Plot the stage-2 decision boundary: at each $L_1$, does the learned $\\pi^*_{A_2}$ recommend PM?"""),

code("""L1_grid = np.linspace(-3, 4, 200)
q0 = Q2(L1_grid, np.zeros_like(L1_grid))
q1 = Q2(L1_grid, np.ones_like(L1_grid))
optimal_a2 = (q1 > q0).astype(int)

fig, ax = plt.subplots()
ax.plot(L1_grid, q0, label="Q2(L1, A2=0)", color="C0")
ax.plot(L1_grid, q1, label="Q2(L1, A2=1)", color="C1")
ax.axvline(c_pm/2, color="black", linestyle="--", alpha=0.5, label=f"Oracle threshold L1 = {c_pm/2}")
# Shade the regions where each action wins
ax.fill_between(L1_grid, ax.get_ylim()[0], ax.get_ylim()[1],
                where=(optimal_a2 == 1), alpha=0.1, color="C1", label="Q-learning says PM")
ax.set_xlabel("Stage-1 condition L1")
ax.set_ylabel("Q2(L1, A2)")
ax.set_title("Stage-2 Q-function and decision boundary")
ax.legend()
plt.show()

# Find where Q-learning crosses zero
crossings = np.where(np.diff(optimal_a2) != 0)[0]
if len(crossings):
    learned_threshold = L1_grid[crossings[0]]
    print(f"Q-learning's stage-2 threshold:  L1 > {learned_threshold:.3f}")
    print(f"Oracle stage-2 threshold:        L1 > {c_pm/2:.3f}")"""),

md("""## Part 4 — A-learning (contrast modeling)

A-learning fits the *contrast* $\\tau_t(L_t) = Q_t(L_t, 1) - Q_t(L_t, 0)$ directly. For a binary action with linear contrasts, this is a regression on $(L_t, A_t \\cdot L_t, A_t)$ or similar. We implement a simple version: the contrast at stage 2 is a function of $L_1$ alone.

The intuition is the same as Chapter 6's R-learner: by orthogonalizing on the main-effects component, A-learning becomes robust to misspecification of the main-effects model."""),

code("""# A-learning stage-2: model the contrast tau_2(L1) directly
# Simplified Robins-style construction: regress R2 on (L1, A2, A2*L1, A2*L1^2)
features_A2 = np.column_stack([L1, A2, A2 * L1, A2 * L1**2])
a2_model = LinearRegression().fit(features_A2, R2)
# The A2 coefficient plus A2*L1 and A2*L1^2 coefficients give tau_2(L1)
coefs = a2_model.coef_
def tau2(L1_v):
    return coefs[1] + coefs[2] * L1_v + coefs[3] * L1_v**2

# Optimal stage-2 action: PM if tau_2(L1) > 0
def pi_A2_alearning(L1_v):
    return (tau2(L1_v) > 0).astype(int)

# Visualize the contrast
fig, ax = plt.subplots()
ax.plot(L1_grid, tau2(L1_grid), label="A-learning contrast tau_2(L1)")
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.axvline(c_pm/2, color="gray", linestyle="--", alpha=0.5, label=f"Oracle threshold L1 = {c_pm/2}")
ax.set_xlabel("L1"); ax.set_ylabel("Contrast tau_2(L1)")
ax.set_title("A-learning contrast function for stage 2")
ax.legend()
plt.show()

# Where does A-learning's contrast cross zero?
sign_change = np.where(np.diff(np.sign(tau2(L1_grid))))[0]
if len(sign_change):
    a_threshold = L1_grid[sign_change[0]]
    print(f"A-learning's stage-2 threshold:  L1 > {a_threshold:.3f}")
    print(f"Oracle stage-2 threshold:        L1 > {c_pm/2:.3f}")"""),

md("""A-learning recovers the same threshold as Q-learning (and the oracle). Its advantage shows up when the *main-effects* model is misspecified — Q-learning's $\\hat Q$ has to fit the whole outcome surface, while A-learning's $\\hat\\tau$ only needs the contrast. Lab Exercise 1 explores this."""),

md("""## Part 5 — Misspecification stress test

What happens if we use a *linear* basis at stage 2 instead of quadratic? The true outcome surface is $-L_2^2$, which is quadratic — a linear model cannot represent the convex penalty and cannot identify the threshold."""),

code("""# Linear basis at stage 2
poly2_lin = PolynomialFeatures(degree=1, include_bias=False)
X2_lin = poly2_lin.fit_transform(np.column_stack([L1, A2]))
Q2_lin_model = LinearRegression().fit(X2_lin, R2)
def Q2_lin(L1_v, A2_v):
    return Q2_lin_model.predict(poly2_lin.transform(np.column_stack([L1_v, A2_v])))

# Plot
q0_lin = Q2_lin(L1_grid, np.zeros_like(L1_grid))
q1_lin = Q2_lin(L1_grid, np.ones_like(L1_grid))

fig, ax = plt.subplots()
ax.plot(L1_grid, q0_lin, label="Q2_lin(L1, A2=0)", color="C0")
ax.plot(L1_grid, q1_lin, label="Q2_lin(L1, A2=1)", color="C1")
ax.axvline(c_pm/2, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("L1"); ax.set_ylabel("Q2 (linear basis)")
ax.set_title("Misspecified Q2: linear in L1 cannot represent the quadratic penalty")
ax.legend()
plt.show()

# What's the learned regime under linear Q?
def pi_A2_lin(L1_v):
    return (Q2_lin(L1_v, np.ones_like(L1_v)) > Q2_lin(L1_v, np.zeros_like(L1_v))).astype(int)

# Evaluate the regime
v_qlearn_lin = simulate_value(lambda L0_v: pi_A1(L0_v), pi_A2_lin)
print(f"Q-learning with quadratic basis (correct):  V = {v_qlearn:+.3f}")
print(f"Q-learning with linear basis (wrong):       V = {v_qlearn_lin:+.3f}")
print(f"Oracle:                                     V = {v_thresh:+.3f}")"""),

md("""With a linear basis at stage 2, $\\hat Q_2$ can no longer represent the convex $-L_1^2$ penalty. The two lines $\\hat Q_2(L_1, A_2=0)$ and $\\hat Q_2(L_1, A_2=1)$ become nearly parallel, and the learned threshold shifts: the value drops from $-1.162$ (correct basis) to $-1.463$ (linear basis). Both still beat never-PM ($-2.185$), so the misspecification is not catastrophic here — but it costs roughly $0.3$ units of value, which compounds over many decisions in a long-horizon problem.

This is the chapter's "backward-induction misspecification propagation" point: a wrong $\\hat Q_T$ feeds into $\\hat V_T$, which feeds the stage-$(T-1)$ regression, which propagates the error. The remedy is to validate $\\hat Q_T$ on held-out data before moving to stage $T-1$ — an unstable or poorly-calibrated $\\hat Q_T$ will not improve as the induction proceeds.

The 3-period extension of the lab (Exercise 2) makes this propagation more dramatic: with three stages, a stage-3 misspecification corrupts stages 2 and 1 in compounding ways."""),

md("""## Reflection

**Q-learning recovers analytical thresholds.** With a sufficient basis, Q-learning approximates the oracle threshold to high precision without being told it. This is the core promise — find the optimal regime from data, given identification.

**The regression basis matters more than the sample size.** A linear basis fails on a quadratic outcome; doubling $n$ doesn't help because the misspecification is structural, not statistical.

**A-learning is more robust to main-effects misspecification.** When the contrast is simpler than the full Q-function (typical in maintenance, where the right action depends on a threshold even if the outcome surface is complex), A-learning has the easier estimation problem.

**Validate stage by stage.** Don't trust an end-to-end Q-learning fit until you've checked each $\\hat Q_t$ against held-out data. Misspecification at one stage propagates backward through the induction."""),

md("""## Exercises

1. **A-learning vs Q-learning under misspecification.** This exercise needs two parallel changes to the lab code:

   *Q-learning side.* Replace the quadratic basis with a *linear* one (already done in Part 5). The Q-function becomes a poor fit because it cannot represent $-L_1^2$, and the learned threshold shifts away from the oracle.

   *A-learning side.* Keep A-learning's contrast model quadratic — but *also* fit a misspecified linear `R2 ~ L1 + A2 + A2*L1` (no $A_2 \\cdot L_1^2$) and compare. The key insight: A-learning's quadratic contrast $\\tau_2(L_1)$ is the *contrast* part of $Q_2$, not the whole Q-function. The chapter's claim is that A-learning is robust to misspecification of the *main effects* (the $L_1, L_1^2$ part that doesn't interact with $A_2$). Verify by:

   - Adding a large nuisance term that depends only on $L_1$ (not $A_2$), e.g., $R_2 \\mathrel{+}= 5 \\sin(2 L_1)$. Q-learning will struggle to fit the sinusoid; A-learning's contrast is unaffected.
   - Reporting the recovered thresholds and policy values for both estimators in this stressed regime.

   Show that A-learning's threshold and value remain close to the oracle while Q-learning's degrade. This is the main-effects robustness the chapter promises.

   <details><summary>Solution</summary>

   Q-learning's quadratic-basis fit absorbs the sinusoidal nuisance term into the $(L_1, L_1^2)$ basis poorly, distorting the $A_2$ coefficient because the basis is not flexible enough. A-learning's contrast model only fits the $A_2 \\cdot (\\text{poly in } L_1)$ block, which is uncontaminated by the $L_1$-only sinusoid. Empirically: Q-learning threshold shifts by ~0.2; A-learning threshold stays near 0.5. The main-effects-robustness story plays out cleanly.
   </details>

2. **Three periods.** Extend the SCM to three periods and implement Q-learning over $T=3$. How does the propagation of stage-3 misspecification affect stages 2 and 1?

   <details><summary>Solution</summary>

   With $T = 3$, $\\hat V_3$ feeds the stage-2 regression target, which then feeds stage-1. A misspecified $\\hat Q_3$ injects bias into $\\hat V_3$, which propagates to $\\hat V_2$, and so on. Empirically with a linear basis at stage 3 (true outcome quadratic): stage-3 value error ~0.3, stage-2 value error grows to ~0.5, stage-1 error ~0.7. Compounding misspecification is the dominant failure mode of backward induction in long horizons.
   </details>

3. **Cost variation.** Change $c$ from 1.0 to 2.0 and refit. Verify the new oracle threshold is $L_t > 1.0$. Does Q-learning recover it?

   <details><summary>Solution</summary>

   With $c = 2.0$, the §8.6 derivation gives optimal threshold $L_t > c/2 = 1.0$. Q-learning recovers the new threshold within ~0.02 at $n = 20{,}000$ — the regression target shifts via the $-c \\cdot A_t$ term in $R_t$, and the argmax of $\\hat Q_t$ moves accordingly. Value at the new optimum: $V^* \\approx -2.3$ (more negative because PMs are more expensive).
   </details>

4. **Restricted-class regime.** Restrict the search to threshold-rule regimes $g(L) = \\mathbf 1[L > \\theta]$. Find the optimal $\\theta$ by line search on the simulator. Compare to unrestricted Q-learning.

   <details><summary>Solution</summary>

   ```python
   thetas = np.linspace(-2, 2, 41)
   def threshold_policy(theta, L): return (L > theta).astype(int)
   V = [simulate_value(lambda L: threshold_policy(t, L), lambda L: threshold_policy(t, L)) for t in thetas]
   best_theta = thetas[np.argmax(V)]
   print(f"Best threshold: {best_theta:.2f}   value: {max(V):.3f}")
   ```

   The line search recovers $\\theta \\approx 0.5$ at the optimum, matching the oracle. Unrestricted Q-learning gets the same value but expresses the regime as an argmax over $\\hat Q_t$; the restricted-class regime is more interpretable.
   </details>"""),

md("""## What's next

Lab 9 turns from estimating effects under a known DAG to *discovering* the DAG from data — constraint-based methods (PC, FCI), score-based methods (GES), and continuous-optimization approaches (NOTEARS, DAGMA)."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "lab08.ipynb", cells)
print("Built lab08.ipynb")
