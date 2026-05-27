"""Build labs/ch11/lab11.ipynb — Off-policy evaluation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 11 — Off-Policy Evaluation: DM, IPS, SNIPS, DR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch11/lab11.ipynb)

**Companion lab to Chapter 11.** Evaluate a candidate dispatch policy from historical data alone — no deployment. Implement DM, IPS, SNIPS, DR. Demonstrate the doubly-robust property on both sides. Extend to sequential OPE on the Lab 8 PM scenario."""),

md("""## What you'll do

1. **Build the §11.9 dispatch SCM** with a fast-tool/complexity interaction.
2. **Implement all four estimators** from scratch: DM, IPS, SNIPS, DR.
3. **Demonstrate the DR doubly-robust property on both sides**: misspecified Q with correct $\\pi_b$, then misspecified $\\pi_b$ with correct Q. DR rescues both.
4. **Variance reduction**: apply weight clipping and the switch estimator to a low-overlap target policy.
5. **Sequential OPE** on the Lab 7 no-cost PM scenario via per-decision IS and DR-per-step.
6. **Effective sample size** as the positivity diagnostic."""),

md("""## Setup"""),

code("""# Colab: install Open Bandit Pipeline (obp) — the standard OPE benchmark
# library (Saito et al. 2021). Also DoWhy if you want to compare via DR.
%pip install --quiet obp 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — Single-step dispatch SCM (§11.9)

- $X \\sim \\mathcal N(0, 1)$ — lot complexity score.
- Behavior $\\pi_b(A=1 \\mid X) = \\sigma(X)$ — complex lots go to fast tool more often.
- $R = 1.0\\,A + 0.5\\,X - 0.3\\,A\\,X + \\varepsilon_R$ — fast tool gives baseline lift but is worse for complex lots due to the interaction.

Target $\\pi_e(A=1 \\mid X) = \\mathbf 1[X > 0]$ — deterministic threshold. Sounds sensible but is wrong: fast tool's interaction penalty exceeds its baseline benefit for $X \\gtrsim 1$."""),

code("""n = 5000

def pi_b_prob(x): return 1 / (1 + np.exp(-x))
def pi_e_act(x):  return (x > 0).astype(int)
def reward(x, a, noise):
    return 1.0 * a + 0.5 * x - 0.3 * a * x + noise

x   = rng.normal(0, 1, n)
a   = rng.binomial(1, pi_b_prob(x))
eps = rng.normal(0, 0.3, n)
r   = reward(x, a, eps)

# Oracle truth
a_tgt    = pi_e_act(x)
eps_tgt  = rng.normal(0, 0.3, n)
V_true   = reward(x, a_tgt, eps_tgt).mean()
print(f"True target-policy value: V_true = {V_true:+.3f}")"""),

md("""## Part 2 — DM (Direct Method)

Fit $\\hat Q(x, a) = E[R \\mid X = x, A = a]$ on the data, average over the observed $X$:

$$\\hat V_{\\text{DM}} = \\frac{1}{n}\\sum_i \\hat Q(X_i, \\pi_e(X_i)).$$"""),

code("""features = np.column_stack([x, a, x * a])
Q_model = LinearRegression().fit(features, r)
def Q_hat(x_v, a_v):
    return Q_model.predict(np.column_stack([x_v, a_v, x_v * a_v]))

V_DM = Q_hat(x, a_tgt).mean()
print(f"DM (correctly specified Q): {V_DM:+.3f}")"""),

md("""## Part 3 — IPS and SNIPS

Importance weights $\\rho_i = \\pi_e(A_i | X_i) / \\pi_b(A_i | X_i)$.

- IPS: $\\hat V = (1/n) \\sum_i \\rho_i R_i$.
- SNIPS: $\\hat V = \\sum_i \\rho_i R_i / \\sum_i \\rho_i$ (self-normalized; biased but lower variance)."""),

code("""# Importance weights
pi_b_a = np.where(a == 1, pi_b_prob(x), 1 - pi_b_prob(x))
pi_e_a = (a == a_tgt).astype(float)    # 1 if observed action matches target, else 0
rho    = pi_e_a / pi_b_a

V_IPS   = (rho * r).mean()
V_SNIPS = (rho * r).sum() / rho.sum()
print(f"IPS:                       {V_IPS:+.3f}")
print(f"SNIPS:                     {V_SNIPS:+.3f}")"""),

md("""## Part 4 — DR (Doubly Robust)

$$\\hat V_{\\text{DR}} = \\frac{1}{n}\\sum_i \\left[\\hat Q(X_i, \\pi_e(X_i)) + \\rho_i (R_i - \\hat Q(X_i, A_i))\\right].$$

The first term is DM. The second is an IPS-weighted residual correction. Consistent if either $\\hat Q$ or $\\pi_b$ is correct."""),

code("""V_DR = Q_hat(x, a_tgt).mean() + (rho * (r - Q_hat(x, a))).mean()
print(f"DR (both correct):         {V_DR:+.3f}")"""),

md("""All four agree closely with the truth when both nuisances are correct. The interesting test is what happens when one is wrong."""),

md("""## Part 5 — The doubly-robust property (both sides)

**Side A: Misspecify $\\hat Q$ (drop the interaction), keep $\\pi_b$ correct.** DR should still recover the truth via the IPS-weighted residual correction."""),

code("""Q_bad = LinearRegression().fit(np.column_stack([x, a]), r)
def Q_hat_bad(x_v, a_v): return Q_bad.predict(np.column_stack([x_v, a_v]))

V_DM_bad   = Q_hat_bad(x, a_tgt).mean()
V_DR_bad_Q = Q_hat_bad(x, a_tgt).mean() + (rho * (r - Q_hat_bad(x, a))).mean()
print(f"DM with misspecified Q:  {V_DM_bad:+.3f}  (biased)")
print(f"DR with misspecified Q:  {V_DR_bad_Q:+.3f}  (rescued by correct pi_b)")
print(f"V_true:                  {V_true:+.3f}")"""),

md("""**Side B: Keep $\\hat Q$ correct, misspecify $\\pi_b$ (assume constant 0.5).** DR should still recover the truth via the Q-based correction."""),

code("""pi_b_a_wrong = np.full(n, 0.5)
rho_wrong    = pi_e_a / pi_b_a_wrong
V_IPS_wrong  = (rho_wrong * r).mean()
V_DR_wrong_b = Q_hat(x, a_tgt).mean() + (rho_wrong * (r - Q_hat(x, a))).mean()
print(f"IPS with misspecified pi_b:  {V_IPS_wrong:+.3f}  (biased)")
print(f"DR with misspecified pi_b:   {V_DR_wrong_b:+.3f}  (rescued by correct Q)")
print(f"V_true:                      {V_true:+.3f}")"""),

md("""Both sides of the doubly-robust property are now empirically demonstrated. DR rescues misspecification in either nuisance."""),

md("""## Part 6 — Variance reduction with weight clipping

Replace the target with an aggressive threshold $\\pi_e(A=1 | X) = \\mathbf 1[X > -2]$ — almost everyone goes to the fast tool. Now the importance weights are extreme and IPS variance blows up."""),

code("""def pi_e_aggressive(x): return (x > -2).astype(int)

a_tgt_agg = pi_e_aggressive(x)
pi_e_a_agg = (a == a_tgt_agg).astype(float)
rho_agg = pi_e_a_agg / pi_b_a

# IPS variance vs weight clipping at different thresholds
for clip_pct in [None, 95, 99, 99.9]:
    if clip_pct is None:
        rho_clip = rho_agg
    else:
        thr = np.percentile(rho_agg, clip_pct)
        rho_clip = np.minimum(rho_agg, thr)
    V = (rho_clip * r).mean()
    var = (rho_clip * r).var()
    label = "no clip" if clip_pct is None else f"clip @ {clip_pct}th pct"
    print(f"{label:<20s} V = {V:+.4f}   var = {var:.4f}   ESS = {(rho_clip.sum())**2 / (rho_clip**2).sum():.0f}")"""),

md("""Weight clipping reduces variance at the cost of bias. The effective sample size (ESS) is the standard positivity diagnostic — when ESS is a small fraction of $n$, positivity is marginal."""),

md("""## Part 7 — Sequential OPE on the Lab 7 PM scenario

Take the 2-period PM scenario from Lab 7 (no PM cost; outcome is $Y = -2 L_1 + A_1 + A_2 + \\varepsilon$). Behavior policy: $A_t = \\sigma(L_t)$. Target policy: $A_t = \\mathbf 1[L_t > 0.5]$ (a threshold rule). Estimate the target-policy value from data.

(The cost-aware Lab 8 SCM is the natural next step — Exercise 1 walks through it. The no-cost version here keeps the trajectory-IS arithmetic simpler.)"""),

code("""# Regenerate Lab 7's PM data
n_seq = 5000
def step_L(prev_L, A, noise): return prev_L + 0.5 - 1.0 * A + noise
def yield_fn(L1, A1, A2, noise): return -2.0 * L1 + 1.0 * A1 + 1.0 * A2 + noise

noise_L1 = rng.normal(0, 0.3, n_seq)
noise_Y  = rng.normal(0, 0.5, n_seq)
L0 = rng.normal(0, 1.0, n_seq)
A1 = rng.binomial(1, 1 / (1 + np.exp(-L0)))
L1 = step_L(L0, A1, noise_L1)
A2 = rng.binomial(1, 1 / (1 + np.exp(-L1)))
Y  = yield_fn(L1, A1, A2, noise_Y)

# Target policy
def pi_e_seq(L_v): return (L_v > 0.5).astype(int)
a1_target = pi_e_seq(L0)
a2_target = pi_e_seq(L1)

# Behavior policy probabilities
p_A1_obs = np.where(A1 == 1, 1 / (1 + np.exp(-L0)), 1 - 1 / (1 + np.exp(-L0)))
p_A2_obs = np.where(A2 == 1, 1 / (1 + np.exp(-L1)), 1 - 1 / (1 + np.exp(-L1)))

# Trajectory IS weight
rho_traj = ((A1 == a1_target).astype(float) / p_A1_obs) * ((A2 == a2_target).astype(float) / p_A2_obs)

V_traj_IPS = (rho_traj * Y).mean()

# Oracle truth via simulation
L1_pol = step_L(L0, a1_target, noise_L1)
a2_pol = pi_e_seq(L1_pol)
Y_pol = yield_fn(L1_pol, a1_target, a2_pol, noise_Y)
V_seq_true = Y_pol.mean()

print(f"Trajectory IS estimate:  {V_traj_IPS:+.3f}")
print(f"Oracle (sim):             {V_seq_true:+.3f}")
print(f"ESS:                      {(rho_traj.sum())**2 / (rho_traj**2).sum():.0f} out of {n_seq}")"""),

md("""Trajectory IS recovers the truth with high variance — the ESS is much smaller than $n$ because the deterministic target only matches the behavior at a fraction of observations. The chapter's DR-per-step would tighten this; left as Exercise 1."""),

md("""## Part 8 — The production workflow: Open Bandit Pipeline

The chapter (§11.7) calls out the Open Bandit Pipeline (`obp`) as the standard benchmark for contextual-bandit OPE. The library implements all the estimators from Parts 2–4 plus newer ones (switch, MRDR, DR-OS, Continuous IPS) and gives you a single unified interface to run them all on the same data."""),

code("""# Reformat Part 1's data into the dict structure obp expects.
# obp's OffPolicyEvaluation runs all configured estimators in one call.
try:
    from obp.ope import (OffPolicyEvaluation, DirectMethod,
                         InverseProbabilityWeighting, SelfNormalizedInverseProbabilityWeighting,
                         DoublyRobust)
    from obp.dataset import logistic_reward_function

    # obp expects 2D action contexts, integer actions, etc.
    bandit_feedback = dict(
        n_rounds=n,
        n_actions=2,
        action=a.astype(int),
        position=None,
        reward=r,
        context=x.reshape(-1, 1),
        pscore=pi_b_a,  # P(observed action | context) under behavior
    )
    # The target policy as a (n_rounds, n_actions, 1) array of action probabilities.
    action_dist = np.zeros((n, 2, 1))
    action_dist[np.arange(n), a_tgt, 0] = 1.0
    # Q-hat from Part 2 (our DM model), shaped (n, n_actions, 1)
    estimated_rewards = np.zeros((n, 2, 1))
    estimated_rewards[:, 0, 0] = Q_hat(x, np.zeros(n))
    estimated_rewards[:, 1, 0] = Q_hat(x, np.ones(n))

    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[DirectMethod(), InverseProbabilityWeighting(),
                        SelfNormalizedInverseProbabilityWeighting(),
                        DoublyRobust()],
    )
    estimated_values = ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards,
    )
    print("Open Bandit Pipeline estimates:")
    for name, v in estimated_values.items():
        print(f"  {name:<8s}  {v:+.3f}")
    print(f"\\nOur manual estimates: V_DM={V_DM:+.3f}  V_IPS={V_IPS:+.3f}  "
          f"V_SNIPS={V_SNIPS:+.3f}  V_DR={V_DR:+.3f}")
    print(f"Oracle truth: V_true = {V_true:+.3f}")
except ImportError:
    print("Open Bandit Pipeline not installed. %pip install obp in Colab.")
except Exception as e:
    print(f"obp call failed: {type(e).__name__}: {e}")
    print("(The library's API can drift across versions; see obp docs for the current call signature.)")"""),

md("""The library gives you the same point estimates with one call instead of four separate manual implementations. The win is uniformity: switching to a switch estimator, MRDR, or any other obp estimator is a one-line change. The library also implements confidence-interval procedures (efficient-influence-function based) that are non-trivial to derive by hand."""),

md("""## Reflection

**The DR property is the practical workhorse.** DM is fast but bias-prone; IPS is unbiased but variance-prone; DR combines them with the best of both. The two demos in Part 5 are the canonical demonstration.

**Positivity diagnostics are mandatory.** Always compute ESS before reporting IPS or DR. An ESS below 30% of $n$ is a red flag.

**Sequential OPE has compound variance.** Per-decision IS and DR-per-step are the modern defaults. Raw trajectory IS works only for short horizons or when the policies agree most of the time."""),

md("""## Exercises

1. **DR-per-step.** Implement the Jiang-Li (2016) DR-per-step estimator on the 2-period PM scenario. Compare ESS-effective variance to trajectory IS.

   <details><summary>Solution</summary>

   ```python
   # DR-per-step: V_DR = sum_t rho_{1:t} * (r_t - q_hat_t) + sum_t rho_{1:t-1} * v_hat_t
   def dr_per_step(rewards, rhos, q_hats, v_hats):
       T = rewards.shape[1]
       rho_cum = np.cumprod(rhos, axis=1)
       rho_prev = np.concatenate([np.ones((len(rewards), 1)), rho_cum[:, :-1]], axis=1)
       return (rho_cum * (rewards - q_hats) + rho_prev * v_hats).sum(axis=1).mean()
   ```

   Compute $\\hat q_t(s_t, a_t)$ and $\\hat v_t(s_t) = E_{a \\sim \\pi_e}[\\hat q_t(s_t, a)]$ via Q-learning on the logged trajectories. DR-per-step uses *cumulative* importance weights through step $t-1$ only, not the full trajectory product, so its variance scales linearly with horizon instead of exponentially. On the 2-period PM scenario, expect roughly a 2-3x ESS improvement over trajectory IS.
   </details>

2. **Switch estimator.** Implement the Wang-Agarwal-Dudík (2017) switch estimator: use DR when $\\rho_i$ is moderate, DM when it's extreme. Sweep the switch threshold and find the best bias-variance tradeoff.

   <details><summary>Solution</summary>

   ```python
   def switch(actions_e, q_hat, rhos, rewards, tau):
       mask = rhos <= tau
       dr_part = mask * (rhos * (rewards - q_hat[np.arange(len(actions_e)), actions_e]) + q_hat.mean(axis=1))
       dm_part = (1 - mask) * q_hat.mean(axis=1)
       return (dr_part + dm_part).mean()

   for tau in [1, 2, 5, 10, 20, np.inf]:
       print(tau, switch(..., tau))
   ```

   Small $\\tau$ → behaves like DM (low variance, model-bias dominated). Large $\\tau$ → behaves like DR (low bias, weight-variance dominated). The sweet spot is usually $\\tau \\in [3, 10]$; pick it by minimising bootstrap MSE on a held-out slice.
   </details>

3. **Real bandit data.** Apply IPS, SNIPS, DR to the OpenBanditDataset via `obp`. Compare to a randomized-controlled-trial baseline.

   <details><summary>Solution</summary>

   ```python
   from obp.dataset import OpenBanditDataset
   from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting, SelfNormalizedInverseProbabilityWeighting, DoublyRobust

   ds = OpenBanditDataset(behavior_policy="random", campaign="all")
   bandit = ds.obtain_batch_bandit_feedback()
   q_hat = ...  # fit a RegressionModel on the logged context-action-reward triples
   action_dist = ...  # define the evaluation policy
   ope = OffPolicyEvaluation(bandit_feedback=bandit,
       ope_estimators=[InverseProbabilityWeighting(), SelfNormalizedInverseProbabilityWeighting(), DoublyRobust()])
   print(ope.estimate_policy_values(action_dist=action_dist, estimated_rewards_by_reg_model=q_hat))
   ```

   The "random" log gives unbiased true policy values via Monte Carlo — compare your OPE estimates to that. The "bts" (Bernoulli Thompson Sampling) log lets you test OPE under a non-uniform behaviour policy. Expect SNIPS to be much closer to truth than raw IPS when ESS is low.
   </details>

4. **Pre-deployment validation.** Use the four estimators in disagreement-as-diagnostic mode: when DM and DR disagree, what does that tell you? Run the analysis on the Lab 5 dataset with a candidate threshold policy.

   <details><summary>Solution</summary>

   Compute DM, IPS, SNIPS, DR on the same logged data with the same target policy. If **DM and DR agree** within Monte Carlo error → the outcome model is well-specified and the answer is trustworthy. If **DM and DR disagree** but DR and SNIPS agree → the outcome model is biased; trust the weight-based estimators (subject to ESS check). If **DR and SNIPS disagree** → weight clipping or trimming is doing significant work; investigate positivity. If **all four disagree** → either the target policy strays too far from the behaviour policy (low ESS, even after clipping) or the data is too small. Don't deploy; collect more data or randomise a small exploration arm.
   </details>"""),

md("""## What's next

Lab 12 closes the loop on policy learning: causal RL where the *target* policy is itself being trained from a combination of observational data, twin rollouts, and (where allowed) online interaction."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch11" / "lab11.ipynb", cells)
print("Built lab11.ipynb")
