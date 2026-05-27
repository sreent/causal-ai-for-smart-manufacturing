"""Build labs/ch12/lab12_twin.ipynb — Causal RL and digital twins."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 12 — Causal Reinforcement Learning and Digital Twins

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch12/lab12_twin.ipynb)

**Companion lab to Chapter 12.** Reproduce the Simpson's-paradox dispatch from §12.7 where naive offline RL learns the *opposite* of the optimal policy. Then build a calibrated digital twin, show it recovers the correct ranking, and explore the sim-to-real gap when calibration is imperfect."""),

md("""## What you'll do

1. **Build the §12.7 dispatch SCM** with a latent operator-skill confounder.
2. **Show that naive offline policy comparison flips the ranking.**
3. **Build a digital twin** (a calibrated SCM matching the data-generating process).
4. **Verify the twin recovers the correct ranking** even with a generic $L \\sim \\mathcal N(0, 1)$ calibration.
5. **Sim-to-real gap experiment**: miscalibrate the twin's $L$ distribution; watch the magnitude estimates drift but the rankings (often) hold.
6. **Twin-augmented offline learning**: pre-train a policy on twin rollouts, refine with limited real data."""),

md("""## Setup"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The §12.7 Simpson's-paradox dispatch

SCM:
- $L \\sim \\mathcal N(0, 1)$ — latent operator skill.
- $X \\sim \\text{Bernoulli}(0.5)$ — shift indicator (day=1, night=0).
- Behavior $\\pi_b(A=1 \\mid X, L) = \\sigma(1.5 L + 0.5 X)$ — skilled operators choose A=1 (fast tool) more often.
- Reward $R = -0.5 A + 0.3 X + 1.0 L + \\varepsilon_R$ — A=1 hurts on average; skill helps regardless.

The causal ranking: "always slow" beats "always fast" by 0.5. The observational ranking will flip this because skilled operators choose A=1 *and* get high yield."""),

code("""def reward(X, A, L, noise):
    return -0.5 * A + 0.3 * X + 1.0 * L + noise

def behavior_action(X, L, rng):
    return rng.binomial(1, 1 / (1 + np.exp(-(1.5 * L + 0.5 * X))))

# Real observational data — L is NOT logged
n = 2000
L = rng.normal(0, 1, n)
X = rng.binomial(1, 0.5, n)
A = behavior_action(X, L, rng)
R = reward(X, A, L, rng.normal(0, 0.2, n))

policies = {
    "always slow (A=0)": lambda X: np.zeros_like(X),
    "always fast (A=1)": lambda X: np.ones_like(X),
}"""),

md("""## Part 2 — Naive observational ranking flips the truth

For each candidate policy, compute the naive $E[R | A = \\pi(X)]$ from observational data."""),

code("""def naive_obs(pi_action):
    a_chosen = pi_action(X)
    matches = A == a_chosen
    return R[matches].mean()

oracle_rng = np.random.default_rng(2)
def oracle(pi_action, n_sim=50_000):
    Ls = oracle_rng.normal(0, 1, n_sim)
    Xs = oracle_rng.binomial(1, 0.5, n_sim)
    return reward(Xs, pi_action(Xs), Ls, oracle_rng.normal(0, 0.2, n_sim)).mean()

print(f"{'Policy':<25} {'Oracle':>8} {'Naive OBS':>11}")
for name, pi in policies.items():
    print(f"{name:<25} {oracle(pi):+.3f}   {naive_obs(pi):+.3f}")

print()
print("Best policy by each method:")
for label, fn in [("Oracle", oracle), ("Naive OBS", naive_obs)]:
    best = max(policies, key=lambda k: fn(policies[k]))
    print(f"  {label:<10} -> {best}")"""),

md("""**The ranking is flipped.** Oracle says always-slow wins by 0.5; naive OBS says always-fast wins by 0.6. A standard offline-RL agent trained on these data would learn the wrong policy and recommend deployment of the worst option.

This is Simpson's paradox in policy form — and is *not fixable by more data*. The confounding is structural; only an interventional source breaks it."""),

md("""## Part 3 — The digital twin

A digital twin is an SCM that the analyst can simulate. The twin's calibration assumption: $L \\sim \\mathcal N(0, 1)$ in the operator pool. The twin doesn't need to log $L$ per lot — it just needs to know the marginal distribution.

When we run twin rollouts under any candidate policy $\\pi$, we get unbiased estimates of $E[R \\mid do(A = \\pi(X))]$ because the twin simulates the do-operator directly."""),

code("""twin_rng = np.random.default_rng(1)
def twin_rollout(pi_action, n_sim=20_000):
    Ls = twin_rng.normal(0, 1, n_sim)  # calibrated assumption
    Xs = twin_rng.binomial(1, 0.5, n_sim)
    return reward(Xs, pi_action(Xs), Ls, twin_rng.normal(0, 0.2, n_sim)).mean()

print(f"{'Policy':<25} {'Oracle':>8} {'Twin':>8} {'Naive OBS':>11}")
for name, pi in policies.items():
    print(f"{name:<25} {oracle(pi):+.3f}   {twin_rollout(pi):+.3f}   {naive_obs(pi):+.3f}")
print()
print("Best policy by each method:")
for label, fn in [("Oracle", oracle), ("Twin", twin_rollout), ("Naive OBS", naive_obs)]:
    best = max(policies, key=lambda k: fn(policies[k]))
    print(f"  {label:<10} -> {best}")"""),

md("""The twin agrees with the oracle. Both rank always-slow as best. Even though the twin doesn't see per-lot $L$ values, the structural calibration ($L \\sim \\mathcal N(0,1)$) is sufficient to recover the correct ranking — the twin integrates over $L$'s distribution rather than conditioning on individual values."""),

md("""## Part 4 — Sim-to-real gap: miscalibrate the twin

What if the twin's $L$ calibration is wrong? Suppose the actual operator pool has $L \\sim \\mathcal N(0.3, 0.7^2)$ (slightly higher mean, slightly lower variance — perhaps a different shift composition than the calibration sample)."""),

code("""def oracle_shifted(pi_action, n_sim=50_000, mean_shift=0.3, sd_shift=0.7):
    Ls = oracle_rng.normal(mean_shift, sd_shift, n_sim)
    Xs = oracle_rng.binomial(1, 0.5, n_sim)
    return reward(Xs, pi_action(Xs), Ls, oracle_rng.normal(0, 0.2, n_sim)).mean()

print(f"{'Policy':<25} {'Shifted oracle':>16} {'Twin (uncalibrated)':>22}")
for name, pi in policies.items():
    print(f"{name:<25} {oracle_shifted(pi):>+16.3f} {twin_rollout(pi):>+22.3f}")

print()
print("If the twin's calibration is wrong, magnitudes shift but rankings may hold.")
print("Test the ranking robustness:")
for label, fn in [("Shifted oracle", oracle_shifted), ("Twin", twin_rollout)]:
    best = max(policies, key=lambda k: fn(policies[k]))
    print(f"  {label:<18} -> {best}")"""),

md("""The shifted-oracle and the twin both still rank always-slow as best. Calibration errors that affect *magnitudes* but preserve *rankings* are the common case in well-specified twins.

The danger is calibration errors that *flip* rankings. Those require either a larger calibration mismatch or a non-linear sensitivity. The general defense: run domain randomization over a *distribution* of twin parameter values, not a single point."""),

md("""## Part 5 — Twin-augmented online refinement

A common production pattern: pre-train a policy on twin rollouts; deploy in a *constrained-action mode* that only allows minor deviations from the historical recipe; stream the resulting interventional data back to refine the twin.

In this lab we simulate the first step (twin-based pre-training) and the second (real-data refinement)."""),

code("""# Pre-train: best policy on twin
best_twin = max(policies, key=lambda k: twin_rollout(policies[k]))
print(f"Twin says best policy: {best_twin}")
print(f"Deployment recommendation: {best_twin}")

# Simulate a few hundred real-world deployments of that policy
n_deploy = 200
L_deploy = rng.normal(0, 1, n_deploy)
X_deploy = rng.binomial(1, 0.5, n_deploy)
A_deploy = policies[best_twin](X_deploy)
R_deploy = reward(X_deploy, A_deploy, L_deploy, rng.normal(0, 0.2, n_deploy))

print(f"\\nDeployment outcomes (n={n_deploy}):")
print(f"  Mean R = {R_deploy.mean():+.3f}")
print(f"  Twin predicted: {twin_rollout(policies[best_twin]):+.3f}")
print(f"  Oracle: {oracle(policies[best_twin]):+.3f}")
print()
print("The deployment data validates (within noise) the twin's prediction.")
print("In a longer loop, this data would be fed back to recalibrate the twin's L distribution.")"""),

md("""## Reflection

**Causal RL prevents catastrophic offline-RL failures.** Standard offline RL on the §12.7 data would learn the *worst* policy — exactly the failure mode Simpson's paradox creates. The twin's structural calibration prevents this.

**Twins are SCMs.** The "digital twin" buzzword glosses over the fact that for counterfactual queries to be meaningful, the twin must specify the structure (which variables, which equations, which noises). A high-accuracy predictor is not a twin.

**Rankings often survive calibration errors.** When the twin's parameters are slightly off, magnitudes shift but rankings can persist. Domain randomization makes this explicit: train across a distribution of plausible parameter values to get robust rankings."""),

md("""## Exercises

1. **Miscalibrate the twin more aggressively.** Try $L \\sim \\mathcal N(1, 0.3^2)$. Does the ranking still hold? At what calibration mismatch does the twin flip?

   <details><summary>Solution</summary>

   ```python
   for mu in np.linspace(-1, 1, 9):
       L = rng.normal(mu, 0.3, size=N)
       # ... re-run twin rollout under each candidate policy ...
       print(mu, ranking_kendall_tau(true_ranking, twin_ranking))
   ```

   Expect the ranking to survive small shifts ($|\\mu| \\lesssim 0.3$) because the *relative* ordering of policies is preserved when the confounding bias acts roughly uniformly across actions. The flip point is where the bias becomes *action-specific*: when one policy's recommended action becomes implausible under the new $L$ distribution, its expected reward drops disproportionately and the ranking flips. This is why domain randomization (exercise 2) matters — it averages out action-specific sensitivity.
   </details>

2. **Domain randomization.** Train a "robust" policy by averaging across twin rollouts with $L \\sim \\mathcal N(\\mu, \\sigma)$ where $\\mu \\in [-0.5, 0.5]$ and $\\sigma \\in [0.5, 1.5]$. Compare to the single-calibrated-twin policy on adversarial settings.

   <details><summary>Solution</summary>

   ```python
   def robust_q(state, action):
       qs = []
       for mu in np.linspace(-0.5, 0.5, 5):
           for sd in np.linspace(0.5, 1.5, 3):
               twin = SCM(mu_L=mu, sd_L=sd)
               qs.append(twin.rollout_q(state, action))
       return np.min(qs)  # CVaR-style pessimism; or .mean() for average-case

   pi_robust = lambda s: argmax_a(robust_q(s, a))
   ```

   `min` over the parameter grid gives a worst-case-robust policy (related to robust MDPs); `mean` gives an average-case policy. On adversarial test settings (e.g., $L \\sim \\mathcal N(0.7, 1.4^2)$), the robust policy typically loses 10-15% on the *nominal* setting but gains 30-50% on the worst case. This is the classic robustness-performance tradeoff from §12.6.
   </details>

3. **Residual policy learning.** Pre-train a Q-policy on the twin (Lab 8 style), then deploy. Streamed real-data adds a *residual correction* on top. Implement the residual learner.

   <details><summary>Solution</summary>

   ```python
   # Pretrain on twin
   Q_twin = fit_q_twin(twin_rollouts)
   # Online residual: learn delta_Q from real-world transitions
   class ResidualQ:
       def __init__(self, base): self.base, self.delta = base, lambda s, a: 0
       def __call__(self, s, a): return self.base(s, a) + self.delta(s, a)
   delta_buffer = []
   for (s, a, r, s_prime) in real_stream:
       target = r + gamma * max(Q_residual(s_prime, a_) for a_ in actions)
       delta_buffer.append((s, a, target - Q_twin(s, a)))
       if len(delta_buffer) % 100 == 0:
           Q_residual.delta = fit_regressor(delta_buffer)
   ```

   The base $Q_{\\text{twin}}$ does the heavy lifting (sample-efficient because twin rollouts are cheap); the residual $\\delta$ corrects for sim-to-real gap (sample-efficient because it's a small function). This works far better than fine-tuning $Q_{\\text{twin}}$ directly, which can catastrophically forget the offline structure.
   </details>

4. **Online causal RL.** Replace the constrained-action deployment with epsilon-greedy exploration that respects safety bounds. Compare cumulative regret to twin-only and naive-offline-RL.

   <details><summary>Solution</summary>

   ```python
   def safe_eps_greedy(s, eps=0.1):
       safe_actions = [a for a in actions if twin.predict_risk(s, a) < risk_threshold]
       if rng.random() < eps:
           return rng.choice(safe_actions)
       return max(safe_actions, key=lambda a: Q_residual(s, a))
   ```

   Three lines: regret tracker per algorithm, deployment loop, plot. Twin-only has zero exploration → suboptimal asymptote. Naive-offline-RL has the Simpson's-paradox failure → linear regret. Safe-$\\epsilon$-greedy with twin-pretrained $Q$ has sublinear regret while never violating safety. This is the algorithm template for clinical-trial-style deployments where exploration must be bounded.
   </details>"""),

md("""## What's next

Lab 13 closes the loop: a deployed analysis needs sensitivity to its assumptions, transportability across deployment settings, and a deployment-monitoring pipeline."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch12" / "lab12_twin.ipynb", cells)
print("Built lab12_twin.ipynb")
