"""Build labs/ch12/lab12b.ipynb — fitting a digital twin of TE and validating against the simulator."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 12B — Fitting and Validating a Digital Twin of the Tennessee Eastman Plant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch12/lab12b.ipynb)

**Companion to Lab 12A.** Lab 12A built a twin of a synthetic two-state MDP, used it to plan a policy, and verified that the twin-planned policy outperformed an observational-RL baseline that suffered from confounding. **Lab 12B does the same exercise on the Tennessee Eastman simulator — but the twin we fit is *learned* (a linear structural model from logged data), not the true Fortran simulator.**

The deliverable is a *calibrated learned twin*, a *twin-based prediction* of an alternative policy's outcome, and an honest report of the *sim-to-real gap* between the twin's prediction and the simulator's reality. This is the workflow every team running twin-augmented RL in production must execute before they trust the twin enough to plan a deployment with it."""),

md("""## What this lab is *not* doing

- **Running a full RL algorithm.** We compare a small number of *constant-action* policies (a = 0 always vs a = 1 always) on the learned twin. Lab 12A's Q-learning machinery generalises here, but adding it would obscure the twin-validation point.
- **Building a high-fidelity twin.** A linear next-state model is the simplest structural-twin form: $X_{t+1} = A X_t + B a_t + \\varepsilon_t$. TE is nonlinear; the twin's residual error is partly the sim-to-real gap, partly the model's parametric limit.
- **Confounded-MDP demo.** Our logged behaviour policy is *random* (Bernoulli(0.5) on IDV(1)); there's no confounding by past policy choices the way Ch 12 §12.7 has. The lab focuses on the twin-validation workflow; confounded-MDP recovery is studied in Lab 12A."""),

code("""%pip install -q numpy pandas matplotlib scikit-learn"""),

code("""import os, sys, urllib.request, pathlib

PREP_PATH = pathlib.Path("/content/te_prep.py")
if not PREP_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/te_prep.py",
        PREP_PATH,
    )
sys.path.insert(0, str(PREP_PATH.parent))

for name in ("te_logged.csv", "te_candidate.csv"):
    p = pathlib.Path("/content") / name
    if not p.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/{name}",
            p,
        )

from te_prep import load_te

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load logged + candidate trajectories

`logged` is our *training data* for the twin (random-action behaviour). `candidate` is the *interventional ground truth* we will validate the twin against."""),

code("""log  = load_te("logged")
cand = load_te("candidate")

state_cols = [c for c in log.columns if c.startswith("XMEAS")]
n_state = len(state_cols)

S_log = log[state_cols].values
a_log = log["action_idv1"].values

S_cand = cand[state_cols].values
a_cand = cand["action_idv1"].values   # all zeros

print(f"State dim:            {n_state}")
print(f"Logged trajectory:    {S_log.shape}  action rate {a_log.mean():.2f}")
print(f"Candidate trajectory: {S_cand.shape} action rate {a_cand.mean():.2f}")"""),

md("""## Part 2 — Fit a linear structural twin

The twin's structural equations are:

$$X_{t+1} = A \\, X_t + B \\, a_t + \\varepsilon_t$$

with $A \\in \\mathbb{R}^{n \\times n}$, $B \\in \\mathbb{R}^{n}$, and i.i.d. residuals $\\varepsilon_t$. Fitting reduces to one OLS per state component (or equivalently a single multi-output linear regression). This is the *simplest possible* structural twin; the chapter's §12.5 generalises to nonlinear $f$, GP residual, or hybrid physics-plus-correction forms."""),

code("""# Build (X_t, a_t) -> X_{t+1} pairs from the logged trajectory.
X_t   = S_log[:-1]
a_t   = a_log[:-1].reshape(-1, 1)
X_t1  = S_log[1:]
inputs = np.hstack([X_t, a_t])

twin = LinearRegression()
twin.fit(inputs, X_t1)

# Twin parameters: A is the leading n x n block of coef, B is the trailing n x 1 column.
A_hat = twin.coef_[:, :n_state]
B_hat = twin.coef_[:, n_state:n_state+1].ravel()
bias_hat = twin.intercept_

train_pred = twin.predict(inputs)
train_resid = X_t1 - train_pred
train_rmse = float(np.sqrt(np.mean(train_resid ** 2)))

print(f"Twin training RMSE (state-averaged):  {train_rmse:.3f}")
print(f"|A| spectral radius:                  {np.linalg.norm(A_hat, ord=2):.3f}")
print(f"|B| (action sensitivity, L2):         {np.linalg.norm(B_hat):.3f}")"""),

md("""## Part 3 — Validate the twin: roll forward under the candidate's actions

The twin's calibration test is *not* its training RMSE. It is: *given the candidate trajectory's initial state and the candidate's action sequence, does the twin reproduce the candidate trajectory?*

Concretely: starting from `S_cand[0]`, apply $X_{t+1} = A X_t + B a_t + \\text{bias}$ with $a_t = 0$ for all $t$ (the candidate policy). Compare the twin's rollout to the actual candidate trajectory."""),

code("""def twin_rollout(S0, actions, A, B, bias):
    n = len(actions)
    X = np.zeros((n + 1, len(S0)))
    X[0] = S0
    for t in range(n):
        X[t + 1] = A @ X[t] + B * actions[t] + bias
    return X[1:]  # match S_cand shape (drop the initial state)

S_twin = twin_rollout(S_cand[0], a_cand[:-1], A_hat, B_hat, bias_hat)
# Compare element-wise on the overlapping window.
resid_validation = S_cand[1:] - S_twin
rmse_per_state = np.sqrt(np.mean(resid_validation ** 2, axis=0))

print(f"Validation RMSE (state-averaged):      {float(rmse_per_state.mean()):.3f}")
print(f"Worst-fit states (top 5):")
order = np.argsort(rmse_per_state)[::-1]
for i in order[:5]:
    print(f"  {state_cols[i]:<12}  RMSE = {rmse_per_state[i]:.3f}")"""),

md("""**Why validation RMSE differs from training RMSE.** Training error is a one-step-ahead residual; validation here is a *multi-step rollout* whose errors compound. A modest one-step error can grow over 500 steps if the twin's $A$ matrix has eigenvalues close to or above 1 (unstable / weakly damped). The chapter's §12.5 calls this the twin's *stability* property — distinct from its single-step calibration."""),

md("""## Part 4 — Use the twin to predict policy returns

We define the reward (same as Lab 11B):

$$r_t = -(XMEAS(7)_t - 2700)^2 / 10^6$$

— negative squared deviation of reactor pressure from the operating-point target.

Then we use the twin to predict the expected per-step reward under two constant-action policies (a = 0 forever, a = 1 forever) and compare to the simulator's actual reward on the candidate trajectory."""),

code("""pressure_idx = state_cols.index("XMEAS(7)")

def reward_from_states(states, pressure_target=2700.0):
    return -((states[:, pressure_idx] - pressure_target) ** 2) / 1e6

# Twin-predicted trajectory under a = 0 (matches candidate policy).
S_twin_off = twin_rollout(S_cand[0], np.zeros(len(a_cand) - 1), A_hat, B_hat, bias_hat)
r_twin_off = reward_from_states(S_twin_off)

# Twin-predicted trajectory under a = 1.
S_twin_on  = twin_rollout(S_cand[0], np.ones(len(a_cand) - 1),  A_hat, B_hat, bias_hat)
r_twin_on  = reward_from_states(S_twin_on)

# Simulator ground truth for a = 0 (the candidate CSV).
r_sim_off  = reward_from_states(S_cand[1:])

print(f"Twin-predicted V (a = 0 always):  {r_twin_off.mean():+.4f}")
print(f"Simulator actual V (a = 0 always): {r_sim_off.mean():+.4f}")
print(f"Sim-to-real gap (twin - sim):     {r_twin_off.mean() - r_sim_off.mean():+.4f}")
print()
print(f"Twin-predicted V (a = 1 always):  {r_twin_on.mean():+.4f}")
print(f"Twin-predicted ranking: {'a=1 > a=0' if r_twin_on.mean() > r_twin_off.mean() else 'a=0 > a=1'}")"""),

md("""## Part 5 — Diagnostic plot: twin rollout vs simulator

A picture of the reactor-pressure trace under the candidate policy, from both the twin and the simulator. Any divergence visualises the sim-to-real gap directly."""),

code("""fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(S_cand[1:, pressure_idx], label="Simulator (truth)", alpha=0.8)
ax.plot(S_twin_off[:, pressure_idx], label="Twin rollout (a=0)", alpha=0.8)
ax.axhline(2700, color="grey", linestyle="--", linewidth=0.7, label="Reward target")
ax.set_xlabel("Sample step (3 min each)")
ax.set_ylabel("XMEAS(7) Reactor Pressure (kPa)")
ax.set_title("Twin vs simulator on the candidate policy (a = 0 always)")
ax.legend()
plt.tight_layout()
plt.show()"""),

md("""## Part 6 — When the twin is wrong, *how* is it wrong?

Three diagnostic questions to read the twin against:

1. **Bias.** Does the twin systematically predict reactor pressure above (or below) the simulator?
2. **Variance.** Does the twin's trace under-shoot or over-shoot the simulator's natural fluctuations?
3. **Long-horizon drift.** Does the twin-vs-sim gap grow over the 500-step horizon? If so, the twin's $A$ matrix is propagating errors unstably."""),

code("""mean_gap = float((S_twin_off[:, pressure_idx] - S_cand[1:, pressure_idx]).mean())
std_twin = float(S_twin_off[:, pressure_idx].std())
std_sim  = float(S_cand[1:, pressure_idx].std())

early_gap = float(np.abs(S_twin_off[:50,   pressure_idx] - S_cand[1:51,   pressure_idx]).mean())
late_gap  = float(np.abs(S_twin_off[-50:,  pressure_idx] - S_cand[-50:,   pressure_idx]).mean())

print(f"Mean bias (twin pressure - sim pressure):    {mean_gap:+.2f} kPa")
print(f"Twin trace std:  {std_twin:.2f}   |   Sim trace std: {std_sim:.2f}")
print(f"  -> ratio twin/sim std: {std_twin / max(std_sim, 1e-9):.2f}  (1.0 = matched variance)")
print()
print(f"Mean |gap| in first 50 steps:  {early_gap:.2f} kPa")
print(f"Mean |gap| in last 50 steps:   {late_gap:.2f} kPa")
print(f"  -> drift factor (late/early): {late_gap / max(early_gap, 1e-9):.2f}  (>>1 = unstable rollout)")"""),

md("""## Part 7 — Decision

Three bullets, the deliverable a process-control team would read before trusting the twin for policy planning:

1. **The learned twin's policy ranking** (read Part 4: which constant-action policy does the twin think is better) is *the recommendation the twin makes*. The simulator's actual value on the candidate (a = 0) is *what really happens*. The gap between the two is the sim-to-real error budget you must accept if you act on the twin.

2. **The twin is calibrated / fragile** based on the bias, variance, and drift diagnostics from Part 6. A bias > 5% of the target value, a variance ratio outside [0.5, 2.0], or a drift factor > 2.0 is the signal to refit the twin with a richer parametric form (nonlinear, hybrid, GP-residual) before using it for planning.

3. **A linear twin is a baseline, not a final twin.** For a real deployment in chemical-process control, a structural twin should encode the known mass-balance and energy-balance equations as priors, with the data used to identify *only the residual* between physics and observation. The chapter's §12.5 develops this hybrid form."""),

md("""## Reflection

**A twin that fits training data well can still misrank policies.** The chapter's pitfall #1 (§12.8) names this directly: *predictive accuracy is observational; counterfactual accuracy is structural*. Our linear twin happens to have a structural form (a linear SCM), but its calibration only tested it on the action distribution that produced the logged data. Whether it predicts correctly under a *different* action distribution is the question Part 4 actually answers — and that is the question that matters for deployment.

**The validation step is the entire point.** Before a twin is used to plan, it must be *interventionally* validated — its rollout under a policy that differs from the logged behaviour must match real observations under that same policy. Without this step the twin is, in industrial parlance, *not a twin* — it is a predictor."""),

md("""## What's next

Lab 13B closes the SECOM Lab B arc with transportability — *given an estimate from one period of data, does it generalise to the next?* That is the deployment-readiness question every causal analysis ultimately has to answer."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch12" / "lab12b.ipynb", cells)
print("Built lab12b.ipynb")
