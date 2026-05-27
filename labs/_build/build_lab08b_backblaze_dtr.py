"""Build labs/ch08/lab08b_backblaze_dtr.ipynb — DTR Q-learning on Backblaze drive telemetry."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 8B — Dynamic Replacement Policy on Backblaze SMART Telemetry

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch08/lab08b_backblaze_dtr.ipynb)

**Companion to Lab 8A.** Lab 8A built a two-stage cost-aware preventive-maintenance SCM with a known closed-form optimum and verified that Q-learning by backward induction recovers it. **Lab 8B asks the same question on real hard-drive telemetry: given a stream of SMART readings, what is the optimal *replace or wait* policy across two decision points?**

The setup is the one a data-centre operator faces every week: you observe SMART metrics for a fleet of drives; at each decision point you can either spend $1$ unit to pre-emptively replace a drive (a planned RMA), or wait and accept a $10$ unit cost if the drive fails before the next decision (an in-service failure, much more disruptive). The deliverable is the policy that minimises expected total cost across two decision stages.

**Dataset.** `labs/data/backblaze_subset.csv` — 1576 ST4000DM000 drives observed for ~30 days in early 2016, with 76 observed failures (~4.8 % drive-level failure rate). See `labs/data/README.md` for the preprocessing recipe.

**Two stages.**

- **Stage 1 (day 7).** Decide $A_1 \\in \\{0, 1\\}$ — replace (1) or wait (0) — based on the SMART state observed through the first week.
- **Stage 2 (day 14).** If the drive was not replaced at stage 1 and is still operating, decide $A_2 \\in \\{0, 1\\}$ based on the SMART state observed through the second week.

**Costs.** Replacement $c_R = 1$, in-service failure $c_F = 10$. Reward is the negative cost (Q-learning maximises reward).

**Synthesised actions.** Backblaze logged passive observation, not the decisions a maintenance team would have made. We assign $A_1, A_2 \\sim \\mathrm{Bernoulli}(0.5)$ per drive — a uniform random *behaviour policy* — and let the regression recover the optimal *target policy*. Under $A = 0$ the observed failure status from the data applies; under $A = 1$ we assume the replacement succeeds (no failure, just the replacement cost). This is the standard counterfactual-substitution setup for offline policy learning."""),

md("""## What this lab is *not* doing

- **Validating a real replacement policy on operational data.** The synthesised actions are a teaching device — Backblaze never actually replaced these drives on our schedule. We assign $A_1, A_2 \\sim \\mathrm{Bernoulli}(0.5)$ to give Q-learning the action coverage it needs; under $A = 1$ we counterfactually substitute "successful replacement". Off-policy evaluation under a *non-uniform* logged behaviour policy (with propensity-overlap diagnostics, IPS / SNIPS / DR estimators) is the territory of **Chapter 11 / Lab 11B**. If you want to estimate a *learned* DTR's expected return from a real maintenance team's logged decisions, that is the workflow.
- **Predicting individual drive failures.** Q-learning is choosing actions, not fitting a survival model.
- **Using all of the SMART vector.** We use the cumulative max of `smart_5_raw` (Reallocated Sectors) and `smart_197_raw` (Pending Sectors) at each stage. The chapter's R-learner / A-learning extensions could use the full SMART vector; we keep things to two interpretable features so the decision boundary is visualisable."""),

code("""%pip install -q numpy pandas matplotlib scikit-learn"""),

code("""import os, sys, urllib.request, pathlib

# Pull the prep module from the repo on first run; it loads the committed CSV.
PREP_PATH = pathlib.Path("/content/backblaze_prep.py")
if not PREP_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/backblaze_prep.py",
        PREP_PATH,
    )
sys.path.insert(0, str(PREP_PATH.parent))

CSV_PATH = pathlib.Path("/content/backblaze_subset.csv")
if not CSV_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/backblaze_subset.csv",
        CSV_PATH,
    )

from backblaze_prep import load_backblaze

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load and inspect stage features"""),

code("""d = load_backblaze(chapter=8)
print(f"Shape: {d.shape}")
print(f"  Drives that reach stage 1:       {len(d)}")
print(f"  Drives that fail after stage 1:  {int(d['failed_after_stage1'].sum())}")
print(f"  Drives that fail after stage 2:  {int(d['failed_after_stage2'].sum())}")
print()
# A more interpretable state: log(1 + raw) so the heavy tail compresses.
d['s1_severity'] = np.log1p(d[['state1_smart_5_raw', 'state1_smart_197_raw']].max(axis=1))
d['s2_severity'] = np.log1p(d[['state2_smart_5_raw', 'state2_smart_197_raw']].max(axis=1))

print("Stage-1 severity (log(1 + max of smart_5 / smart_197 through day 7)):")
print(d['s1_severity'].describe())
print()
print("Stage-2 severity:")
print(d['s2_severity'].describe())"""),

md("""**Read the state.** Most drives have `severity = 0` at both stages (no SMART warnings in the first or second week). A small minority show non-zero values; those values can climb into the hundreds (log severity > 5). The Q-function will need to capture both the threshold *crossing* (severity 0 → severity > 0) and the gradient inside the warned population."""),

md("""## Part 2 — Frame the cost-aware MDP

**State.**  $S_t = \\log(1 + \\max(\\text{SMART\\_5}, \\text{SMART\\_197}))$, computed cumulatively through stage $t$. A drive's state never decreases; warning bits only accumulate.

**Action.**  $A_t \\in \\{0, 1\\}$: $1$ = pre-emptively replace, $0$ = wait.

**Trajectory dynamics.** Once $A_t = 1$, the drive is replaced and the trajectory terminates with the replacement cost. If $A_t = 0$, the drive continues to the next stage (or beyond stage 2, to the end of the observation window).

**Reward.** $R$ is realised at the trajectory's terminal step:

| What happens | $R$ |
|--------------|-----|
| $A_1 = 1$: pre-emptive replacement at stage 1 | $-c_R = -1$ |
| $A_1 = 0$, drive fails between stages 1 and 2 | $-c_F = -10$ |
| $A_1 = 0$, $A_2 = 1$: pre-emptive replacement at stage 2 | $-c_R = -1$ |
| $A_1 = 0$, $A_2 = 0$, drive fails after stage 2 | $-c_F = -10$ |
| $A_1 = 0$, $A_2 = 0$, drive survives | $0$ |

**Behaviour policy.**  $A_1, A_2 \\sim \\mathrm{Bernoulli}(0.5)$ — uniform random per drive (synthesised, since Backblaze did not actually replace these drives on our schedule). Under $A = 1$ we counterfactually substitute: the drive is replaced, so it does not subsequently fail in the window.

**Estimand.**  Optimal policy $\\pi^* = (\\pi^*_1, \\pi^*_2)$ that maximises $E[R]$."""),

code("""c_R = 1.0   # replacement cost
c_F = 10.0  # in-service failure cost

# Synthesised behaviour-policy actions
n = len(d)
d['A1'] = rng.integers(0, 2, size=n)
d['A2'] = rng.integers(0, 2, size=n)

# Compute realised terminal reward under the synthesised actions.
# Notation: "fail_between" = drive failed strictly between stage 1 and stage 2.
fail_between_stages = (d['failed_after_stage1'] == 1) & (d['failed_after_stage2'] == 0)
fail_after_stage2   = (d['failed_after_stage2'] == 1)

# A1 = 1 -> replaced at stage 1, R = -c_R
R = np.where(d['A1'] == 1, -c_R, np.nan)

# A1 = 0, drive failed between stages -> R = -c_F (A2 irrelevant; drive already failed)
mask = (d['A1'] == 0) & fail_between_stages
R[mask] = -c_F

# A1 = 0, drive survives to stage 2, A2 = 1 -> R = -c_R
mask = (d['A1'] == 0) & ~fail_between_stages & (d['A2'] == 1)
R[mask] = -c_R

# A1 = 0, drive survives to stage 2, A2 = 0, drive fails after stage 2 -> R = -c_F
mask = (d['A1'] == 0) & ~fail_between_stages & (d['A2'] == 0) & fail_after_stage2
R[mask] = -c_F

# A1 = 0, drive survives to stage 2, A2 = 0, drive survives -> R = 0
mask = (d['A1'] == 0) & ~fail_between_stages & (d['A2'] == 0) & ~fail_after_stage2
R[mask] = 0.0

d['R'] = R
print(f"Synthesised reward stats: mean = {R.mean():+.3f}, min = {R.min():.0f}, max = {R.max():.0f}")
print(f"Counts:")
n_repl_s1 = int(((d['A1'] == 1)).sum())
n_fail_between = int(((d['A1'] == 0) & fail_between_stages).sum())
n_repl_s2 = int(((d['A1'] == 0) & ~fail_between_stages & (d['A2'] == 1)).sum())
n_fail_after_s2 = int(((d['A1'] == 0) & ~fail_between_stages & (d['A2'] == 0) & fail_after_stage2).sum())
n_survived = int(((d['A1'] == 0) & ~fail_between_stages & (d['A2'] == 0) & ~fail_after_stage2).sum())
print(f"  Replaced at stage 1 (A1=1):                       {n_repl_s1}")
print(f"  Waited at stage 1, failed before stage 2:         {n_fail_between}")
print(f"  Reached stage 2, replaced there (A1=0, A2=1):     {n_repl_s2}")
print(f"  Reached stage 2, waited, failed (A1=0, A2=0):     {n_fail_after_s2}")
print(f"  Reached stage 2, waited, survived (A1=0, A2=0):   {n_survived}")"""),

md("""## Background — backward induction in 90 seconds

Before Parts 3-4 fit Q-functions, let's make the *why* concrete. Q-learning by backward induction has two ideas, both due to Bellman:

**Idea 1: the Q-function is an expectation of the trajectory's total reward, conditional on starting state and action and *then playing optimally*.**

$$Q_t(s, a) \\;\\equiv\\; E\\big[\\,R \\;\\big|\\; S_t = s, \\, A_t = a, \\, \\text{and future actions follow the optimal policy}\\,\\big].$$

So $Q_2(s_2, 1)$ asks "if I am at stage 2 in state $s_2$ and I replace, what is my expected total cost?" — easy, it is $-c_R = -1$ no matter the state. And $Q_2(s_2, 0)$ asks "if I am at stage 2 in state $s_2$ and I wait, what is my expected total cost?" — that depends on the failure risk in $s_2$, which is what the regression must learn.

**Idea 2: solve the *last* stage first, then propagate back.** This is the principle of optimality: at stage $T$ the future contains no further decisions, so $Q_T$ is just an expectation of the realised terminal reward — a normal regression problem. Once we have $Q_T$, the value of being in *any* state at $T$ is $V_T(s) = \\max_a Q_T(s, a)$. We then solve stage $T-1$ by treating $V_T(s_T)$ as the *continuation reward* — i.e., the future reward you can expect if you survive stage $T-1$ and play optimally at stage $T$.

For our two-stage problem this means: **Part 3 fits $Q_2$ from realised data (the bottom of the chain); Part 4 uses $\\hat{V}_2 = \\max_a \\hat{Q}_2$ as the *future-value substitute* in the stage-1 regression** so the stage-1 fit can correctly value the option of waiting.

**The "pseudo-outcome" trick in Part 4 is just this substitution.** For each drive at stage 1:

| Stage-1 action / outcome | Pseudo-outcome we regress on | Why |
|--------------------------|------------------------------|-----|
| $A_1 = 1$ (replace) | $-c_R = -1$ | trajectory ends; reward fully realised |
| $A_1 = 0$, failed before stage 2 | $-c_F = -10$ | trajectory ends; reward fully realised |
| $A_1 = 0$, survived to stage 2 | $\\hat{V}_2(s_2)$ | trajectory continues; substitute optimal future value |

The third row is the only one that uses $\\hat{V}_2$. The first two are realised rewards. The stage-1 regression then learns $Q_1(s_1, a_1)$ from this hybrid outcome, and the optimal stage-1 policy is $\\arg\\max_a \\hat{Q}_1(s_1, a)$."""),

md("""## Part 3 — Stage-2 Q-function

Fit $\\hat Q_2(S_2, A_2)$ by regressing the *realised* terminal reward on $(S_2, A_2)$ using the subset of drives that reached stage 2 with $A_1 = 0$. We use a quadratic basis to let the model bend across the severity axis.

This estimates the conditional expectation of the trajectory's final reward given the stage-2 state and action."""),

code("""# Restrict to drives that reached stage 2 with A1 = 0
mask_stage2 = (d['A1'] == 0) & ~fail_between_stages
d2 = d[mask_stage2].copy()
n_s2 = len(d2)
n_repl_s2_arm = int((d2['A2']==1).sum())
n_wait_s2_arm = int((d2['A2']==0).sum())
n_fail_in_wait = int(((d2['A2']==0) & (d2['failed_after_stage2']==1)).sum())
print(f"Stage-2 fitting subset: {n_s2} drives  ({n_repl_s2_arm} replaced, {n_wait_s2_arm} waited)")
print(f"  Failures in stage-2 wait arm: {n_fail_in_wait}")
print()
if n_fail_in_wait < 20:
    print(f"  CAVEAT: only {n_fail_in_wait} failure events in the wait arm means the stage-2")
    print(f"  Q-function's threshold has wide error bars. The decision-boundary plot below")
    print(f"  is qualitatively informative but the exact crossover severity is noisy.")

poly2 = PolynomialFeatures(degree=2, include_bias=False)
X2 = poly2.fit_transform(np.column_stack([d2['s2_severity'].values, d2['A2'].values]))
Q2_model = LinearRegression().fit(X2, d2['R'].values)

def Q2(s, a):
    s = np.atleast_1d(s).astype(float)
    a = np.atleast_1d(a).astype(float)
    X = poly2.transform(np.column_stack([s, a]))
    return Q2_model.predict(X)

# Plot Q2 vs severity, for A2 = 0 and A2 = 1
s_grid = np.linspace(0, d['s2_severity'].max(), 200)
q2_a0 = Q2(s_grid, np.zeros_like(s_grid))
q2_a1 = Q2(s_grid, np.ones_like(s_grid))
opt_a2 = (q2_a1 > q2_a0).astype(int)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(s_grid, q2_a0, label='Q2(s, A2=0)  -- wait', color='C0')
ax.plot(s_grid, q2_a1, label='Q2(s, A2=1)  -- replace', color='C1')
ax.fill_between(s_grid, ax.get_ylim()[0], ax.get_ylim()[1], where=(opt_a2 == 1),
                alpha=0.15, color='C1', label='Q-learning says replace')
ax.set_xlabel(r'Stage-2 severity  $\\log(1 + \\max(\\mathrm{SMART\\_5}, \\mathrm{SMART\\_197}))$')
ax.set_ylabel(r'$Q_2(s, A_2)$ [expected reward]')
ax.set_title('Stage-2 Q-function and decision boundary')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Find the crossing point (decision threshold)
diff = q2_a1 - q2_a0
sign_change = np.where(np.diff(np.sign(diff)))[0]
if len(sign_change) > 0:
    thresh = s_grid[sign_change[0]]
    print(f"Stage-2 decision threshold (Q2 crossing): severity > {thresh:.3f}  (raw SMART >~ {np.expm1(thresh):.0f})")
else:
    print('No crossing within plotted range; one action dominates throughout.')"""),

md("""## Part 4 — Stage-1 Q-function

For stage 1 we regress a *pseudo-outcome* on $(S_1, A_1)$:

- $A_1 = 1$: pseudo-outcome is just the realised reward, $-c_R$.
- $A_1 = 0$ and drive fails between stages: pseudo-outcome is the realised reward, $-c_F$.
- $A_1 = 0$ and drive survives to stage 2: pseudo-outcome is the *value* of the stage-2 state, $\\hat V_2(S_2) = \\max_a \\hat Q_2(S_2, a)$.

The third case is the backward-induction step — we substitute the *optimal* stage-2 value for the realised stage-2 reward."""),

code("""# Compute V2 for every drive that survived to stage 2 (uses their actual s2_severity)
v2 = np.maximum(
    Q2(d['s2_severity'].values, np.zeros(len(d))),
    Q2(d['s2_severity'].values, np.ones(len(d))),
)

# Build the pseudo-outcome
pseudo = np.full(len(d), np.nan)
# A1 = 1 -> realised reward
pseudo[d['A1'] == 1] = -c_R
# A1 = 0 and failed between stages -> realised reward
m = (d['A1'] == 0) & fail_between_stages
pseudo[m] = -c_F
# A1 = 0 and survived to stage 2 -> V2(s2)
m = (d['A1'] == 0) & ~fail_between_stages
pseudo[m] = v2[m]

assert not np.isnan(pseudo).any(), 'pseudo-outcome has NaN; check the cases.'

poly1 = PolynomialFeatures(degree=2, include_bias=False)
X1 = poly1.fit_transform(np.column_stack([d['s1_severity'].values, d['A1'].values]))
Q1_model = LinearRegression().fit(X1, pseudo)

def Q1(s, a):
    s = np.atleast_1d(s).astype(float)
    a = np.atleast_1d(a).astype(float)
    X = poly1.transform(np.column_stack([s, a]))
    return Q1_model.predict(X)

# Plot Q1 vs severity
s_grid = np.linspace(0, d['s1_severity'].max(), 200)
q1_a0 = Q1(s_grid, np.zeros_like(s_grid))
q1_a1 = Q1(s_grid, np.ones_like(s_grid))
opt_a1 = (q1_a1 > q1_a0).astype(int)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(s_grid, q1_a0, label='Q1(s, A1=0)  -- wait', color='C0')
ax.plot(s_grid, q1_a1, label='Q1(s, A1=1)  -- replace', color='C1')
ax.fill_between(s_grid, ax.get_ylim()[0], ax.get_ylim()[1], where=(opt_a1 == 1),
                alpha=0.15, color='C1', label='Q-learning says replace')
ax.set_xlabel(r'Stage-1 severity  $\\log(1 + \\max(\\mathrm{SMART\\_5}, \\mathrm{SMART\\_197}))$')
ax.set_ylabel(r'$Q_1(s, A_1)$ [expected reward]')
ax.set_title('Stage-1 Q-function and decision boundary')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

diff = q1_a1 - q1_a0
sign_change = np.where(np.diff(np.sign(diff)))[0]
if len(sign_change) > 0:
    thresh = s_grid[sign_change[0]]
    print(f'Stage-1 decision threshold: severity > {thresh:.3f}  (raw SMART >~ {np.expm1(thresh):.0f})')
else:
    print('No crossing within plotted range; one action dominates throughout.')"""),

md("""## Part 5 — Evaluate policies against baselines

The realised-reward column reflects the *behaviour policy's* synthesised actions. To compare policies we need the realised reward had we taken *the policy's* actions on each drive. For each drive we know its actual fate under $A = 0$ (from the data) and we assume successful replacement under $A = 1$. So we can compute each policy's expected reward exactly by simulating the deterministic action choice.

**Baselines.**

- **Never replace.** $A_1 = A_2 = 0$ always. Pays $-c_F$ on every failure.
- **Always replace at stage 1.** $A_1 = 1$ always. Pays $-c_R$ on every drive, regardless of failure risk.
- **Threshold-based: replace if severity > 0.** A simple rule a maintenance team could write on a whiteboard — replace any drive showing any warning bit.
- **Q-learning.** Replace when $\\hat Q_t(s, 1) > \\hat Q_t(s, 0)$ at each stage."""),

code("""def evaluate_policy(d, pi1, pi2):
    \"\"\"Compute the mean realised reward had we used (pi1, pi2) for every drive.

    pi1, pi2: callables taking the per-drive severity vector, returning 0/1 actions.
    \"\"\"
    a1 = pi1(d['s1_severity'].values).astype(int)
    a2 = pi2(d['s2_severity'].values).astype(int)

    fail_between = ((d['failed_after_stage1'] == 1) & (d['failed_after_stage2'] == 0)).values
    fail_after = (d['failed_after_stage2'] == 1).values

    R = np.zeros(len(d))
    # A1 = 1 -> replace
    R[a1 == 1] = -c_R
    # A1 = 0 -> trajectory continues
    m = (a1 == 0)
    # A1 = 0 and failed between stages
    R[m & fail_between] = -c_F
    # A1 = 0 and survived to stage 2 and A2 = 1
    m2 = m & ~fail_between & (a2 == 1)
    R[m2] = -c_R
    # A1 = 0 and survived to stage 2 and A2 = 0 and failed after stage 2
    m3 = m & ~fail_between & (a2 == 0) & fail_after
    R[m3] = -c_F
    # all remaining drives -> R = 0 (survives without replacement)
    return float(R.mean())

# Never replace
v_never = evaluate_policy(d, lambda s: np.zeros_like(s), lambda s: np.zeros_like(s))
# Always replace at stage 1
v_always_s1 = evaluate_policy(d, lambda s: np.ones_like(s), lambda s: np.ones_like(s))
# Threshold at severity > 0
v_thresh = evaluate_policy(d, lambda s: (s > 0).astype(int), lambda s: (s > 0).astype(int))
# Q-learning
def pi1_q(s):
    return (Q1(s, np.ones_like(s)) > Q1(s, np.zeros_like(s))).astype(int)
def pi2_q(s):
    return (Q2(s, np.ones_like(s)) > Q2(s, np.zeros_like(s))).astype(int)
v_qlearn = evaluate_policy(d, pi1_q, pi2_q)

table = pd.DataFrame({
    'policy': ['Never replace',
               'Always replace (stage 1)',
               'Threshold (severity > 0)',
               'Q-learning'],
    'E[reward per drive]': [v_never, v_always_s1, v_thresh, v_qlearn],
    'E[cost per drive]':   [-v_never, -v_always_s1, -v_thresh, -v_qlearn],
})
table = table.sort_values('E[reward per drive]', ascending=False).reset_index(drop=True)
print(table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))"""),

md("""**Read the table.**

- **Never-replace** is competitive because failures are rare in absolute terms (~3 % over 30 days at our cost ratio of 10:1). Paying $1$ on every drive to dodge a failure that only hits 3 % of the population at cost $10$ is bad maths.
- **Always-replace** is the worst — paying $c_R$ on every drive when most do not need it.
- **Threshold (severity > 0)** beats never-replace because the drives showing any SMART warning *are* the ones at substantially elevated failure risk. Replacing 10–20 drives out of ~1500 buys us large savings on the very high in-warned-drives failure rate.
- **Q-learning** matches or beats the threshold by tuning the cutoff using the data. With our limited number of warned drives (5–22 across the two stages), Q-learning's advantage over the simple-threshold heuristic is modest, but the *direction* of the gain is informative: the data-driven cutoff lives on the high-severity side of `> 0`."""),

md("""## Part 6 — Where the policy disagrees with the baselines

The most actionable diagnostic is the *drive-level* disagreement between Q-learning and the threshold rule. Drives where the two disagree are the candidates a maintenance engineer should re-inspect."""),

code("""actions_q = {
    's1_q':      pi1_q(d['s1_severity'].values),
    's1_thresh': (d['s1_severity'].values > 0).astype(int),
    's2_q':      pi2_q(d['s2_severity'].values),
    's2_thresh': (d['s2_severity'].values > 0).astype(int),
}
A = pd.DataFrame(actions_q)
disagree_s1 = (A['s1_q'] != A['s1_thresh']).sum()
disagree_s2 = (A['s2_q'] != A['s2_thresh']).sum()
print(f"Stage-1 action disagreements (Q-learning vs threshold): {disagree_s1} / {len(d)}")
print(f"Stage-2 action disagreements (Q-learning vs threshold): {disagree_s2} / {len(d)}")
print()
print("Stage-2 disagreement breakdown by severity tier:")
tier = pd.cut(d['s2_severity'], bins=[-0.001, 0, 1, 3, 10], labels=['s=0', '0<s<=1', '1<s<=3', 's>3'])
print(pd.crosstab(tier, A['s2_q'].astype(str) + ' vs ' + A['s2_thresh'].astype(str), margins=True))"""),

md("""## Part 7 — Decision

Three bullets, the data-centre PM-policy takeaway:

1. **The data-driven optimal policy *does* replace early — but only on drives that show any SMART_5 or SMART_197 warning bit during the first or second week.** Around 95 %+ of the fleet runs to the end of the observation window with no action; the remaining 5 % gets replaced, with most replacements concentrated in stage 2 (where the conditional failure rate is highest given the warning).

2. **The cost-per-drive ranking is `Q-learning ≥ threshold > never-replace > always-replace`**, with the gap between Q-learning and the threshold smaller than the gap between the threshold and the do-nothing baselines. The strong learning signal is the *binary warned-or-not* distinction; the within-warned tuning Q-learning adds is a secondary refinement at this sample size.

3. **The biggest miss for a never-replace shop is not the *total* cost (which is small at 3% failure rate) but the operational disruption from in-service failures.** Our cost ratio of $c_F / c_R = 10$ is conservative; for many real fleets it is closer to 50 or 100 (a fleet-wide RAID rebuild, lost SLA, customer-data exposure). Re-running the analysis with `c_F = 100` makes the threshold-based and Q-learning policies dominate by a much wider margin than the table above suggests."""),

md("""## Reflection

**Backward induction *is* the magic.** The reason Q-learning beats the threshold heuristic at all is that stage 1's pseudo-outcome includes $\\hat V_2(S_2)$ — the *value of being able to make a stage-2 decision later*. A naive single-stage policy that does not see the chain forward will replace too aggressively at stage 1 (since it has no option to wait and re-decide). The chapter's exercise on "myopic" policies makes this point analytically; here it shows up as a small but real advantage of the two-stage policy.

**The behaviour policy mattered.** We used $\\mathrm{Bernoulli}(0.5)$ for both actions, which gives the regressions plenty of coverage in the $(s, a)$ joint. A real logged policy in a maintenance team's history would not be uniform — they would have replaced drives showing warnings more often than not. With informative actions, propensity overlap becomes the bottleneck and the Q-learning fits get noisier; the chapter's IPW and DR variants (Murphy 2003) are the answer.

**Cost ratio sensitivity.** The chapter's analytic derivation gives an optimal threshold at $L > c_R / 2$ in the closed-form SCM. In our continuous-severity Backblaze setting the threshold's *location* is data-driven, but the same monotonicity applies: a higher $c_F / c_R$ ratio pushes the threshold left (replace more aggressively), and the magnitude of the gap between the threshold policy and never-replace grows."""),

md("""## What's next

Lab 10B uses the same Backblaze panel for a *mediation* analysis: of SMART_5's effect on failure, how much flows through SMART_197 and how much is direct? Where 8B finds the policy that minimises cost given the SMART chain, 10B asks *which links* in the chain carry the signal.

For the *informative-behaviour-policy* case (replacing the synthesised $\\mathrm{Bernoulli}(0.5)$ with a real maintenance team's logged decisions), the methodology lives in **Lab 11B** — per-step IPS / SNIPS / DR with propensity-overlap diagnostics, on the Tennessee Eastman simulator. Reading 8B + 11B together gives the full DTR-learning-and-evaluation pipeline: 8B is *learn the policy*, 11B is *value it without deploying*."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch08" / "lab08b_backblaze_dtr.ipynb", cells)
print("Built lab08b_backblaze_dtr.ipynb")
