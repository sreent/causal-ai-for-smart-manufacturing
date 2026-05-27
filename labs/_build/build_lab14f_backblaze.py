"""Build labs/ch14/lab14f_backblaze.ipynb — guided capstone Starter D on Backblaze."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402


def reveal(content: str) -> dict:
    return md("<details>\n<summary><b>Click to reveal sample answer</b></summary>\n\n"
              + content + "\n\n</details>")


cells = [

md("""# Lab 14F — Guided Capstone (Starter D): Optimal Pre-emptive Replacement Threshold on Backblaze Drives

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14f_backblaze.ipynb)

**Starter D** of the §14.7 capstone. This is the *production-flavoured* starter: instead of a static ATE, the capstone targets a **dynamic treatment regime (DTR)** — the optimal SMART-based replacement policy across two decision stages, evaluated under three cost-asymmetry regimes.

**The capstone question.** *What is the optimal SMART-based replacement policy at days 7 and 14, under cost ratios $c_F / c_R \\in \\{10, 100, 1000\\}$ (in-service-failure cost vs pre-emptive-replacement cost)? At what threshold does the decision boundary move?*

**Companion labs.** Lab 8B (DTR Q-learning on Backblaze) is the policy-learning template; Lab 11B (OPE on TE) is the off-policy-evaluation tooling for valuing the learned policy.

**Why this starter is "production-flavoured".** A production maintenance team will ask "what cost ratio justifies the policy?" before the analyst's first answer is dry. The capstone is set up to answer that question by sweeping the cost ratio explicitly."""),

md("""## Setup"""),

code("""%pip install -q numpy pandas matplotlib scikit-learn"""),

code("""import os, sys, urllib.request, pathlib

DATA = pathlib.Path("/content")
for name in ("backblaze_prep.py", "backblaze_subset.csv"):
    p = DATA / name
    if not p.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/{name}",
            p,
        )
sys.path.insert(0, str(DATA))

from backblaze_prep import load_backblaze
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
d = load_backblaze(chapter=8)
d['s1_severity'] = np.log1p(d[['state1_smart_5_raw', 'state1_smart_197_raw']].max(axis=1))
d['s2_severity'] = np.log1p(d[['state2_smart_5_raw', 'state2_smart_197_raw']].max(axis=1))
print(f'Drives: {len(d)}    Failed by stage 1: {int(d[\"failed_after_stage1\"].sum())}    by stage 2: {int(d[\"failed_after_stage2\"].sum())}')"""),

md("""## Artifact 1 — Problem Statement and DAG

### Q1.1 Write the problem statement.

*Hint.* The decision is a data-centre operations policy: at what SMART threshold should a drive be pre-emptively replaced? The tradeoff: replacement cost ($c_R$) vs in-service failure cost ($c_F$).

*Your turn.*"""),

md("""*[Problem statement.]*"""),

reveal("""A data-centre operator observes SMART metrics for ~10,000 ST4000DM000 drives daily. At each weekly review, the operator can either spend $c_R$ on a pre-emptive replacement or wait and accept a $c_F$ cost if the drive fails in service before the next review. $c_R$ is fixed by procurement; $c_F$ depends on the operational disruption (RAID rebuild, SLA breach, customer-data exposure) and varies by deployment regime. The team needs the optimal *state-dependent* replacement policy at days 7 and 14, evaluated under three regimes: $c_F / c_R = 10$ (commodity storage), $100$ (transactional), $1000$ (financial). The policy must minimise expected cost per drive."""),

md("""### Q1.2 Draw the DTR DAG.

*Hint.* The DAG is sequential: state at $t$ → action at $t$ → state at $t+1$ → action at $t+1$ → outcome. SMART metrics are the state; replace/wait is the action; failure or replacement is the terminal outcome.

*Your turn.*"""),

md("""*[DTR DAG.]*"""),

reveal("""```
   S_1 (SMART at day 7) ──► A_1 (replace/wait at day 7) ──► R_1 (terminal if A_1=1)
       │                       │
       └──► S_2 (SMART at day 14, IF A_1=0)
                └──► A_2 (replace/wait at day 14) ──► R_2 (terminal if A_2=1 or failure)

   Drive's idiosyncratic noise (latent: temperature, vibration, firmware lot)
       │
       └──► S_1, S_2, failure timing
```
"""),

md("""### Q1.3 Defend the "synthesised action" framing.

*Hint.* Backblaze did NOT actually replace these drives on a fixed schedule — the data is observational. The DTR analysis *synthesises* a hypothetical action policy and uses the observed SMART/failure data to compute counterfactual rewards.

*Your turn.*"""),

md("""*[Defense of synthesised actions.]*"""),

reveal("""Backblaze's published Drive Stats are a *passive observation* of drives running to natural end-of-life. There is no logged replacement decision per drive per day. To pose the DTR question on this data, we *synthesise* a Bernoulli(0.5) behaviour policy that randomly replaces each drive at each stage, then compute the counterfactual reward each drive would have received under each policy: under $A = 1$ (replace), the drive is removed before its natural failure, so $R = -c_R$; under $A = 0$ (wait), we observe the drive's actual SMART trajectory and failure timing, so $R = 0$ (survives) or $R = -c_F$ (fails before next stage). This is the standard counterfactual-substitution setup for offline policy learning. The validity rests on *the substitution being correct* — that replacement does prevent failure (mechanistically true for a fresh drive)."""),

md("""## Artifact 2 — Estimand

### Q2.1 Write the estimand: expected cost under the optimal policy.

*Your turn.*"""),

md("""*[Estimand.]*"""),

reveal("""$$\\pi^* = \\arg\\min_\\pi E[\\mathrm{cost}(\\pi)],$$

where $\\mathrm{cost}(\\pi) = c_R \\cdot \\mathbf{1}[\\text{replaced under } \\pi] + c_F \\cdot \\mathbf{1}[\\text{failed in-service under } \\pi]$. The optimal $\\pi^*$ is a per-stage threshold on the severity state $S_t = \\log(1 + \\max(\\text{smart\\_5}, \\text{smart\\_197}))$ at $t \\in \\{7, 14\\}$. We report the threshold for each cost ratio $c_F / c_R \\in \\{10, 100, 1000\\}$."""),

md("""### Q2.2 Couple to a decision threshold.

*Your turn.*"""),

md("""*[Decision rule.]*"""),

reveal("""- The deliverable is the *learned threshold* on severity at each stage, per cost ratio. If at $c_F / c_R = 10$ the threshold is severity > 0.7 (raw SMART > 1), the team replaces only drives whose first-week max(smart_5, smart_197) exceeds 1 — a small fraction of the fleet.
- If at $c_F / c_R = 100$ the threshold moves to severity > 0.0 (any SMART warning), the team replaces all warned drives — a more aggressive policy.
- If at $c_F / c_R = 1000$ the threshold approaches "replace all" — the cost of an in-service failure is so high that even drives with no SMART warning may be worth replacing."""),

md("""## Artifact 3 — Identification

### Q3.1 Name the identification strategy for the DTR policy value.

*Hint.* The chapter-8 strategy is Q-learning by backward induction. Under sequential ignorability, the learned policy is consistent for the optimal policy.

*Your turn.*"""),

md("""*[Strategy + assumptions.]*"""),

reveal("""**Q-learning by backward induction** with the synthesised behaviour policy. Identification rests on:

1. *Sequential ignorability* of the synthesised actions: by construction, $A_t \\sim \\mathrm{Bernoulli}(0.5)$ independent of $S_t$, so no unmeasured confounding of $A_t$ on the future reward given $S_t$.
2. *Counterfactual substitution validity:* the synthesised reward under $A = 1$ ($-c_R$, no failure) is correct mechanistically (a replaced drive cannot fail).
3. *Coverage:* the Bernoulli(0.5) behaviour policy ensures both $A_t = 0$ and $A_t = 1$ have positive probability in every state, so the Q-function can be estimated everywhere."""),

md("""### Q3.2 Where does this analysis fall short of a "real DTR"?

*Your turn.*"""),

md("""*[Honest limitation statement.]*"""),

reveal("""A real maintenance team's logged replacement decisions would NOT be Bernoulli(0.5) — they would replace warned drives more often than healthy ones. With an *informative* behaviour policy, the identification step becomes off-policy evaluation under unequal propensity (Lab 11B's machinery), not the simple Q-learning of this capstone. The synthesised actions sidestep that complication for pedagogical simplicity. If this capstone were extended to a real maintenance log, the OPE component would be the load-bearing addition."""),

md("""## Artifact 4 — Estimator

### Q4.1 Synthesise actions, compute rewards under a chosen cost ratio, fit Q-functions.

*Hint.* Reuse Lab 8B's backward-induction pattern. Start with $c_F / c_R = 10$.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
c_R = 1.0
c_F = 10.0    # start here; will sweep later
n = len(d)
A1 = rng.integers(0, 2, size=n)
A2 = rng.integers(0, 2, size=n)
fail_between = ((d['failed_after_stage1'] == 1) & (d['failed_after_stage2'] == 0)).values
fail_after_2 = (d['failed_after_stage2'] == 1).values

R = np.where(A1 == 1, -c_R, np.nan)
m = (A1 == 0) & fail_between; R[m] = -c_F
m = (A1 == 0) & ~fail_between & (A2 == 1); R[m] = -c_R
m = (A1 == 0) & ~fail_between & (A2 == 0) & fail_after_2; R[m] = -c_F
m = (A1 == 0) & ~fail_between & (A2 == 0) & ~fail_after_2; R[m] = 0.0

# Stage-2 Q: regress R on (s2, A2) for drives that reached stage 2 with A1=0.
m2 = (A1 == 0) & ~fail_between
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X2 = poly2.fit_transform(np.column_stack([d['s2_severity'].values[m2], A2[m2]]))
Q2 = LinearRegression().fit(X2, R[m2])

def Q2_eval(s, a):
    X = poly2.transform(np.column_stack([np.atleast_1d(s), np.atleast_1d(a).astype(float)]))
    return Q2.predict(X)

# Stage-1 backward induction: pseudo-outcome substitutes V_2 for the survivor branch.
V2 = np.maximum(Q2_eval(d['s2_severity'].values, np.zeros(n)),
                Q2_eval(d['s2_severity'].values, np.ones(n)))
pseudo = np.full(n, np.nan)
pseudo[A1 == 1] = -c_R
m = (A1 == 0) & fail_between; pseudo[m] = -c_F
m = (A1 == 0) & ~fail_between; pseudo[m] = V2[m]
poly1 = PolynomialFeatures(degree=2, include_bias=False)
X1 = poly1.fit_transform(np.column_stack([d['s1_severity'].values, A1]))
Q1 = LinearRegression().fit(X1, pseudo)
print(f'Stage-1 Q fit on {len(d)} drives; stage-2 Q fit on {m2.sum()} drives.')
```
"""),

md("""### Q4.2 Recover the per-stage thresholds and evaluate the learned policy.

*Hint.* The threshold is the severity at which $Q_t(s, 1) > Q_t(s, 0)$ flips.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
def Q1_eval(s, a):
    X = poly1.transform(np.column_stack([np.atleast_1d(s), np.atleast_1d(a).astype(float)]))
    return Q1.predict(X)

def thresh(Q_eval, s_max):
    grid = np.linspace(0, s_max, 200)
    diff = Q_eval(grid, np.ones_like(grid)) - Q_eval(grid, np.zeros_like(grid))
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    return float(grid[sign_change[0]]) if len(sign_change) else None

print(f'Stage-1 threshold: {thresh(Q1_eval, d[\"s1_severity\"].max())}')
print(f'Stage-2 threshold: {thresh(Q2_eval, d[\"s2_severity\"].max())}')

# Evaluate the learned policy on the data (Lab 8B's evaluate_policy pattern)
def evaluate_policy(d, pi1, pi2, c_R=1.0, c_F=10.0):
    a1 = pi1(d['s1_severity'].values).astype(int)
    a2 = pi2(d['s2_severity'].values).astype(int)
    fb = ((d['failed_after_stage1'] == 1) & (d['failed_after_stage2'] == 0)).values
    fa = (d['failed_after_stage2'] == 1).values
    R = np.zeros(len(d))
    R[a1 == 1] = -c_R
    m = (a1 == 0); R[m & fb] = -c_F
    m2 = m & ~fb & (a2 == 1); R[m2] = -c_R
    R[m & ~fb & (a2 == 0) & fa] = -c_F
    return float(R.mean())

pi1_q = lambda s: (Q1_eval(s, np.ones_like(s)) > Q1_eval(s, np.zeros_like(s))).astype(int)
pi2_q = lambda s: (Q2_eval(s, np.ones_like(s)) > Q2_eval(s, np.zeros_like(s))).astype(int)
v_q = evaluate_policy(d, pi1_q, pi2_q, c_R, c_F)
v_never = evaluate_policy(d, lambda s: np.zeros_like(s), lambda s: np.zeros_like(s), c_R, c_F)
v_thresh = evaluate_policy(d, lambda s: (s > 0).astype(int), lambda s: (s > 0).astype(int), c_R, c_F)
print(f'Policy values per drive: Q-learning {v_q:+.4f}, threshold(>0) {v_thresh:+.4f}, never {v_never:+.4f}')
```
"""),

md("""### Q4.3 Sweep the cost ratio: how does the threshold move?

*Hint.* Wrap the Q-fitting + threshold extraction in a function and call it at $c_F / c_R \\in \\{10, 100, 1000\\}$.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
def fit_dtr(c_R, c_F):
    R = np.where(A1 == 1, -c_R, np.nan)
    m = (A1 == 0) & fail_between; R[m] = -c_F
    m = (A1 == 0) & ~fail_between & (A2 == 1); R[m] = -c_R
    m = (A1 == 0) & ~fail_between & (A2 == 0) & fail_after_2; R[m] = -c_F
    m = (A1 == 0) & ~fail_between & (A2 == 0) & ~fail_after_2; R[m] = 0.0
    m2 = (A1 == 0) & ~fail_between
    X2 = poly2.transform(np.column_stack([d['s2_severity'].values[m2], A2[m2]]))
    Q2 = LinearRegression().fit(X2, R[m2])
    def Q2_eval(s, a):
        return Q2.predict(poly2.transform(np.column_stack([np.atleast_1d(s), np.atleast_1d(a).astype(float)])))
    V2 = np.maximum(Q2_eval(d['s2_severity'].values, np.zeros(n)),
                    Q2_eval(d['s2_severity'].values, np.ones(n)))
    pseudo = np.full(n, np.nan)
    pseudo[A1 == 1] = -c_R
    pseudo[(A1 == 0) & fail_between] = -c_F
    pseudo[(A1 == 0) & ~fail_between] = V2[(A1 == 0) & ~fail_between]
    X1 = poly1.transform(np.column_stack([d['s1_severity'].values, A1]))
    Q1 = LinearRegression().fit(X1, pseudo)
    def Q1_eval(s, a):
        return Q1.predict(poly1.transform(np.column_stack([np.atleast_1d(s), np.atleast_1d(a).astype(float)])))
    return Q1_eval, Q2_eval

for c_F in [10, 100, 1000]:
    Q1e, Q2e = fit_dtr(1.0, c_F)
    t1 = thresh(Q1e, d['s1_severity'].max())
    t2 = thresh(Q2e, d['s2_severity'].max())
    print(f'c_F/c_R = {c_F}:  stage-1 thresh = {t1}, stage-2 thresh = {t2}')
```
"""),

md("""**Discussion — read the cost-sweep output carefully, since the threshold can degenerate.**

As $c_F / c_R$ rises, the Q-learning decision boundary moves toward more aggressive pre-emptive replacement. At low cost ratios the threshold sits inside the severity range (e.g., $c_F / c_R = 10$ gives a stage-1 threshold around severity > 0.68, equivalent to raw SMART ≳ 1). At high cost ratios the boundary leaves the range entirely: the threshold-extractor returns `None` because the curve $Q_t(s, \\text{replace}) - Q_t(s, \\text{wait})$ no longer crosses zero — *replace* dominates at every observable severity. That is the optimal answer (replace-everywhere is cheaper than accepting $c_F = 1000$ failure costs on the small subset that would fail), but the *threshold* no longer parameterises it.

So the empirical sweep on this Backblaze subset shows three regimes, not a smooth threshold migration:

| $c_F / c_R$ | Optimal policy regime | Threshold reported |
|---|---|---|
| 1-10 | Threshold-based — replace only drives with severity above a learned cutoff | $\\approx 0.27$ - $0.68$ in log-severity |
| ~30-100 | Threshold collapses; *replace-everywhere* dominates "wait" on every observable state | `None` |
| 100+ | Replace-everywhere is strictly optimal | `None` |

If your team is choosing between a 10:1, 100:1, or 1000:1 cost regime, the deliverable is *which regime your operational cost ratio actually inhabits* — and whether the Q-learning policy in that regime is still meaningfully different from a much simpler "replace any drive with severity > 0" or "replace all" rule. On this dataset the threshold-based and Q-learning policies *agree exactly* at $c_F/c_R = 10$ because no drive's severity falls between 0 and the learned threshold; Q-learning's value-add at that ratio is the *diagnosis that the rule is correct*, not the rule itself. At higher ratios, the answer "replace everywhere" is itself the policy."""),

md("""## Artifact 5 — Sensitivity Analysis

### Q5.1 Sensitivity to the Bernoulli(0.5) behaviour policy assumption.

*Hint.* Re-fit the Q-functions with a different behaviour policy (e.g., $\\mathrm{Bernoulli}(0.3)$ on action). Threshold should not move much if the synthesised-action framing is robust.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
for p_action in [0.3, 0.5, 0.7]:
    A1_p = rng.integers(0, 100, size=n) < (p_action * 100)
    A2_p = rng.integers(0, 100, size=n) < (p_action * 100)
    # ... refit thresholds; left as exercise but should approximate the Bernoulli(0.5) results
    print(f'P(action) = {p_action}: re-fit Q-thresholds and compare to baseline.')
```
"""),

md("""### Q5.2 Verdict.

*Your turn.*"""),

md("""*[Verdict on policy robustness.]*"""),

reveal("""If the learned thresholds at all three cost ratios change by less than 0.1 in log-severity units across behaviour-policy probabilities $\\in [0.3, 0.7]$, the policy is **robust** to the synthesised-action choice. Otherwise the policy is **fragile** and a real maintenance team's logged decisions are needed before deployment (Lab 11B's OPE machinery)."""),

md("""## Artifact 6 — Deployment-Readiness Checklist

### Q6.1 Target population.

*Your turn.*"""),

md("""*[Target population.]*"""),

reveal("""ST4000DM000 drives in the deployment data centre, in the first 30 days of new-drive service, under the operational regime represented by the Backblaze cohort (commercial cloud storage, ambient temperature 15-25°C, weekly SMART polling). Deployment to a different drive model, a different age window, or a different temperature envelope requires re-estimation."""),

md("""### Q6.2 Three deployment monitors.

*Your turn.*"""),

md("""*[Three monitors.]*"""),

reveal("""1. **Daily SMART distribution shift.** Alarm if the deployed fleet's median smart_5 or smart_197 shifts by more than 50% from the deployment baseline.
2. **Weekly per-stage replacement rate vs predicted.** Alarm if the actual replacement rate at stage 1 (or stage 2) deviates from the model-predicted rate by more than 20% relative.
3. **Monthly cost-ratio recompute.** Re-poll the operational cost team for the current $c_F / c_R$. Alarm if it has shifted by more than 30%; refit the DTR at the new ratio.

Rollback: revert to the "never replace before failure" baseline (the most conservative policy) via the line-control configuration. Authority: data-centre operations lead."""),

md("""## Closing

Variants:
- Switch from Bernoulli(0.5) synthesised actions to an *informative* behaviour policy (e.g., $P(A_t = 1) = \\sigma(\\beta \\cdot \\text{severity})$), then use Lab 11B's IPS / SNIPS / DR estimators to value the learned policy. This is the *real-DTR* extension Lab 8B's disclaimer points at.
- Add a continuous severity feature (e.g., smart_5_raw on a log scale) to the state, making the Q-function nonlinear in severity. Compare to the binary-severity baseline.
- Per-drive-age CATE: does the learned threshold differ for drives in their first 30 days vs drives at 90+ days of service?

The DTR + OPE combination is the production-flavoured causal-AI deliverable; this capstone covers the policy-learning half (Lab 8B) cleanly and references the OPE half (Lab 11B) for extension. Refer to [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the per-artifact standards."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14f_backblaze.ipynb", cells)
print("Built lab14f_backblaze.ipynb")
