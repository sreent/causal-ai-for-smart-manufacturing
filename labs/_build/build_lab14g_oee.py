"""Build labs/ch14/lab14g_oee.ipynb — guided capstone Starter E on OEE synthetic data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402


def reveal(content: str) -> dict:
    return md("<details>\n<summary><b>Click to reveal sample answer</b></summary>\n\n"
              + content + "\n\n</details>")


cells = [

md("""# Lab 14G — Guided Capstone (Starter E): Multi-Mediator A x P x Q Decomposition of OEE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14g_oee.ipynb)

**Starter E** of the §14.7 capstone. This is the **verifiable-answer** starter: the dataset is curated synthetic, with a fully documented SCM, and the `oee_synthetic.py` module ships a `true_oee_decomposition()` helper that returns the *analytic* ground-truth NDE / NIE per mediator. Your capstone is correct iff your estimator recovers those numbers.

**The capstone question.** *A maintenance program was rolled out unevenly across four production lines. The mean OEE rose by ~4 percentage points in the treated shifts; how much of that rise is attributable to each of the three OEE drivers (Availability A, Performance P, Quality Q)?*

**Companion labs.** Lab 10B (mediation NDE/NIE on Backblaze) is the methodological reference; Lab 13B is the sensitivity tooling.

**Why this starter is "verifiable".** Every other capstone starter (A-D, F) operates on real data where the truth is unknown. Starter E ships a documented SCM in `oee_synthetic.py`, so the capstone has a *numerical ground truth* you can validate against. If your sequential-regression NDE/NIE estimator recovers $\\mathrm{NIE}_{\\text{via A}} \\approx 0.035$ within sampling error, you have done mediation correctly. If it doesn't, the disagreement points at a specific bug."""),

md("""## Setup"""),

code("""%pip install -q numpy pandas matplotlib scikit-learn statsmodels"""),

code("""import os, sys, urllib.request, pathlib

DATA = pathlib.Path("/content")
if not (DATA / "oee_synthetic.py").exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/oee_synthetic.py",
        DATA / "oee_synthetic.py",
    )
sys.path.insert(0, str(DATA))

from oee_synthetic import load_oee, true_oee_decomposition
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
df = load_oee(n_shifts=2000, seed=0)
print(df.head(3).round(3).to_string())
print()
print(f'Program rate by line:')
print(df.groupby('line_id')['program'].mean().round(3).to_string())
print()
print('Ground truth from the SCM (so you know what to recover in Q4.2):')
for k, v in true_oee_decomposition().items():
    print(f'  {k}: {v:+.5f}')"""),

md("""**This dataset is curated synthetic.** Unlike the four real-data starters (SECOM, AI4I, LFP, Backblaze), the OEE log is generated from a fully documented structural causal model (the SCM is in `oee_synthetic.py`'s module docstring). That gives us something no real dataset can: an **analytic ground truth** for the NDE/NIE decomposition, exposed via `true_oee_decomposition()`. Your capstone's correctness is *verifiable* — if your estimator recovers those numbers within sampling error, you've done mediation correctly; if it doesn't, the disagreement points at a specific bug (mis-specified mediator regression, missing interaction term, wrong outcome model). Lean on this as a debugging aid — it is the main pedagogical asset of this starter."""),

md("""## Artifact 1 — Problem Statement and DAG

### Q1.1 Write the problem statement.

*Hint.* OEE = A x P x Q is the canonical manufacturing KPI. The decision: knowing the maintenance program raises OEE, the team wants to know *which driver carries most of the gain* so they can target follow-up investment.

*Your turn.*"""),

md("""*[Problem statement.]*"""),

reveal("""A plant rolled out a maintenance program unevenly across four production lines; treated shifts show a ~4-percentage-point higher OEE on average than untreated shifts. Before scaling the program to the entire fleet, the operations team wants to know *which of the three OEE drivers* (Availability A, Performance P, Quality Q) carries most of the effect. If most of the gain runs through A (availability), the next investment should be in similar maintenance programs; if most runs through Q (quality), the next investment is in process-control improvements; if the contributions are evenly distributed, the program does not have a single dominant mechanism and a more diverse follow-up is warranted."""),

md("""### Q1.2 Draw the multi-mediator DAG.

*Your turn.*"""),

md("""*[DAG.]*"""),

reveal("""```
   line_id ──┬──► program       (older lines need maintenance more; back-door confound)
             ├──► A             (per-line baseline differences)
             ├──► P
             └──► Q
                  │
   program ──┬──► A             (the causal effects)
             ├──► P
             └──► Q

   A, P, Q ──► OEE = A * P * Q  (composite KPI; deterministic given A, P, Q)
```

The SCM has NO direct program → OEE arrow; every program → OEE path runs through one of A / P / Q. This makes the analysis a clean *multi-mediator decomposition* — the NDE of program on OEE is zero by construction."""),

md("""### Q1.3 Defend the multi-mediator structure (no direct edge).

*Your turn.*"""),

md("""*[Defense + reference to ground truth.]*"""),

reveal("""The maintenance program does not *directly* set the OEE value — OEE is mechanically determined by the three drivers. The program's effect manifests *only* by changing one or more of A / P / Q. This is documented in `oee_synthetic.true_oee_decomposition()`, which returns NDE = 0 by construction. A capstone that finds NDE significantly different from zero has a bug — most likely an mis-specified mediator model that lets some of the indirect effect leak into the direct path."""),

md("""## Artifact 2 — Estimand

### Q2.1 Write the multi-mediator decomposition estimand.

*Hint.* For a single mediator we have NDE + NIE = TE. With three parallel mediators, the total indirect effect decomposes into per-mediator contributions: TE = NDE + NIE_A + NIE_P + NIE_Q (approximate; ignores higher-order interactions for now).

*Your turn.*"""),

md("""*[Estimand.]*"""),

reveal("""$$\\mathrm{TE} = E[\\mathrm{OEE}(\\text{program} = 1)] - E[\\mathrm{OEE}(\\text{program} = 0)] \\;=\\; \\mathrm{NDE} + \\mathrm{NIE}_A + \\mathrm{NIE}_P + \\mathrm{NIE}_Q,$$

with each $\\mathrm{NIE}_X = E[\\mathrm{OEE}(\\text{program} = 1, X(\\text{program} = 1))] - E[\\mathrm{OEE}(\\text{program} = 1, X(\\text{program} = 0))]$ — the change in OEE from shifting mediator $X$'s distribution from its untreated value to its treated value, with the other two mediators held fixed."""),

md("""### Q2.2 Couple to a decision threshold.

*Your turn.*"""),

md("""*[Decision rule.]*"""),

reveal("""- Largest $\\mathrm{NIE}_X$ exceeds 50% of TE → focus next investment on driver $X$'s mechanism.
- Two NIEs each exceed 30% of TE → mixed mechanism; balance next investment between them.
- All three NIEs roughly equal → program is broadly beneficial without a single dominant mechanism; investment can be flexible."""),

md("""## Artifact 3 — Identification

### Q3.1 Name the strategy and the adjustment set.

*Hint.* For each mediator, the back-door criterion needs to be satisfied between program and the mediator (given line_id). Then the mediation formula propagates.

*Your turn.*"""),

md("""*[Strategy + adjustment set.]*"""),

reveal("""**Strategy.** Sequential-regression mediation analysis (VanderWeele 2015), one mediator at a time. The back-door from program to each driver is blocked by *line_id* (the documented confounder). Identification of NIE_X requires:

1. No unmeasured confounders of program → X given line.
2. No unmeasured confounders of X → OEE given program + line.
3. Consistency.

For this synthetic dataset, all three hold *by construction* — the SCM has no latent confounders. A real OEE dataset would need a sensitivity analysis."""),

md("""### Q3.2 Identify the assumption that would break first on a real OEE log.

*Your turn.*"""),

md("""*[Weakest assumption + how it would break.]*"""),

reveal("""On a real OEE log, **shift-pattern operators** would be the most likely unmeasured confounder of X → OEE. Two operators on the same line at different times produce different A / P / Q profiles, and the same operator's skill drives both higher A (less unplanned downtime via better setup) and higher Q (fewer rework defects). With operators unmeasured, the M-Y confounding inflates the NIE_X estimate.

On this synthetic dataset, the issue does not arise (no operators in the SCM); the capstone treats the synthetic-data exercise as the validation step *before* applying the same machinery to a real log."""),

md("""## Artifact 4 — Estimator

### Q4.1 Compute the cell-level conditional probabilities (P(driver | program, line)) by mediation formula.

*Hint.* For multi-mediator with continuous drivers, the cleanest estimator is regression-based: fit $E[\\mathrm{OEE} \\mid A, P, Q, \\text{program}, \\text{line}]$ and the three mediator regressions $E[X \\mid \\text{program}, \\text{line}]$, then integrate.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
line_dummies = pd.get_dummies(df['line_id'], drop_first=True).astype(float)

# Mediator regressions: each driver on (program, line).
mediator_models = {}
for X in ['A', 'P', 'Q']:
    features = np.column_stack([df['program'].values, line_dummies.values])
    mediator_models[X] = LinearRegression().fit(features, df[X].values)

# Outcome regression: OEE on (A, P, Q, program, line). Include program even though
# its coefficient should be ~0 in this SCM, so we can detect mis-specification.
features = np.column_stack([df['A'].values, df['P'].values, df['Q'].values,
                             df['program'].values, line_dummies.values])
outcome_model = LinearRegression().fit(features, df['OEE'].values)
print(f'Outcome model coefficients on (A, P, Q, program, line dummies):')
for name, val in zip(['A', 'P', 'Q', 'program'] + list(line_dummies.columns), outcome_model.coef_):
    print(f'  {name}: {val:+.4f}')
```
"""),

md("""### Q4.2 Compute NIE_A, NIE_P, NIE_Q by Monte Carlo integration over the line distribution.

*Hint.* For each shift in the dataset, predict OEE under four counterfactuals: program=1 with X at program=1 value vs program=0 value. The difference is NIE_X.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
def predict_driver(X, program_value):
    f = np.column_stack([np.full(len(df), program_value), line_dummies.values])
    return mediator_models[X].predict(f)

A1 = predict_driver('A', 1); A0 = predict_driver('A', 0)
P1 = predict_driver('P', 1); P0 = predict_driver('P', 0)
Q1 = predict_driver('Q', 1); Q0 = predict_driver('Q', 0)

def predict_oee(A_, P_, Q_, program_value):
    f = np.column_stack([A_, P_, Q_, np.full(len(df), program_value), line_dummies.values])
    return outcome_model.predict(f)

# NIE_A: program=1, A varies (A0 vs A1), P and Q stay at A1's value.
NIE_A = (predict_oee(A1, P1, Q1, 1) - predict_oee(A0, P1, Q1, 1)).mean()
NIE_P = (predict_oee(A1, P1, Q1, 1) - predict_oee(A1, P0, Q1, 1)).mean()
NIE_Q = (predict_oee(A1, P1, Q1, 1) - predict_oee(A1, P1, Q0, 1)).mean()

NDE   = (predict_oee(A0, P0, Q0, 1) - predict_oee(A0, P0, Q0, 0)).mean()

print(f'NDE      : {NDE:+.5f}  (true: 0)')
print(f'NIE_A    : {NIE_A:+.5f}  (true: {true_oee_decomposition()[\"NIE_via_A\"]:+.5f})')
print(f'NIE_P    : {NIE_P:+.5f}  (true: {true_oee_decomposition()[\"NIE_via_P\"]:+.5f})')
print(f'NIE_Q    : {NIE_Q:+.5f}  (true: {true_oee_decomposition()[\"NIE_via_Q\"]:+.5f})')

TE_recovered = NDE + NIE_A + NIE_P + NIE_Q
TE_direct = df.loc[df['program'] == 1, 'OEE'].mean() - df.loc[df['program'] == 0, 'OEE'].mean()
print(f'Total: NDE + NIE = {TE_recovered:+.5f}; direct TE diff = {TE_direct:+.5f}')
```
"""),

md("""**Discussion.** This is the validation step that's only possible on a curated dataset. The estimated NIE_A, NIE_P, NIE_Q should each fall within sampling error of the analytic ground truth from `true_oee_decomposition()`. If your numbers are off by more than ~10%, the mediator regressions are mis-specified (e.g., interaction terms missing). If NDE is significantly different from zero, the outcome model is incorrectly attributing indirect effects to the direct path — typically a sign that one of the mediators is mis-modelled."""),

md("""## Artifact 5 — Sensitivity Analysis

### Q5.1 Run a γ-sweep on the A → OEE link (the dominant mediator) to bound unmeasured M-Y confounding.

*Hint.* The γ-sweep (Lab 10B's pattern) parameterises the impact of an unmeasured M-to-Y confounder. Set α = 1 − exp(−γ); attribute that fraction of the *observed* A → OEE association to a hidden confounder (i.e., scale down A's contribution to the outcome by 1 − α); recompute NIE_A under the adjusted A-effect. Sweep γ from 0 (no confounder) to ~4 (very strong confounder); track how NIE_A changes. A robust mediator survives γ ≈ 2 with NIE_A still meaningfully different from zero.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
def NIE_A_under_gamma(gamma):
    alpha = 1.0 - np.exp(-gamma)
    A_coef_adjusted = outcome_model.coef_[0] * (1 - alpha)
    nie_baseline = NIE_A
    # Approximate: scaling A's contribution to outcome scales NIE_A proportionally.
    return nie_baseline * (A_coef_adjusted / outcome_model.coef_[0])

for g in [0.0, 0.5, 1.0, 2.0, 4.0]:
    print(f'gamma = {g}:  NIE_A under unmeasured A-OEE confounder = {NIE_A_under_gamma(g):+.5f}')
```
"""),

md("""### Q5.2 Verdict.

*Your turn.*"""),

md("""*[Verdict on the multi-mediator decomposition's robustness.]*"""),

reveal("""If at $\\gamma = 2$ (a strong but defensible unmeasured A-OEE confounder), the NIE_A remains > 50% of TE, the conclusion *"A is the dominant mediator"* is **robust**. If NIE_A shrinks below 30% of TE by $\\gamma = 1$, the conclusion is **fragile** and a domain expert should weigh in on whether such a confounder is plausible. On this synthetic dataset (no actual unmeasured confounding), the verdict is robust by construction; the γ-sweep is a check the student does to demonstrate they would apply the same machinery on a real log."""),

md("""## Artifact 6 — Deployment-Readiness Checklist

### Q6.1 Target population + monitoring.

*Your turn.*"""),

md("""*[Population + monitors.]*"""),

reveal("""**Target population.** Production shifts on the four lines covered in the dataset, with the same per-line program rollout cadence. Deployment to other product families or other plants requires re-fitting (Lab 14H's transportability machinery).

**Monitors.**
1. **Weekly per-line decomposition recompute.** Alarm if any NIE_X shifts by > 30% relative to the deployment baseline (could indicate a maintenance regime change has altered the program-driver relationship).
2. **Monthly NDE check.** Alarm if NDE drifts > 0.005 OEE points from zero (could indicate the program has acquired a direct effect not running through A/P/Q — for example, by changing operator scheduling).
3. **Quarterly cross-line variance check.** Alarm if the per-line program rate spread exceeds 0.30 (the rollout has become so uneven that the back-door adjustment may not be enough)."""),

md("""## Closing

Variants to extend this capstone:
- Switch to the **doubly-robust mediation estimator** (Tchetgen Tchetgen & Shpitser 2012) instead of the plug-in regression-based decomposition; compare on the same synthetic data.
- Add a **fourth mediator** (e.g., simulate an X-shift in `oee_synthetic.py` for energy efficiency); decompose against it.
- Apply the same decomposition to a **real OEE log** (your own or a sponsor's); use the sensitivity sweep from Q5.1 as the load-bearing assumption check.

The verifiable-ground-truth feature of this dataset is its main pedagogical asset. The same machinery, applied to a real dataset, would have to rely on sensitivity rather than direct validation. Refer to [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the per-artifact standards."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14g_oee.ipynb", cells)
print("Built lab14g_oee.ipynb")
