"""Build labs/ch03/lab03b.ipynb — front-door identification on a stipulated SECOM chain."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 3B — Front-Door Identification on a Stipulated SECOM Chain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03b.ipynb)

**Companion to Lab 3A.** Lab 3A built a synthetic three-variable chain X → M → Y with a latent confounder U → {X, Y}, applied the front-door formula, and verified it recovers the true ATE while back-door adjustment on the observed covariates does not. **Lab 3B asks: when do we have any business invoking front-door on real data, and what does the estimate look like when we do?**

The honest answer for SECOM: we cannot. The 590 sensors are anonymised; there is no published process schematic that tells us which sensor is upstream of which. **This lab STIPULATES a chain** — picks two sensors that, on the surface, plausibly form an X → M → Y path — and runs the front-door machinery as if that stipulation were true. The deliverable is the analysis under the stipulation, plus an explicit statement of what would be needed to defend the chain in production (process knowledge we do not have)."""),

md("""## What this lab is *not* doing

- **Justifying the chain from physics.** We pick X and M algorithmically (top |corr with yield|; sensor most correlated with X). That is *not* a defensible front-door justification — it is a placeholder so the lab can exercise the formula. The lab's final bullet names what a defensible justification would require.
- **Handling continuous-mediator complications.** Both X and M are continuous; we use the two-stage regression form of front-door, which is exact under linearity and approximate under non-linearity.
- **Comparing many sensor triples.** A more thorough analysis would search over plausible chains and report sensitivity to the chain choice. We use one chain to keep the lab focused on the mechanics."""),

code("""%pip install -q ucimlrepo dowhy"""),

code("""import os, sys, urllib.request, pathlib

PREP_PATH = pathlib.Path("/content/secom_prep.py")
if not PREP_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/secom_prep.py",
        PREP_PATH,
    )
sys.path.insert(0, str(PREP_PATH.parent))

from secom_prep import load_secom

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load the stipulated chain

The Ch 3 slice picks X = top sensor by |corr with yield|, then M = sensor most correlated with X among the rest, plus `period` and `yield_fail`. The original sensor IDs are recorded in `df.attrs` for traceability."""),

code("""df = load_secom(chapter=3)

print(f"Stipulated upstream sensor X:   {df.attrs['x_sensor']}  (renamed to 'X')")
print(f"Stipulated mediator sensor M:   {df.attrs['m_sensor']}  (renamed to 'M')")
print()
print(f"Sample size:                    {len(df)}")
print(f"Failure rate:                   {df['yield_fail'].mean():.3%}")
print(f"corr(X, M):                     {df['X'].corr(df['M']):+.3f}")
print(f"corr(X, yield_fail):            {df['X'].corr(df['yield_fail']):+.3f}")
print(f"corr(M, yield_fail):            {df['M'].corr(df['yield_fail']):+.3f}")"""),

md("""## Part 2 — The trap: pretend `period` is unobserved

We saw in Lab 1B that `period` confounds every sensor → yield relationship. Here we *pretend* we cannot measure period (suppose its raw timestamp wasn't logged). What does a naive analyst conclude from the X→Y correlation?"""),

code("""# Naive OLS of yield_fail on X, no adjustment.
naive = sm.OLS(df["yield_fail"].values,
               sm.add_constant(df[["X"]].values)).fit()
naive_coef = float(naive.params[1])
naive_se   = float(naive.bse[1])

print(f"Naive OLS (no adjustment):")
print(f"  coef on X = {naive_coef:+.5f}  (SE {naive_se:.5f})")
print(f"  Interpretation: a 1-unit increase in X is associated with")
print(f"  a {naive_coef:+.5%} change in failure probability.")"""),

md("""**The trap.** Without `period`, this regression cannot distinguish three explanations:

1. X *causes* failure (X → Y).
2. X *correlates with* a calibration cycle / supplier rotation / seasonal effect that also drives failure (U → X, U → Y).
3. Some combination of both.

If the team had period, they would block the back-door with a stratified regression (Lab 1B). They do not. The question is: **does the front-door criterion let us identify the X→Y effect using only X, M, Y?**"""),

md("""## Part 3 — The stipulated DAG and the front-door criterion

```
   U (period/lot, UNOBSERVED) ──┬──► X
                                └──► Y

   X ────► M ────► Y

   (assumption: U does NOT affect M except through X)
```

The front-door criterion (Pearl 1995) says: if a set of nodes M *intercepts* the entire directed path from X to Y, *no element of M is a descendant of X via a confounded path*, and *every back-door from X to M is blocked by X itself*, then $P(Y \\mid do(X))$ is identified by the front-door formula even when U is unobserved.

**For our stipulated chain:** the assumption that lets us proceed is *U → M is blocked by X*. Concretely, this means the period-driven mechanism that pushes both X and Y up or down does *not* leak into M except by first changing X. That assumption is *not testable from the data* — it must be argued from process knowledge.

If we accept the stipulation, the **front-door formula** for a continuous chain (linearised) is:

$$\\text{ATE}(X \\to Y) = \\frac{\\partial M}{\\partial X} \\cdot \\frac{\\partial Y}{\\partial M\\,\\vert\\,X}$$

— the indirect effect through M, computed as the product of two regression slopes."""),

md("""## Part 4 — Front-door estimate via two-stage regression"""),

code("""# Stage 1: regress M on X. Slope = dE[M|X]/dX.
stage1 = sm.OLS(df["M"].values, sm.add_constant(df[["X"]].values)).fit()
b_M_on_X    = float(stage1.params[1])
b_M_on_X_se = float(stage1.bse[1])

# Stage 2: regress Y on M, adjusting for X (blocks the X-mediated back-door from M to Y).
stage2 = sm.OLS(df["yield_fail"].values,
                sm.add_constant(df[["M", "X"]].values)).fit()
d_Y_on_M    = float(stage2.params[1])
d_Y_on_M_se = float(stage2.bse[1])

ate_front = b_M_on_X * d_Y_on_M

# Delta-method SE for the product (treating cov(stage1, stage2) ~ 0 for simplicity).
ate_front_se = np.sqrt(
    (d_Y_on_M  * b_M_on_X_se) ** 2 +
    (b_M_on_X * d_Y_on_M_se) ** 2
)

print(f"Stage 1  dM/dX     = {b_M_on_X:+.5f}  (SE {b_M_on_X_se:.5f})")
print(f"Stage 2  dY/dM | X = {d_Y_on_M:+.5f}  (SE {d_Y_on_M_se:.5f})")
print()
print(f"Front-door ATE      = {ate_front:+.5f}  (SE {ate_front_se:.5f})")
print(f"95% CI              = [{ate_front - 1.96*ate_front_se:+.5f}, {ate_front + 1.96*ate_front_se:+.5f}]")"""),

md("""## Part 5 — Compare three estimates

We can now line up:

1. **Naive** — `Y ~ X`, ignoring everything. This is what an analyst without the front-door insight would report.
2. **Front-door** — the Part 4 product. This is what front-door says under the stipulation.
3. **Cheating back-door** — `Y ~ X + period dummies`. This uses the observation we pretended was unavailable; we look at it as the *target* the front-door estimate should approximate, *if* the stipulated chain were correct."""),

code("""period_dummies = pd.get_dummies(df["period"], drop_first=True).astype(float)
X_aug = sm.add_constant(np.hstack([df[["X"]].values, period_dummies.values]))
back  = sm.OLS(df["yield_fail"].values, X_aug).fit()
back_coef = float(back.params[1])
back_se   = float(back.bse[1])

comparison = pd.DataFrame([
    {"estimator": "Naive (Y ~ X)",                 "coef": naive_coef,  "se": naive_se},
    {"estimator": "Front-door (stipulated chain)", "coef": ate_front,   "se": ate_front_se},
    {"estimator": "Back-door (period observed)",   "coef": back_coef,   "se": back_se},
])
print(comparison.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))"""),

md("""**How to read the comparison.**

- If **front-door ≈ back-door**, the stipulated chain is consistent with the period-adjusted gold standard. We have evidence (under the stipulation) that the front-door machinery would have rescued us if period were truly unobserved.
- If **front-door ≈ naive** but very different from back-door, the stipulated chain failed to block the confounding — most likely because U → M is not in fact zero (the chain assumption was wrong). The front-door estimate is leaking back-door bias.
- If **front-door diverges from both**, the linearisation may be inadequate (non-linear M response), or M is a weak mediator (low $\\partial M/\\partial X$ means the front-door product is small even when the true effect isn't)."""),

md("""## Part 6 — Sensitivity: what about the stipulation itself?

The front-door identification rests on *one* assumption — that period (or any other unobserved confounder) does not affect M except through X. This is *not statistically testable from the data*. The right sensitivity exercise is to ask: **how strongly would period have to leak into M to invalidate the conclusion?**

A pragmatic proxy: re-run Stage 2 with period dummies added. If the slope $\\partial Y/\\partial M\\,\\vert\\,X$ shifts substantially when we control for period, then period is leaking into the M→Y arc, which contradicts the front-door assumption. If it barely moves, the assumption is at least consistent with the data we have."""),

code("""stage2_with_period = sm.OLS(
    df["yield_fail"].values,
    sm.add_constant(np.hstack([df[["M", "X"]].values, period_dummies.values])),
).fit()
d_Y_on_M_robust = float(stage2_with_period.params[1])

print(f"Stage-2 slope dY/dM | X         = {d_Y_on_M:+.5f}")
print(f"Stage-2 slope dY/dM | X, period = {d_Y_on_M_robust:+.5f}")
print(f"Shift from adding period to Stage-2: {(d_Y_on_M_robust - d_Y_on_M):+.5f}")
print()
print(f"If the shift is small (<10% of the original slope), the front-door")
print(f"assumption (period does not leak into M) is at least consistent with")
print(f"the data. If it is large, period IS leaking and front-door identification")
print(f"is compromised.")"""),

md("""## Part 7 — Decision

Three bullets, the deliverable a process engineer would read:

1. **Under the stipulated chain, the front-door estimate of X's effect on yield is X.X (95% CI [...])**. This number is *only as defensible as the chain itself*: a process engineer must look at sensor IDs `{df.attrs['x_sensor']}` and `{df.attrs['m_sensor']}` and confirm or reject the proposed physical path. If they reject it, the estimate is meaningless and should be withdrawn.

2. **Comparison to back-door (when period is observed) gives a sanity check**, but does not validate the stipulation in production — production scenarios are exactly the ones where the would-be back-door variable is *not* available, so the comparison only tells us about the lab setting.

3. **A defensible front-door analysis on SECOM would require: (a)** a process flow diagram identifying which sensors lie on the chemical path to yield, **(b)** a substantive argument that the candidate mediator's *only* upstream confounder route is through X, and **(c)** a robustness check across plausible alternative chains. This lab demonstrates the mechanics; it does not deliver any of (a), (b), (c)."""),

md("""## Reflection

**Front-door is the most fragile identification strategy in the chapter precisely because it asks for the most.** Back-door asks us to name the confounders; front-door asks us to *defend the absence* of a confounder leak into the mediator. On SECOM with anonymised sensors, that defence is impossible. On a real fab with a process map and a chemical engineer, it might be entirely natural.

**The lesson is not that front-door doesn't work** — it is that *the identifiability of a causal estimand depends as much on what you know about the system as on what you measure*. The estimator is straightforward (two regressions and a delta-method SE). The hard part is the assumption it rests on."""),

md("""## What's next

Lab 4B uses instrumental-variable identification — a third strategy alongside back-door and front-door — applied to Bosch Production Line Performance data, where a phased calibration rollout creates the IV structure."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch03" / "lab03b.ipynb", cells)
print("Built lab03b.ipynb")
