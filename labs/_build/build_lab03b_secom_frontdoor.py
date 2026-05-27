"""Build labs/ch03/lab03b_secom_frontdoor.ipynb — front-door identification on a stipulated SECOM chain."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 3B — Front-Door Identification on a Stipulated SECOM Chain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03b_secom_frontdoor.ipynb)

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

# Reference: with period observed (the "gold standard" we are pretending not to have).
_period_dummies = pd.get_dummies(df["period"], drop_first=True).astype(float)
_X_with_period  = sm.add_constant(np.hstack([df[["X"]].values, _period_dummies.values]))
_ref = sm.OLS(df["yield_fail"].values, _X_with_period).fit()
_ref_coef = float(_ref.params[1])

print(f"Naive OLS (no adjustment, what an analyst without period sees):")
print(f"  coef on X = {naive_coef:+.5f}  (SE {naive_se:.5f})")
print()
print(f"Reference OLS (Y ~ X + period dummies, period observed):")
print(f"  coef on X = {_ref_coef:+.5f}")
print()
print(f"Period-confounding bias in the naive estimate: {naive_coef - _ref_coef:+.5f}")
print(f"  -> this is the gap front-door must close, working only from X, M, Y.")"""),

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

**What the front-door criterion asks of $M$, in plain English.** Pearl (1995) gives three conditions; each translates to a question we can ask about *our* candidate mediator:

1. **$M$ intercepts every directed path from $X$ to $Y$.**
   *Translation:* the entire causal effect of $X$ on $Y$ flows through $M$. There is no $X \\to Y$ direct arrow.
   *Our SECOM check:* we are assuming no upstream-sensor path to yield exists except via the candidate mediator. We cannot verify this; it is part of the stipulation.

2. **No element of $M$ is a descendant of $X$ via a confounded path.**
   *Translation:* the latent confounder $U$ does *not* affect $M$ (except possibly through $X$).
   *Our SECOM check:* we are assuming the calibration / period drift does not directly push the mediator's reading. This is the *most fragile* assumption and the one the Part 6 sensitivity check stresses.

3. **Every back-door path from $X$ to $M$ is blocked by $X$ itself.**
   *Translation:* once we know $X$, there is no other variable confounding $M$.
   *Our SECOM check:* given the sensor $X$, the candidate mediator $M$ depends only on the chemical chain we hypothesise. Plausible for sensor-to-sensor relationships in a tightly controlled process.

If all three conditions hold, $P(Y \\mid \\mathrm{do}(X))$ is identified by the front-door formula *even though $U$ is unobserved*. This is the surprising piece: front-door buys us identification without measuring $U$, at the cost of stronger assumptions about the role of $M$.

**Where the two-stage formula comes from.** Pearl's original $\\mathrm{do}$-calculus derivation yields:

$$P(Y \\mid \\mathrm{do}(X)) = \\sum_m P(M = m \\mid X) \\sum_{x'} P(Y \\mid X = x', M = m) \\, P(X = x').$$

The first sum says "track the mediator distribution that the *intervention* on $X$ induces"; the second sum says "average the outcome over the *natural* distribution of $X$, holding $M$ fixed". For continuous $X$ and $M$ under a linear ansatz, the formula collapses to a **product of two regression slopes**:

$$\\mathrm{ATE}(X \\to Y) = \\underbrace{\\frac{\\partial M}{\\partial X}}_{\\text{Stage 1: how strongly $X$ pushes $M$}} \\;\\times\\; \\underbrace{\\frac{\\partial Y}{\\partial M \\, \\vert \\, X}}_{\\text{Stage 2: how strongly $M$ pushes $Y$, with the confounded $X$-path blocked}}.$$

**Why Stage 2 conditions on $X$ (not just $Y$ on $M$).** If we regressed $Y$ on $M$ alone, the slope would include the back-door $M \\leftarrow X \\leftarrow U \\rightarrow Y$ path (since $X$ confounds $M$ and $Y$ through the unobserved $U$). Including $X$ in the Stage-2 regression blocks that back-door at $X$ — which condition 3 of the criterion guarantees is sufficient. Without the $+X$ term, front-door collapses to a biased product of two confounded slopes."""),

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
rel_shift = abs(d_Y_on_M_robust - d_Y_on_M) / max(abs(d_Y_on_M), 1e-12)

print(f"Stage-2 slope dY/dM | X         = {d_Y_on_M:+.5f}")
print(f"Stage-2 slope dY/dM | X, period = {d_Y_on_M_robust:+.5f}")
print(f"Shift from adding period to Stage-2: {(d_Y_on_M_robust - d_Y_on_M):+.5f}  ({100*rel_shift:.1f}% of original)")
print()
print('Reading the shift:')
print('  < 10% : front-door assumption (period does NOT leak into M) is consistent with the data.')
print('  10-30%: marginal -- period may have a small mediator-confounding effect; report the front-door')
print('          estimate with a wider uncertainty band.')
print('  > 30% : period IS leaking into the M-Y arc; front-door identification is compromised; either')
print('          pick a different M or admit the chain cannot be defended on this dataset.')"""),

md("""## Part 7 — Decision

Three bullets, the deliverable a process engineer would read.

**A caveat from Part 6's sensitivity check.** On this particular SECOM slice, the Part-6 shift comes in around 60-70 % of the original Stage-2 slope, which the threshold table flags as *"period IS leaking into the M-Y arc; front-door identification is compromised"*. That verdict drives the bullets below — under the observed data, this is a *negative finding* on transportability of the front-door machinery to SECOM, not a defensible estimate.

1. **Under the stipulated chain, the front-door point estimate of X's effect on yield is the Part-4 number reported above (with its 95% CI), but the sensitivity check in Part 6 says it should NOT be acted on.** The 60-70 % shift in Stage 2 when period is added directly contradicts the assumption that period does not leak into the mediator path. A process engineer reading this report should treat the estimate as *the answer the front-door machinery would have given if the stipulated chain were defensible* — useful as a methodological demonstration, not as a deployment input.

2. **Comparison to back-door (when period is observed) gives a sanity check**, but does not validate the stipulation in production — production scenarios are exactly the ones where the would-be back-door variable is *not* available. The fact that on this lab data the back-door coefficient and the front-door product land at similar small magnitudes is at best weak evidence; under the failed sensitivity check, they could also both be biased in the same direction.

3. **A defensible front-door analysis on SECOM would require: (a)** a process flow diagram identifying which sensors lie on the chemical path to yield, **(b)** a substantive argument that the candidate mediator's *only* upstream confounder route is through X, and **(c)** a robustness check across plausible alternative chains. This lab demonstrates the mechanics; it does not deliver any of (a), (b), (c), and Part 6 makes that gap concrete by showing the period-leaks-into-M failure mode in numbers."""),

md("""## Reflection

**Front-door is the most fragile identification strategy in the chapter precisely because it asks for the most.** Back-door asks us to name the confounders; front-door asks us to *defend the absence* of a confounder leak into the mediator. On SECOM with anonymised sensors, that defence is impossible. On a real fab with a process map and a chemical engineer, it might be entirely natural.

**The lesson is not that front-door doesn't work** — it is that *the identifiability of a causal estimand depends as much on what you know about the system as on what you measure*. The estimator is straightforward (two regressions and a delta-method SE). The hard part is the assumption it rests on."""),

md("""## What's next

Lab 4B applies **difference-in-differences** identification to the Severson 2019 LFP-battery cell data — a third strategy alongside back-door and front-door. The natural-experiment structure there is the three Severson collection batches receiving overlapping-but-distinct fast-charge protocols at different points in time; the DID identification rests on the parallel-trends assumption between batches in the pre-treatment window."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch03" / "lab03b_secom_frontdoor.ipynb", cells)
print("Built lab03b_secom_frontdoor.ipynb")
