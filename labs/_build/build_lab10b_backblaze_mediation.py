"""Build labs/ch10/lab10b_backblaze_mediation.ipynb — mediation analysis on Backblaze SMART telemetry."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 10B — Mediation: Reallocated → Pending Sectors → Drive Failure

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch10/lab10b_backblaze_mediation.ipynb)

**Companion to Lab 10A.** Lab 10A built a multi-stage wafer-flow SCM with an explicit recanting witness and showed how the *mediation formula* and the doubly-robust mediation estimator behave on a controlled synthetic problem. **Lab 10B asks the same question on real hard-drive telemetry: of the total effect that early SMART warnings have on failure, how much is *mediated* by the next-stage warning, and how much is *direct*?**

The chain we test is the canonical Backblaze degradation path:

$$\\text{SMART\\_5 (Reallocated Sectors)} \\;\\to\\; \\text{SMART\\_197 (Pending Sectors)} \\;\\to\\; \\text{Drive Failure}$$

A reallocated sector means the firmware retired a bad block at write time. A *pending* sector means the firmware found a block it cannot read but hasn't been able to swap yet — the canonical "death rattle". The natural causal story is that reallocations *cause* pending sectors (the underlying physical wear keeps producing bad blocks faster than the spare-block pool can absorb them), and pending sectors *cause* observed failures. If that mediation story were the whole story, controlling for SMART_197 should explain away SMART_5's effect on failure. We will see that it does not, and the residual direct effect is the lesson.

**Dataset.** `labs/data/backblaze_subset.csv` — 1576 ST4000DM000 drives observed for ~30 days in early 2016, with 76 observed failures (drive-level failure rate ~4.8 %). See `labs/data/README.md` for the preprocessing recipe.

**Estimand.** Decompose the total effect of treatment $X$ (SMART_5 ever non-zero in the window) on outcome $Y$ (drive failed in the window) into the *natural direct effect* (NDE — the part of $X$'s effect that does not flow through the mediator $M$ = SMART_197) and the *natural indirect effect* (NIE — the part that flows through $M$):

$$\\mathrm{TE} \\;=\\; \\mathrm{NDE} \\;+\\; \\mathrm{NIE}.$$"""),

md("""## What this lab is *not* doing

- **Claiming a one-line causal story.** Drives that reallocate sectors are *not* a random subset of drives. Manufacturing defects, controller-firmware vintage, and physical operating conditions all confound the picture. We will report the NDE/NIE under the *no-unmeasured-confounding* assumption and then test sensitivity to that assumption.
- **Predicting individual drive lifetimes.** We are decomposing a population-level association, not building a per-drive failure predictor.
- **Reproducing the chapter's recanting-witness machinery.** Backblaze has one mediator, not two, so there is no recanting witness here. The point of 10B is to put NDE / NIE on real telemetry and see what the *direct* path looks like when the obvious mediator is taken away."""),

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
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load and inspect the X / M / Y table"""),

code("""df = load_backblaze(chapter=10)
print(f"Shape: {df.shape}")
print(f"  N drives observed: {len(df)}")
print(f"  X (SMART_5 ever > 0):    {int(df['X'].sum())} drives  ({df['X'].mean():.1%})")
print(f"  M (SMART_197 ever > 0):  {int(df['M'].sum())} drives  ({df['M'].mean():.1%})")
print(f"  Y (drive failed):        {int(df['Y'].sum())} drives  ({df['Y'].mean():.1%})")
print()
print("Joint distribution (X, M, Y):")
print(pd.crosstab([df['X'], df['M']], df['Y'], rownames=['X','M'], colnames=['Y'], margins=True))"""),

md("""**Read the table.** The (X=0, M=0) cell — drives with no observed sector damage — has a baseline failure rate just under 2 %, the actuarial background. The (X=1, M=1) cell — drives with the full diagnostic chain — has a failure rate of 100 % in this window (every drive in that cell died). The middle cells matter the most for mediation: drives with reallocations but no pending sectors (X=1, M=0) still fail at 4/7 = 57 %, far above background, *which already tells us the direct path is large*."""),

md("""## Part 2 — Total effect (TE)

The total effect of $X$ on $Y$ on the risk-difference scale:

$$\\mathrm{TE} \\;=\\; P(Y = 1 \\mid X = 1) \\,-\\, P(Y = 1 \\mid X = 0).$$

This is what an analyst who *ignored* the mediator would report — the marginal association between SMART_5 and failure."""),

code("""p_y1_given_x1 = df.loc[df['X'] == 1, 'Y'].mean()
p_y1_given_x0 = df.loc[df['X'] == 0, 'Y'].mean()
TE = p_y1_given_x1 - p_y1_given_x0

print(f"P(Y=1 | X=1)         = {p_y1_given_x1:.4f}")
print(f"P(Y=1 | X=0)         = {p_y1_given_x0:.4f}")
print(f"Total effect (TE)    = {TE:+.4f}")
print(f"Relative risk        = {p_y1_given_x1 / p_y1_given_x0:.1f}x")"""),

md("""## Part 3 — Mediation formula (plug-in)

**The "two worlds, three scenarios" intuition.** Before writing formulas, here is the counterfactual framing the formulas are operationalising. Pearl's (2001) decomposition imagines *three* parallel worlds for each drive:

| Scenario | What we set | What $M$ does | Outcome we observe |
|----------|-------------|---------------|--------------------|
| World A (untreated) | $X = 0$ | $M$ takes its natural $X=0$ value: $M(0)$ | $Y(0, M(0))$ |
| World B (treated) | $X = 1$ | $M$ takes its natural $X=1$ value: $M(1)$ | $Y(1, M(1))$ |
| Hybrid (counterfactual) | $X = 1$ | $M$ is *forced* to its $X=0$ value: $M(0)$ | $Y(1, M(0))$ |

The total effect is World B − World A: $\\mathrm{TE} = E[Y(1, M(1))] - E[Y(0, M(0))]$.

We decompose this by inserting the **Hybrid** in the middle:

- **Natural direct effect (NDE)** = Hybrid − World A = $E[Y(1, M(0))] - E[Y(0, M(0))]$ — the change in $Y$ from flipping $X$ alone, with the mediator pinned at its untreated value. *"What is the effect of treatment that goes around the mediator?"*
- **Natural indirect effect (NIE)** = World B − Hybrid = $E[Y(1, M(1))] - E[Y(1, M(0))]$ — the change in $Y$ from letting the mediator move from $M(0)$ to $M(1)$, with $X$ held at 1. *"What is the effect of treatment that goes through the mediator?"*

By construction, $\\mathrm{NDE} + \\mathrm{NIE} = \\mathrm{TE}$ — the Hybrid term cancels.

**Why the Hybrid world is observable from data even though we can't run it.** Under the three no-unmeasured-confounding assumptions (X-Y, M-Y, X-M all unconfounded given observed covariates — see Part 6), $E[Y(1, M(0))]$ equals an average of the *observed* $E[Y \\mid X = 1, M = m]$ weighted by the *observed* distribution of $M$ at $X = 0$. That averaging is what the formulas below compute.

**The formulas.**

$$\\mathrm{NDE} \\;=\\; \\sum_{m} \\underbrace{\\big[ P(Y=1 \\mid X=1, M=m) - P(Y=1 \\mid X=0, M=m) \\big]}_{\\text{$X$ effect on $Y$ at fixed $m$}} \\, \\underbrace{P(M=m \\mid X=0)}_{\\text{weight from untreated $M$ dist}},$$

$$\\mathrm{NIE} \\;=\\; \\sum_{m} \\underbrace{P(Y=1 \\mid X=1, M=m)}_{\\text{$Y$ at treated $X$, fixed $m$}} \\, \\underbrace{\\big[ P(M=m \\mid X=1) - P(M=m \\mid X=0) \\big]}_{\\text{shift in $M$ dist from $X = 0$ to $X = 1$}}.$$

For binary $X$ and binary $M$ each formula has two terms (one for $m = 0$, one for $m = 1$) and is exact cell-arithmetic — no model required.

**Worked tiny example.** Imagine a population where $P(M=1 \\mid X=0) = 0.1$, $P(M=1 \\mid X=1) = 0.6$ (treatment really pushes the mediator), and the outcome probabilities are: $P(Y=1 \\mid X=0, M=0) = 0.05$, $P(Y=1 \\mid X=0, M=1) = 0.30$, $P(Y=1 \\mid X=1, M=0) = 0.20$, $P(Y=1 \\mid X=1, M=1) = 0.45$.

- $\\mathrm{NDE} = (0.20 - 0.05) \\times 0.9 + (0.45 - 0.30) \\times 0.1 = 0.135 + 0.015 = 0.150$.
- $\\mathrm{NIE} = 0.20 \\times (0.4 - 0.9) + 0.45 \\times (0.6 - 0.1) = -0.100 + 0.225 = 0.125$.
- Total = 0.150 + 0.125 = **0.275**, which matches $E[Y(1)] - E[Y(0)]$ computed directly.

The code below does exactly this arithmetic on our Backblaze cells."""),

code("""def cell_prob(df, x, m, target):
    sub = df[(df['X'] == x) & (df['M'] == m)]
    if len(sub) == 0:
        return np.nan
    return float(sub[target].mean())

def m_marginal(df, x):
    return float(df[df['X'] == x]['M'].mean())

# Outcome conditional probabilities P(Y=1 | X, M)
py_x1m0 = cell_prob(df, 1, 0, 'Y')
py_x1m1 = cell_prob(df, 1, 1, 'Y')
py_x0m0 = cell_prob(df, 0, 0, 'Y')
py_x0m1 = cell_prob(df, 0, 1, 'Y')

# Mediator marginal P(M=1 | X)
pm1_x0 = m_marginal(df, 0)
pm1_x1 = m_marginal(df, 1)

print(f"P(Y=1 | X=0, M=0) = {py_x0m0:.4f}")
print(f"P(Y=1 | X=0, M=1) = {py_x0m1:.4f}")
print(f"P(Y=1 | X=1, M=0) = {py_x1m0:.4f}")
print(f"P(Y=1 | X=1, M=1) = {py_x1m1:.4f}")
print()
print(f"P(M=1 | X=0)      = {pm1_x0:.4f}")
print(f"P(M=1 | X=1)      = {pm1_x1:.4f}")
print()

# NDE: average X effect on Y over M's X=0 distribution
NDE = (py_x1m0 - py_x0m0) * (1 - pm1_x0) + (py_x1m1 - py_x0m1) * pm1_x0
# NIE: average outcome at X=1 over M's shift from X=0 to X=1
NIE = py_x1m0 * ((1 - pm1_x1) - (1 - pm1_x0)) + py_x1m1 * (pm1_x1 - pm1_x0)

print(f"NDE (plug-in)        = {NDE:+.4f}")
print(f"NIE (plug-in)        = {NIE:+.4f}")
print(f"NDE + NIE            = {NDE + NIE:+.4f}   (should equal TE = {TE:+.4f})")
print()
print(f"Share mediated through M (NIE/TE)  = {100 * NIE / TE:.1f}%")
print(f"Share direct (NDE/TE)              = {100 * NDE / TE:.1f}%")"""),

md("""**Read the decomposition.** A majority of SMART_5's effect on failure is *direct* — it does not flow through SMART_197. Mechanistically that makes sense: a drive reallocating sectors has been writing to bad blocks faster than the spare pool can keep up. The drive's hardware is *already* compromised at the moment we observe a non-zero SMART_5; subsequent block-read failures (which is what SMART_197 measures) are a downstream symptom, but they are not the *cause* of the eventual failure. The cause is the underlying mechanical wear, of which both SMART_5 and SMART_197 are markers."""),

md("""## Part 4 — Regression-based NDE / NIE

The plug-in formulas are exact for binary variables but break down with continuous mediators or covariates. The regression-based estimator generalises by fitting models for $E[Y \\mid X, M]$ and $E[M \\mid X]$ and computing the expectations by integration (or simulation).

We will use logistic regression for both the outcome and mediator models, integrating out $M$ analytically since it is binary."""),

code("""# Fit logistic regression: Y on (X, M) with X*M interaction so the cell probabilities are saturated.
# Use C=1e6 (essentially no regularization) so the regression recovers cell probabilities at small sample.
df_fit = df.assign(XM=df['X'] * df['M'])
X_y = df_fit[['X', 'M', 'XM']].values
y = df_fit['Y'].values
y_model = LogisticRegression(C=1e6, max_iter=2000).fit(X_y, y)

def py(x, m):
    return float(y_model.predict_proba([[x, m, x * m]])[0, 1])

# Fit logistic regression: M on X
X_m = df[['X']].values
m = df['M'].values
m_model = LogisticRegression(C=1e6, max_iter=2000).fit(X_m, m)

def pm(x):
    return float(m_model.predict_proba([[x]])[0, 1])

# Plug into NDE / NIE formulas via integration over M
NDE_reg = (py(1, 0) - py(0, 0)) * (1 - pm(0)) + (py(1, 1) - py(0, 1)) * pm(0)
NIE_reg = py(1, 0) * ((1 - pm(1)) - (1 - pm(0))) + py(1, 1) * (pm(1) - pm(0))

print(f"NDE (regression)     = {NDE_reg:+.4f}")
print(f"NIE (regression)     = {NIE_reg:+.4f}")
print(f"NDE + NIE            = {NDE_reg + NIE_reg:+.4f}")
print()
print("Side-by-side with plug-in:")
print(f"  NDE plug-in: {NDE:+.4f}   NDE regression: {NDE_reg:+.4f}")
print(f"  NIE plug-in: {NIE:+.4f}   NIE regression: {NIE_reg:+.4f}")"""),

md("""The plug-in and regression numbers agree closely; for binary $X$ and $M$ with no extra covariates, the two estimators are essentially the same, modulo logistic regression's L2 penalty pulling cell probabilities very slightly toward the marginal. The point of fitting the regressions is to set up the next two parts, where we add bootstrap CIs and a sensitivity analysis."""),

md("""## Part 5 — Bootstrap CIs

The plug-in NDE / NIE are point estimates; we need uncertainty. A nonparametric bootstrap over drives gives a confidence interval that automatically accounts for the small sample sizes in the cells driving the decomposition (the X=1 cells have just 18 drives total)."""),

code("""def nde_nie_for(sample):
    # Cell probabilities (NaN-safe)
    def cp(x, m, target):
        sub = sample[(sample['X'] == x) & (sample['M'] == m)]
        return float(sub[target].mean()) if len(sub) else np.nan
    py00 = cp(0, 0, 'Y'); py01 = cp(0, 1, 'Y')
    py10 = cp(1, 0, 'Y'); py11 = cp(1, 1, 'Y')
    pm0  = float(sample[sample['X']==0]['M'].mean())
    pm1  = float(sample[sample['X']==1]['M'].mean())
    # If any cell is empty (no drives with that X,M combo), fall back to 0 for the
    # missing outcome probability; this is a finite-sample artifact.
    py00 = 0.0 if np.isnan(py00) else py00
    py01 = 0.0 if np.isnan(py01) else py01
    py10 = 0.0 if np.isnan(py10) else py10
    py11 = 0.0 if np.isnan(py11) else py11
    nde = (py10 - py00) * (1 - pm0) + (py11 - py01) * pm0
    nie = py10 * ((1 - pm1) - (1 - pm0)) + py11 * (pm1 - pm0)
    te  = nde + nie
    return nde, nie, te

B = 500
n = len(df)
boot = np.empty((B, 3))
for b in range(B):
    idx = rng.integers(0, n, size=n)
    boot[b] = nde_nie_for(df.iloc[idx])

names = ['NDE', 'NIE', 'TE']
print(f"Bootstrap (B={B}) 95% CIs:")
for i, name in enumerate(names):
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    print(f"  {name}: {boot[:, i].mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")"""),

md("""## Part 6 — Sensitivity to unmeasured $M$-$Y$ confounding

The NDE / NIE identifiability rests on three no-unmeasured-confounding assumptions:

1. No unmeasured confounder of $X$ on $Y$ (defensible: the SMART telemetry plus the single-model filter remove most of the drive-family variation we worry about).
2. No unmeasured confounder of $M$ on $Y$ (the question of this section).
3. No unmeasured confounder of $X$ on $M$.

The $M$-$Y$ assumption is the most vulnerable. There may be an unobserved drive-level state $U$ — physical operating temperature, age, vibration history — that drives *both* the SMART_197 reading and the eventual failure even at fixed SMART_5. If so, the observed (M, Y) association inflates the *causal* M→Y effect, and the NDE shrinks toward TE while the NIE shrinks toward zero.

A simple sensitivity model (Imai, Keele, Yamamoto 2010): assume $U$ enters the outcome model linearly with effect $\\gamma$ on the logit scale, and adjusts $P(Y \\mid X, M)$ accordingly. We sweep $\\gamma$ and trace how NDE / NIE move.

**Anchoring $\\gamma$ to something concrete.** $\\gamma$ is the log-odds effect of the unmeasured $U$ on $Y$; we re-parametrise it via $\\alpha = 1 - e^{-\\gamma}$, which is the *fraction of the observed M=1 vs M=0 excess in $Y$-rate that we are attributing to $U$ rather than to $M$*:

- $\\gamma = 0$ ($\\alpha = 0$): no unmeasured confounding; recover the plug-in.
- $\\gamma = 1$ ($\\alpha = 0.63$): a moderately strong $U$ that explains ~⅔ of the M=1 vs M=0 excess in failure rate.
- $\\gamma = 2$ ($\\alpha = 0.86$): a very strong $U$ — would double the odds of Y (an effect size that would itself be diagnosable by domain experts).
- $\\gamma = 4$ ($\\alpha = 0.98$): essentially attribute the entire M=1 vs M=0 outcome gap to $U$, leaving $M$ with almost no causal role.

For SMART telemetry on a single drive model, $\\gamma > 2$ is implausible — drive-firmware vintage, temperature, vibration would each have to be doing the work that pending sectors visibly do, and they would have been spotted by Backblaze's reliability team."""),

code("""def sensitivity_sweep(df, gamma_range):
    \"\"\"Re-compute NDE/NIE assuming an unmeasured U with effect gamma on Y.

    Model: a fraction alpha = 1 - exp(-gamma) of the observed P(Y | X, M=1)
    is attributable to U rather than to M. At gamma = 0, alpha = 0 (no
    adjustment, recover plug-in). At gamma = infinity, alpha = 1 (the entire
    M = 1 excess Y-rate is U, not M).
    \"\"\"
    out = []
    for g in gamma_range:
        alpha = 1.0 - np.exp(-g)
        # Attribute fraction alpha of the (M=1 - M=0) Y excess to U.
        adj_py01 = py_x0m1 - alpha * (py_x0m1 - py_x0m0)
        adj_py11 = py_x1m1 - alpha * (py_x1m1 - py_x1m0)
        nde = (py_x1m0 - py_x0m0) * (1 - pm1_x0) + (adj_py11 - adj_py01) * pm1_x0
        nie = py_x1m0 * ((1 - pm1_x1) - (1 - pm1_x0)) + adj_py11 * (pm1_x1 - pm1_x0)
        out.append((g, nde, nie, nde + nie))
    return pd.DataFrame(out, columns=['gamma', 'NDE', 'NIE', 'NDE+NIE'])

sens = sensitivity_sweep(df, np.linspace(0, 4, 9))
print(sens.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))"""),

code("""fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sens['gamma'], sens['NDE'], 'o-', label='NDE', color='C0')
ax.plot(sens['gamma'], sens['NIE'], 's-', label='NIE', color='C1')
ax.plot(sens['gamma'], sens['NDE+NIE'], '^--', label='NDE+NIE', color='C2', alpha=0.6)
ax.axhline(TE, color='k', linestyle=':', alpha=0.5, label=f'TE = {TE:.3f}')
ax.set_xlabel(r'$\\gamma$ — assumed log-odds effect of unmeasured U on Y')
ax.set_ylabel('Effect size')
ax.set_title('Sensitivity of NDE / NIE to unmeasured $M$–$Y$ confounding')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""),

md("""**Read the curve.** At $\\gamma = 0$ (no unmeasured confounding), we recover the plug-in NDE and NIE. As we let an unmeasured $U$ explain more of the $M$-$Y$ association, the NIE (the *mediated* effect) shrinks toward zero and the NDE absorbs the difference. The qualitative conclusion — that the majority of the effect is direct — is robust: even at $\\gamma = 4$ (a strong residual confounder, more than reasonable for SMART telemetry on a single drive model), the NIE remains positive and the NDE remains the larger component.

This is the standard *direction*-of-bias argument: if there is residual M-Y confounding, we are *over-stating* the NIE, which only strengthens the conclusion that the direct path dominates."""),

md("""## Part 7 — Controlled direct effect

The NDE marginalises over $M$'s untreated distribution. A simpler sanity-check is the *controlled direct effect* (CDE) — the effect of $X$ on $Y$ at a *fixed* value of $M$:

$$\\mathrm{CDE}(m) \\;=\\; P(Y = 1 \\mid X = 1, M = m) - P(Y = 1 \\mid X = 0, M = m).$$

If CDE(M=0) — the effect of SMART_5 on failure *among drives with no pending sectors* — is substantial, the direct path is real. If it is near zero, the entire effect must run through $M$.

**Anchoring the CDE magnitude on the risk-difference scale.** The baseline failure rate in the (X=0, M=0) cell is ~2% (the actuarial background). A CDE of:

- **≤ +5 percentage points** would be small — comparable to noise in a typical drive cohort and would be consistent with "all of $X$'s effect runs through $M$".
- **+5 to +20 percentage points** is moderate — there is a real direct path, but $M$ remains the dominant route.
- **+20 to +50 percentage points** is large — the direct path is the dominant route (this is what we expect for SMART_5 → failure in our Backblaze subset).
- **+50 percentage points or more** is enormous in absolute terms — the chain hypothesis is essentially broken; $M$ may be a marker, not a mechanism."""),

code("""CDE_m0 = py_x1m0 - py_x0m0
CDE_m1 = py_x1m1 - py_x0m1

print(f"CDE at M=0 (no pending sectors):   P(Y|X=1,M=0) - P(Y|X=0,M=0) = {py_x1m0:.4f} - {py_x0m0:.4f} = {CDE_m0:+.4f}")
print(f"CDE at M=1 (with pending sectors): P(Y|X=1,M=1) - P(Y|X=0,M=1) = {py_x1m1:.4f} - {py_x0m1:.4f} = {CDE_m1:+.4f}")
print()
print("Interpretation:")
print(f"  Among drives that DID NOT develop pending sectors, having seen reallocations")
print(f"  raises failure probability by {CDE_m0:+.1%} -- a direct effect that cannot run through M.")"""),

md("""## Part 8 — Decision

Three bullets, the maintenance-engineering takeaway:

1. **The total association between SMART_5 and failure in this window is large** (TE ≈ +0.79 on the risk-difference scale, ~20× relative risk). A drive observed to have reallocated *any* sectors is, in our subset, ~21× more likely to fail in the next 30 days than a drive with zero reallocations.

2. **Most of that association is *direct*, not mediated by SMART_197**. The plug-in NDE / NIE split is roughly two-thirds direct, one-third mediated. The CDE at M=0 confirms: even drives that *never* developed pending sectors still failed at 57 % when SMART_5 was non-zero. SMART_197 is a strong marker but not the mechanism — both attributes share an upstream physical cause.

3. **Operational consequence: do not wait for SMART_197 to act on SMART_5.** A policy that ignores reallocated-sector alerts unless pending sectors also appear would miss most of the failures we observed, because most of the X→Y signal is direct. SMART_5 is itself an actionable trigger; SMART_197 confirms the diagnosis but is not on the critical path."""),

md("""## Reflection

**The mediation formula does *not* tell us "drive failure is caused by SMART_5".** It tells us how the *observed effect* of one diagnostic indicator on the outcome decomposes given a hypothesised mediator. The chapter's recanting-witness story warned us that mediators which are themselves on multiple causal paths cannot cleanly assign blame. Here, with a single mediator and no recanting witness, we get a clean decomposition — and the lesson is that this single mediator does not absorb the effect.

**The sensitivity analysis is the more important number than the point estimates.** With 18 X=1 drives and 76 total failures, the plug-in NDE and NIE have meaningful bootstrap variance. The *direction* of the conclusion (direct path dominates) survives realistic unmeasured-confounding assumptions, which is what an engineer needs to act on; the precise share (68 / 32) is not robust enough to claim. Report the direction and the sensitivity, not the percentages."""),

md("""## What's next

Lab 8B uses the same Backblaze panel for a *dynamic-treatment-regime* (DTR) analysis: at each day's SMART reading, what is the optimal replace-or-wait decision? Where 10B decomposes a single time-window effect, 8B finds the policy that minimises expected cost across multiple decision points."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch10" / "lab10b_backblaze_mediation.ipynb", cells)
print("Built lab10b_backblaze_mediation.ipynb")
