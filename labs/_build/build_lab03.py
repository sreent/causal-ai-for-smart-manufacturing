"""Build labs/ch03/lab03.ipynb — front-door and do-calculus."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 3 — Front-Door Identification and the do-calculus

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03.ipynb)

**Companion lab to Chapter 3.** Use the front-door criterion to identify the effect of a recipe change on yield when the only confounder is unobserved tool drift. Watch the back-door criterion fail. Implement the two-stage front-door estimator by hand, verify it against the SCM's truth, and stress it under non-linear mediator-outcome surfaces."""),

md("""## What you'll do

1. **Build the SCM** from Chapter 3 §3.5: $U \\to X$, $U \\to Y$, $X \\to M$, $M \\to Y$ — unobserved tool drift $U$ confounds the recipe-yield relationship; $M$ (film thickness) fully mediates the recipe's effect on yield.
2. **Verify back-door is blocked**: the only back-door path from $X$ to $Y$ runs through unobserved $U$. No valid adjustment set exists from the observed variables alone.
3. **Apply the front-door criterion** to $M$ and derive the identifying formula.
4. **Implement the two-stage front-door estimator** by hand, recover the truth.
5. **Stress-test under non-linearity**: replace the linear $M \\to Y$ with a non-linear surface; observe that the "linear shortcut" estimator breaks and the proper Monte-Carlo version still works.
6. **Compare with DoWhy** to validate the manual implementation against a library."""),

md("""## Setup"""),

code("""# Colab: install DoWhy for Part 6's library validation.
%pip install --quiet dowhy 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The SCM with unobserved tool drift

The chapter's running example: a deposition step where the engineer chooses a *pressure recipe* ($X$), the wafer's *film thickness* ($M$) is measured after deposition, and end-of-line *yield* ($Y$) is recorded. An unobserved *tool drift state* ($U$) — chamber wall condition, heater temperature offset, RF coupling efficiency, things that aren't measured per run — affects both the recipe (because the engineer compensates for known drift) and yield directly (because the same drift causes defects independently of the recipe).

The DAG:

```
   U (unobserved)
   │  │
   ▼  ▼
   X ──► M ──► Y ◄── U
```

Edges: $U \\to X$, $U \\to Y$, $X \\to M$, $M \\to Y$. No direct $X \\to Y$ edge — the mediator $M$ carries all of $X$'s effect on $Y$.

Structural equations from §3.5:

- $U \\sim \\mathcal N(0, 1)$ — unobserved tool drift
- $X = 100 + 5 U + \\varepsilon_X$ — pressure recipe (engineer compensates for drift)
- $M = 50 + 0.4 X + \\varepsilon_M$ — film thickness
- $Y = 90 + 0.3 M - 2 U + \\varepsilon_Y$ — yield

The true ATE of $X$ on $Y$: $X$ enters $Y$ only via $M$, with chain coefficient $0.4 \\times 0.3 = 0.12$ per unit $X$."""),

code("""n = 10_000

U = rng.normal(0, 1, n)
X = 100 + 5 * U + rng.normal(0, 2, n)
M = 50 + 0.4 * X + rng.normal(0, 1, n)
Y = 90 + 0.3 * M - 2 * U + rng.normal(0, 1, n)

df = pd.DataFrame({"X": X, "M": M, "Y": Y})   # U is NOT in the data
df.head()"""),

md("""$U$ is not in the data warehouse — only $X$, $M$, $Y$ are. This is the realistic setting: tool-drift sensors may exist but their data isn't joined to the wafer-level outcome table, or the relevant aspect of drift isn't directly measurable at all."""),

md("""## Part 2 — Why back-door fails

The back-door criterion: a set $Z$ is valid if it (a) contains no descendant of $X$ and (b) blocks every back-door path from $X$ to $Y$.

Back-door paths from $X$ to $Y$ in this DAG: just one, $X \\leftarrow U \\to Y$. To block it we'd need $U \\in Z$. But $U$ is unobserved. So no valid back-door set exists in this DAG.

What if we try to adjust on $M$? $M$ is a descendant of $X$, so it violates condition (a). What if we try no adjustment at all? Then the back-door path is wide open and the estimate is biased.

Let's verify the naive estimator is biased."""),

code("""x_low, x_high = np.percentile(df["X"], [25, 75])
contrast = x_high - x_low

# True ATE from the SCM
true_ate = 0.4 * 0.3 * contrast
print(f"X contrast: {x_low:.2f} -> {x_high:.2f}  (range {contrast:.2f})")
print(f"True ATE (analytical): {true_ate:+.3f}")
print()

# Naive Y ~ X
naive_lr = LinearRegression().fit(df[["X"]], df["Y"])
naive_ate = (naive_lr.predict([[x_high]])[0] - naive_lr.predict([[x_low]])[0])
print(f"Naive Y~X slope:       {naive_lr.coef_[0]:.4f} per unit X")
print(f"Naive ATE estimate:    {naive_ate:+.3f}")
print(f"Bias from omitted U:   {naive_ate - true_ate:+.3f}")"""),

md("""The naive slope is negative — the higher the pressure, the lower the yield in the data. The interpretation "raise the pressure recipe → lose yield" sounds plausible. It's also wrong by sign.

What's happening: high-pressure recipes are chosen *because* the operator detects high drift (high $U$). High $U$ directly hurts yield ($-2 U$). The data lake records high pressure with low yield, simply because both are downstream of drift. The recipe itself, at fixed drift, *helps* yield ($+0.12$ per unit X).

A team acting on the naive observational slope would lower the pressure setpoint and watch yield collapse — they removed the recipe's compensating effect while leaving the drift in place."""),

md("""## Part 3 — The front-door criterion

The front-door criterion (Chapter 3 §3.3) requires three conditions on a mediator $M$:

1. $M$ intercepts every directed path from $X$ to $Y$. ✓ — the only directed path is $X \\to M \\to Y$.
2. There is no back-door path from $X$ to $M$. ✓ — paths into $X$ come from $U$, and $U$ doesn't connect to $M$ except through $X$.
3. Every back-door path from $M$ to $Y$ is blocked by conditioning on $X$. ✓ — the only back-door path is $M \\leftarrow X \\leftarrow U \\to Y$, which is blocked at $X$.

The front-door formula:

$$P(Y \\mid \\text{do}(X = x)) = \\sum_m P(M = m \\mid X = x) \\sum_{x'} P(Y \\mid X = x', M = m)\\,P(X = x').$$

In words: integrate over the mediator distribution under the intervention (the inner term $P(M \\mid X = x)$) and then average the outcome model over the *marginal* distribution of $X$ (the outer reweighting). The outer average is what removes the confounding through $U$: it asks "what's the average yield at this film thickness, across the natural distribution of pressures?", not "what's the average yield at this film thickness *for this pressure value*?" The first is a counterfactual quantity; the second is observational.

The estimator is two-stage. (i) Fit a model of $M$ on $X$: $\\hat\\mu_M(x) = E[M \\mid X = x]$. (ii) Fit a model of $Y$ on $X$ and $M$: $\\hat\\mu_Y(x, m) = E[Y \\mid X = x, M = m]$. Then compose by the formula."""),

md("""## Part 4 — The two-stage front-door estimator

The chapter's code (§3.5) implements a linear-shortcut version: replace the inner average by a point evaluation at $\\hat\\mu_M(x)$. This shortcut is exact when $\\hat\\mu_Y$ is linear in $M$. Let's implement both versions — the shortcut and the proper Monte-Carlo — to see when each gives the right answer."""),

code("""# Stage 1: mediator model M ~ X
mu_M = LinearRegression().fit(df[["X"]], df["M"])

# Stage 2: outcome model Y ~ X + M
mu_Y = LinearRegression().fit(df[["X", "M"]], df["Y"])

def fd_outcome_shortcut(x_value):
    \"\"\"Front-door outer expectation, linear shortcut.

    Evaluates mu_Y at (every observed X, m_pred) where m_pred = E[M | X = x_value].
    Valid only when mu_Y is linear in M.
    \"\"\"
    m_pred = mu_M.predict([[x_value]])[0]
    y_pred = mu_Y.predict(np.column_stack([df["X"], np.full(len(df), m_pred)]))
    return y_pred.mean()

def fd_outcome_mc(x_value, n_samples=500, rng=np.random.default_rng(42)):
    \"\"\"Front-door outer expectation, Monte Carlo over P(M | X = x_value).

    Draws n_samples M values from the empirical conditional, evaluates mu_Y
    at each (X_observed, m_sample), averages. Valid for any mu_Y.
    \"\"\"
    # Use the residual variance of the M ~ X model as the conditional sd
    sigma_M = (df["M"] - mu_M.predict(df[["X"]])).std()
    m_mean = mu_M.predict([[x_value]])[0]
    m_samples = rng.normal(m_mean, sigma_M, n_samples)
    ys = []
    for m in m_samples:
        y_pred = mu_Y.predict(np.column_stack([df["X"], np.full(len(df), m)]))
        ys.append(y_pred.mean())
    return float(np.mean(ys))

fd_ate_shortcut = fd_outcome_shortcut(x_high) - fd_outcome_shortcut(x_low)
fd_ate_mc       = fd_outcome_mc(x_high)       - fd_outcome_mc(x_low)

print(f"True ATE:                       {true_ate:+.3f}")
print(f"Naive Y~X:                      {naive_ate:+.3f}  (biased)")
print(f"Front-door (linear shortcut):   {fd_ate_shortcut:+.3f}")
print(f"Front-door (Monte Carlo):       {fd_ate_mc:+.3f}")"""),

md("""Both front-door estimators recover the true ATE: ~$+0.12 \\times 7.5 = +0.9$ for the $X$-contrast used. The naive estimator goes the wrong way. The front-door criterion has identified the causal effect without ever seeing $U$.

This is a remarkable result. The unobserved confounder $U$ creates exactly the kind of bias that derailed Labs 1 and 2 — and yet, by routing through the mediator $M$, we recover the truth. The price is the front-door's structural requirements: full mediation, no $X$-$M$ confounding, and a $M$-$Y$ confounding that conditioning on $X$ removes. In manufacturing these requirements often hold for in-line quality intermediates that fully mediate from inputs to outcomes."""),

md("""## Part 5 — Non-linear $M \\to Y$: the shortcut breaks

The linear-shortcut estimator assumes $\\hat\\mu_Y$ is linear in $M$ — then taking $E_M$ commutes with the model. For non-linear $\\hat\\mu_Y$, the shortcut underestimates or overestimates depending on the curvature.

To demonstrate, let's regenerate the data with a non-linear $M \\to Y$ surface: $Y = 90 + 0.005 (M - 50)^2 + 0.1 M - 2 U + \\varepsilon_Y$. The quadratic term curves the surface; the linear shortcut will be biased."""),

code("""U2 = rng.normal(0, 1, n)
X2 = 100 + 5 * U2 + rng.normal(0, 2, n)
M2 = 50 + 0.4 * X2 + rng.normal(0, 1, n)
# Non-linear M -> Y
Y2 = 90 + 0.005 * (M2 - 50)**2 + 0.1 * M2 - 2 * U2 + rng.normal(0, 1, n)
df2 = pd.DataFrame({"X": X2, "M": M2, "Y": Y2})

# True ATE (analytical): integrate the non-linear chain.
# At each X, M ~ N(50 + 0.4X, 1). E[Y | do(X)] = E_U[E_M[Y | do(X), U]]
# = E_M[0.005 (M - 50)^2 + 0.1 M] + constants  (E_U[-2U] = 0 under intervention)
# E[(M - 50)^2 | do(X)] = (50 + 0.4X - 50)^2 + Var(M | X) = (0.4X)^2 + 1
# E[M | do(X)] = 50 + 0.4 X
# So E[Y | do(X)] = 90 + 0.005 * ((0.4X)^2 + 1) + 0.1*(50 + 0.4X)

def true_y_under_do(x):
    return 90 + 0.005 * ((0.4*x)**2 + 1) + 0.1*(50 + 0.4*x)

x_low2, x_high2 = np.percentile(df2["X"], [25, 75])
true_ate2 = true_y_under_do(x_high2) - true_y_under_do(x_low2)

# Fit models — use gradient boosting to capture the non-linearity of mu_Y
mu_M2 = LinearRegression().fit(df2[["X"]], df2["M"])
mu_Y2 = GradientBoostingRegressor(random_state=0).fit(df2[["X", "M"]], df2["Y"])

def fd_shortcut_v2(x_v):
    m_pred = mu_M2.predict([[x_v]])[0]
    return mu_Y2.predict(np.column_stack([df2["X"], np.full(len(df2), m_pred)])).mean()

def fd_mc_v2(x_v, n_samples=500, rng=np.random.default_rng(42)):
    sigma_M = (df2["M"] - mu_M2.predict(df2[["X"]])).std()
    m_mean  = mu_M2.predict([[x_v]])[0]
    m_samples = rng.normal(m_mean, sigma_M, n_samples)
    return float(np.mean([mu_Y2.predict(np.column_stack([df2["X"], np.full(len(df2), m)])).mean()
                          for m in m_samples]))

print(f"True ATE (non-linear M->Y):     {true_ate2:+.3f}")
print(f"FD shortcut (point-eval of M):  {fd_shortcut_v2(x_high2) - fd_shortcut_v2(x_low2):+.3f}")
print(f"FD Monte Carlo (sample M):      {fd_mc_v2(x_high2) - fd_mc_v2(x_low2):+.3f}")"""),

md("""The shortcut and the Monte-Carlo agree closely here — the quadratic curvature is small relative to the mediator's variance, so $E[\\text{curve}(M)] \\approx \\text{curve}(E[M])$ to a good approximation. With a sharper non-linearity (e.g., a sigmoid bend in $M \\to Y$), the gap widens.

The general rule: the linear-in-mediator shortcut is a parametric assumption baked into the estimator. In the chapter's clean linear SCM, it's exact. In the field, with non-linear outcome surfaces, the Monte-Carlo version is the right default."""),

md("""## Part 6 — Validation with DoWhy

The `dowhy` library implements the front-door estimator (and many others) with the same identification machinery. Let's verify our manual implementation matches DoWhy's output."""),

code("""# DoWhy installation: it's heavy. Try the import; if missing, skip this part gracefully.
try:
    import dowhy
    from dowhy import CausalModel
    have_dowhy = True
except ImportError:
    have_dowhy = False
    print("DoWhy not installed in this environment. Install with: %pip install dowhy")

if have_dowhy:
    # CausalModel API: pass the data, treatment, outcome, and DAG (as GML or dot)
    gml = '''graph [directed 1
        node [id "U" label "U"]
        node [id "X" label "X"]
        node [id "M" label "M"]
        node [id "Y" label "Y"]
        edge [source "U" target "X"]
        edge [source "U" target "Y"]
        edge [source "X" target "M"]
        edge [source "M" target "Y"]
    ]'''
    model = CausalModel(
        data=df.assign(U=U),    # DoWhy needs all variables; we expose U here only for the model spec
        treatment="X", outcome="Y",
        graph=gml,
    )
    # Identify with the front-door criterion
    estimand = model.identify_effect(method_name="default", proceed_when_unidentifiable=True)
    print("Identified estimand (front-door):")
    print(str(estimand).split('\\n')[0:6])
    # Estimate (linear front-door)
    est = model.estimate_effect(estimand, method_name="frontdoor.two_stage_regression")
    # Note: this returns slope per unit X; multiply by the X contrast
    print(f"\\nDoWhy estimate (per unit X): {est.value:.4f}")
    print(f"Per-contrast (x_high - x_low = {contrast:.2f}): {est.value * contrast:+.3f}")
    print(f"Manual front-door estimate:                   {fd_ate_mc:+.3f}")
    print(f"True ATE:                                     {true_ate:+.3f}")"""),

md("""DoWhy's identified estimand should match the manual derivation, and its estimate should match the Monte-Carlo implementation within sampling noise. If DoWhy isn't available in your environment, the manual implementation in Part 4 stands on its own — the lab's point is the *mechanism*, not the library."""),

md("""## Part 7 — The do-calculus, briefly

Front-door is one named criterion. The do-calculus (Pearl 1995) provides three rewriting rules that, applied repeatedly, are *complete* for identification — if a query is identifiable from the DAG and observed variables, the rules will produce an identifying formula (Shpitser & Pearl 2006).

The three rules (informal versions, from §3.4 of the chapter):

1. **Insertion/deletion of observations.** An observation $Z$ can be added to or removed from a conditioning set if $Z$ is d-separated from the outcome in the appropriate manipulated graph.
2. **Action–observation exchange.** A $\\text{do}(Z)$ can be swapped for conditioning on $Z$ when they have the same effect on the outcome.
3. **Insertion/deletion of actions.** A $\\text{do}(Z)$ can be added or removed when it has no causal effect on the outcome in the manipulated graph.

In practice, nobody applies these by hand on a DAG with more than a few nodes. Algorithms — Tian-Pearl ID (2002) and IDC (Shpitser-Pearl 2006) — automate the process. DoWhy implements them; `causaleffect` (R) and `causalfusion.net` (web tool) implement them as well. The role of the human is to defend the DAG; the algorithm derives the estimator.

A quick illustration: the front-door formula can be derived from the do-calculus rules applied to our DAG. Rule 2 lets us replace $P(Y \\mid \\text{do}(X = x))$ with an expression involving $P(M \\mid \\text{do}(X = x))$ (via the mediator). Rule 3 then converts $\\text{do}(X)$ into observational conditioning where appropriate. Several applications of Rules 1-3 yield the front-door formula. The chapter's §3.4 walks through this; the algorithmic point is that we don't have to."""),

md("""## Reflection

**Front-door is a *constructive* identification result.** When back-door fails, front-door may still apply — and when it does, you get an *unbiased* estimate of a confounded effect using only observed variables. The price is structural: full mediation, $X$-$M$ unconfoundedness, and conditional $M$-$Y$ unconfoundedness given $X$. These are strong but checkable assumptions (you defend each on the DAG, then check the implications on the data).

**The shortcut estimator is parametric.** The linear-in-$M$ shortcut assumes $\\hat\\mu_Y$ is linear; for non-linear $\\hat\\mu_Y$, you need the Monte-Carlo version. With small mediator variance and mild non-linearity the gap is small; with sharp non-linearities it can be substantial.

**Algorithmic identification scales beyond named criteria.** When neither back-door nor front-door applies, the do-calculus (via ID/IDC algorithms in DoWhy) may still identify the query. Conversely, the algorithm may return *non-identifiable* — a useful certificate that no estimator from the observed variables will work, and you need a different design (IV, RDD, DID — Lab 4)."""),

md("""## Exercises

1. **Find a DAG where front-door fails but the do-calculus still identifies.** Hint: add a second observed mediator and a path from $U$ to $M$ that breaks one of the front-door conditions. Try DoWhy on it.

   <details><summary>Solution</summary>

   Add an edge $U \\to M$: now $X \\to M$ has a back-door $X \\leftarrow U \\to M$, violating front-door condition 2. Front-door fails. But add a *second* observed mediator $M_2$ with $X \\to M_2 \\to Y$ and no $U$ connection, and the do-calculus (via DoWhy's `identify_effect` running the ID algorithm) may still identify $P(Y \\mid \\text{do}(X))$ through a two-mediator decomposition. The general lesson: when a named criterion fails, the algorithmic ID routine may still find an identifying formula.
   </details>

2. **Add a direct $X \\to Y$ edge** to the DAG. Does front-door still apply? What is the identifying formula now?

   <details><summary>Solution</summary>

   No. Condition 1 of the front-door criterion (full mediation: $M$ intercepts every directed path from $X$ to $Y$) fails. The direct $X \\to Y$ edge bypasses $M$. The effect is no longer identifiable from $(X, M, Y)$ alone; you would need a back-door adjustment on $U$ (impossible if $U$ is unobserved), or an instrument for $X$ (Chapter 4), or sensitivity bounds (Chapter 13).
   </details>

3. **Stress with a categorical $M$.** Replace the continuous $M$ with a discrete two-level film-thickness bin ("thin", "thick"). Redo the front-door estimator — the formula now uses sums instead of integrals.

   <details><summary>Solution</summary>

   ```python
   M_cat = (M > M.mean()).astype(int)
   df_cat = df.assign(M=M_cat)
   # mu_M now returns P(M=1 | X=x)
   mu_M = LinearRegression().fit(df_cat[["X"]], df_cat["M"])
   mu_Y = LinearRegression().fit(df_cat[["X", "M"]], df_cat["Y"])
   # Front-door sum over m ∈ {0,1}, outer expectation over X' ~ P(X)
   ```

   The two-level $M$ loses information about the continuous thickness, so the estimator's variance increases. Finer discretization → continuous limit. The structure of the formula is identical.
   </details>

4. **Sensitivity to the no-$U$-to-$M$ assumption.** Modify the SCM to add a small $U \\to M$ edge: $M = 50 + 0.4 X + 0.5 U + \\varepsilon_M$. Front-door's second condition is now violated. Run the front-door estimator anyway — how much bias does the violation introduce?

   <details><summary>Solution</summary>

   With $U \\to M$ coefficient $\\beta_{UM}$, the front-door estimator's bias grows linearly in $\\beta_{UM} \\cdot \\text{cov}(U, X) \\cdot \\beta_{MY}$. At $\\beta_{UM} = 0.5$, expect ~10–20% bias on the point estimate. The estimator degrades gracefully — small violations give small bias. A Chapter 13 sensitivity analysis bounds the bias against the strength of the hypothetical $U \\to M$ edge.
   </details>"""),

md("""## What's next

Lab 4 turns to quasi-experimental designs: when neither back-door nor front-door applies, an instrumental variable, a regression-discontinuity threshold, or a difference-in-differences design may still let you identify. The setting of Lab 4 is the chapter's three classical designs applied to manufacturing data."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch03" / "lab03.ipynb", cells)
print("Built lab03.ipynb")
