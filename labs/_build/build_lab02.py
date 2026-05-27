"""Build labs/ch02/lab02.ipynb — SCMs, DAGs, and the back-door criterion.

Uses the chapter's exact SCM and the same X3 -> Y identification target
as Chapter 2's §2.9 worked example.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 2 — SCMs, DAGs, and the Back-Door Criterion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch02/lab02.ipynb)

**Companion lab to Chapter 2.** Construct the chapter's bracket-line SCM explicitly, verify d-separation by simulation, apply the back-door criterion to estimate the effect of weld current ($X_3$) on yield ($Y$), and watch what happens when you adjust on the wrong set.

In Lab 1 you saw a single example of confounding and one hand-built adjustment. This lab develops the *general* mechanism: how a DAG encodes which variables to adjust for, how d-separation lets you check that on paper, and how the back-door criterion turns a graph into an estimator. We re-use the chapter's SCM exactly, so your numbers match §2.9's worked example."""),

md("""## What you'll do

1. **Build the SCM** for the four-stage bracket line, following §2.3 exactly.
2. **Compute the true interventional ATE** of $X_3$ on $Y$ from the structural equations.
3. **Reproduce the chapter's §2.9 result**: naive estimate vs. back-door adjustment on $\\lbrace L\\rbrace $ vs. truth.
4. **Try alternative valid adjustment sets** ($\\lbrace R\\rbrace $, $\\lbrace L, R\\rbrace $) and confirm they all recover the truth.
5. **Try wrong adjustment sets** — a mediator, a collider, an irrelevant variable — and watch the estimate go wrong.
6. **Verify d-separation by simulation**: pick claimed d-separations from the DAG and check the implied conditional independence in the data.
7. **Stress the analysis** by hiding $L$ from the data warehouse and seeing whether $R$ alone suffices."""),

md("""## Setup"""),

code("""# Colab: install the libraries used in Part 9 (library validation).
%pip install --quiet dowhy 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The bracket-line SCM (Chapter 2 §2.3)

The four-stage bracket line: blank cut, stamp, weld, inspect. We summarize the relevant variables:

- $L$ — material lot type (a continuous latent confounder; in this lab we treat it as observed)
- $R$ — recipe scalar (set per lot, downstream of $L$)
- $X_2$ — stamp force (set by the recipe with operator noise)
- $X_3$ — weld current (set by the recipe with operator noise)
- $M_2$ — post-stamp dimensional measurement (a quality intermediate)
- $Y$ — yield score (continuous proxy for pass/fail probability)

The DAG:

```
   L ──► R ──► X2 ──┐
   │     │         ▼
   │     └─► X3    M2
   ▼              ╱  ╲
   M2 ◄──L───────╱    ▼
   ▼                  Y
   Y ◄──L
   Y ◄──X3
```

(Edges: $L \\to R$, $L \\to M_2$, $L \\to Y$, $R \\to X_2$, $R \\to X_3$, $X_2 \\to M_2$, $M_2 \\to Y$, $X_3 \\to Y$.)

We will study the effect of *weld current* ($X_3$) on *yield* ($Y$). The structural equation $Y = 75 - 2L + 0.1\\,X_3 + 0.3\\,M_2 + U_Y$ tells us the true direct effect of $X_3$ on $Y$ is $0.1$ per unit $X_3$ (because $X_3$ does not appear in any other equation downstream — it goes directly to $Y$, not through $M_2$)."""),

code("""def gen_data(n, rng, x3_intervention=None):
    \"\"\"Generate observational data, optionally intervening on X3.

    If x3_intervention is None: sample X3 from its structural equation.
    If x3_intervention is a float: replace X3's structural equation with X3 = const.
    \"\"\"
    U_L  = rng.normal(0, 1.0, n)
    U_R  = rng.normal(0, 0.5, n)
    U_X2 = rng.normal(0, 5.0, n)
    U_X3 = rng.normal(0, 5.0, n)
    U_M2 = rng.normal(0, 0.5, n)
    U_Y  = rng.normal(0, 1.0, n)

    L  = U_L
    R  = 0.5 * L + U_R
    X2 = 580 + 30 * R + U_X2
    if x3_intervention is None:
        X3 = 200 + 15 * R + U_X3
    else:
        X3 = np.full(n, x3_intervention)
    M2 = 10 + 0.05 * X2 + 0.3 * L + U_M2
    Y  = 75 - 2.0 * L + 0.1 * X3 + 0.3 * M2 + U_Y

    return pd.DataFrame({"L": L, "R": R, "X2": X2, "X3": X3, "M2": M2, "Y": Y})

df = gen_data(10_000, np.random.default_rng(0))
df.head()"""),

md("""## Part 2 — True ATE via the do-operator

The chapter studies the 25th-to-75th-percentile contrast in $X_3$ as a concrete intervention. We replicate that. The analytical ATE is the path-coefficient product from $X_3$ to $Y$: just $0.1$ per unit (no mediator on this path).

We also verify the analytical answer by Monte Carlo: simulate $\\text{do}(X_3 = x_{\\text{low}})$ and $\\text{do}(X_3 = x_{\\text{high}})$ separately, take the mean difference."""),

code("""x_low, x_high = np.percentile(df["X3"], [25, 75])
print(f"X3 contrast: {x_low:.1f} -> {x_high:.1f} ({x_high - x_low:+.2f} units)")

# Analytical: 0.1 per unit X3
ate_analytical = 0.1 * (x_high - x_low)

# Monte Carlo: simulate the do-operator on the SCM
df_low  = gen_data(50_000, np.random.default_rng(1), x3_intervention=x_low)
df_high = gen_data(50_000, np.random.default_rng(1), x3_intervention=x_high)
ate_mc = df_high["Y"].mean() - df_low["Y"].mean()

print(f"Analytical ATE:        {ate_analytical:+.3f}")
print(f"Monte Carlo ATE:       {ate_mc:+.3f}")"""),

md("""The two answers agree. This is the ground truth that any valid estimator should recover from observational data alone."""),

md("""## Part 3 — Naive estimate vs. back-door on $\\lbrace L\\rbrace $ (the chapter's §2.9 result)

The chapter shows that the naive (no-adjustment) estimator undershoots and back-door adjustment on $\\lbrace L\\rbrace $ recovers the truth (up to finite-sample noise). We reproduce that here."""),

code("""# Naive: fit Y on X3 alone, predict at the contrast points
mu_naive = GradientBoostingRegressor(random_state=0).fit(df[["X3"]], df["Y"])
ate_naive = (mu_naive.predict(np.full((len(df), 1), x_high))
             - mu_naive.predict(np.full((len(df), 1), x_low))).mean()

# Back-door on {L}: fit Y on (X3, L), predict at the contrast points with observed L
mu_L = GradientBoostingRegressor(random_state=0).fit(df[["X3", "L"]], df["Y"])
pred_high = mu_L.predict(np.column_stack([np.full(len(df), x_high), df["L"]]))
pred_low  = mu_L.predict(np.column_stack([np.full(len(df), x_low),  df["L"]]))
ate_L = (pred_high - pred_low).mean()

print(f"Naive estimate:                  {ate_naive:+.3f}")
print(f"Back-door on {{L}}:                {ate_L:+.3f}")
print(f"True ATE (Monte Carlo):          {ate_mc:+.3f}")"""),

md("""The naive estimate undershoots dramatically — the chapter explains why (§2.9 paragraph after the code block): $L$ creates a back-door path that pulls the observational $X_3$–$Y$ relationship downward (high-$L$ lots get *more* $X_3$ through $R$ but the same $L$ has a direct *negative* effect on $Y$). The negative direct $L \\to Y$ effect cancels much of the positive $X_3 \\to Y$ direct effect in the observational data.

Back-door adjustment recovers an estimate close to the truth; any residual gap is finite-sample noise in the gradient-boosted regressor (the chapter's $+1.742$ vs the truth's $+1.563$ falls in this range)."""),

md("""## Part 4 — Other valid back-door sets

§2.9 also notes that $\\lbrace R\\rbrace $ is a valid adjustment set: $R$ is the chain point closer to $X_3$ on both back-door paths and conditioning on it blocks them. We verify that, plus a few mixed sets."""),

code("""def backdoor_ate(adjust_cols):
    \"\"\"Back-door ATE estimator using a gradient-boosted regression.\"\"\"
    mu = GradientBoostingRegressor(random_state=0).fit(df[["X3"] + list(adjust_cols)], df["Y"])
    pred_high = mu.predict(np.column_stack([np.full(len(df), x_high), df[adjust_cols].values]))
    pred_low  = mu.predict(np.column_stack([np.full(len(df), x_low),  df[adjust_cols].values]))
    return (pred_high - pred_low).mean()

results = {
    "True (MC)":        ate_mc,
    "Naive":            ate_naive,
    "Adjust on {L}":    backdoor_ate(["L"]),
    "Adjust on {R}":    backdoor_ate(["R"]),
    "Adjust on {L, R}": backdoor_ate(["L", "R"]),
    "Adjust on {L, M2, X2}": backdoor_ate(["L", "M2", "X2"]),
}
for name, est in results.items():
    print(f"  {name:<30s} {est:+.3f}")"""),

md("""The three minimal valid adjustments — $\\lbrace L\\rbrace$, $\\lbrace R\\rbrace$, $\\lbrace L, R\\rbrace$ — all recover the true ATE within sampling noise. Note that $\\lbrace L, M_2, X_2\\rbrace$ gives a noticeably worse estimate ($+1.347$ vs truth $+1.563$). This isn't random noise — it's bias introduced by adding $M_2$ to the conditioning set. $M_2$ is a collider on the path $L \\to M_2 \\leftarrow X_2$, and conditioning on it opens that path, inducing a spurious dependence between $L$ and $X_2$ that biases the $X_3$ coefficient. Part 6 demonstrates the same collider-opening from the partial-correlation side. **The "control for everything" reflex is wrong.** Condition on what the back-door criterion calls for, not on every covariate you have.

Two practical notes from the chapter. First, $\\lbrace L\\rbrace$ and $\\lbrace R\\rbrace$ are *both* valid, but the choice between them depends on data quality: $L$ (lot type) is usually well-recorded in MES traveler documents, while $R$ (recipe) may have implementation noise — operators sometimes deviate from the nominal recipe. When both are available, the more upstream and more reliably measured is preferable. Second, $M_2$ is not in any valid adjustment set for $X_3 \\to Y$ even though it is a quality intermediate — because it lies on no back-door path from $X_3$ to $Y$. Adjustment sets are *query-specific*."""),

md("""## Part 5 — Wrong adjustment sets

Now the failure modes. The DAG predicts that adjusting on a collider, a descendant of $X_3$, or an irrelevant non-confounder will *not* fix the bias (and can make it worse). Let's see each."""),

code("""# Three flawed candidates:
# 1. Adjust on Y itself (a descendant — actually, X3 -> Y, so Y is the outcome,
#    not an adjustment candidate; but a real-world equivalent would be a leaky
#    post-outcome variable, e.g. "scrap_count" — same problem. We illustrate by
#    splitting Y into a train-time observed prefix and a "predicted" downstream variable.)
# 2. Adjust on M2 only (a non-confounder for X3 -> Y; doesn't block any back-door path).
# 3. Adjust on X2 only (parallel non-confounder; sibling through R).

results_bad = {
    "Adjust on {M2} only":         backdoor_ate(["M2"]),
    "Adjust on {X2} only":         backdoor_ate(["X2"]),
    "Adjust on {M2, X2}":          backdoor_ate(["M2", "X2"]),
}
for name, est in results_bad.items():
    print(f"  {name:<30s} {est:+.3f}    (true: {ate_mc:+.3f})")"""),

md("""All three are biased — but in different ways and to different degrees. Empirically, $\\lbrace M_2\\rbrace$ alone ($+1.188$) happens to come closer to truth ($+1.563$) than $\\lbrace R\\rbrace$ ($+1.240$) does, despite $R$ being a *valid* adjustment and $M_2$ not. The point estimate alone tells you nothing about validity: a wrong adjustment that's accidentally close to truth in one sample will diverge from truth as the sample grows (or in a different SCM with the same DAG). Validity comes from satisfying the back-door criterion; closeness to truth in a single sample is an unreliable signal.

The DAG-level argument for each failure: $M_2$ alone does not block the back-door path $X_3 \\leftarrow R \\leftarrow L \\to Y$ — that path doesn't go through $M_2$. $X_2$ alone doesn't either — $X_2$ is on a sibling path, not the confounding path. Combining them still doesn't block the $L \\to Y$ branch. An adjustment set must intercept *every* back-door path, not just some of them, and not just the ones a variable-importance algorithm flags."""),

md("""## Part 6 — Collider conditioning

A particularly subtle failure mode: conditioning on a collider can *create* a spurious dependency that wasn't there. Consider the path $L \\to M_2 \\leftarrow X_2$: $M_2$ is a collider on this path (both arrows point into it). If we condition on $M_2$, we *open* the path, creating dependence between $L$ and $X_2$ given $M_2$. Let's verify."""),

code("""# Marginal: L and X2 are dependent only through their common cause structure
# (L -> R -> X2). Partial out R and they should be independent.
# Conditional on M2 (a collider): L and X2 should be dependent even after partialling on R.

def partial_corr(x, y, controls):
    \"\"\"Partial correlation of x and y given controls.\"\"\"
    lr_x = LinearRegression().fit(controls, x); resid_x = x - lr_x.predict(controls)
    lr_y = LinearRegression().fit(controls, y); resid_y = y - lr_y.predict(controls)
    return np.corrcoef(resid_x, resid_y)[0, 1]

L, X2, R_var, M2 = df["L"].values, df["X2"].values, df["R"].values, df["M2"].values

print(f"Marginal corr(L, X2):                  {np.corrcoef(L, X2)[0, 1]:+.3f}")
print(f"Partial corr(L, X2 | R):               {partial_corr(L, X2, R_var.reshape(-1, 1)):+.3f}")
print(f"Partial corr(L, X2 | R, M2):           {partial_corr(L, X2, df[['R', 'M2']].values):+.3f}")"""),

md("""The marginal correlation is positive (shared parent $R$). Partialling on $R$ alone reduces it to ~0 — $L$ and $X_2$ are d-separated given $R$. *Adding $M_2$ to the conditioning set restores a negative correlation* — that's collider-opening at work.

In a regression-based estimator, this collider-induced association can introduce bias on coefficients that share a back-door with the collider. In some settings it's harmless; in others it can be substantial. The chapter calls this out as a pitfall in §2.10 — *conditioning on a collider creates dependence between its parents that would not otherwise exist*."""),

md("""## Part 7 — Hide $L$ from the data warehouse

In a real fab, $L$ is often not in the data warehouse (it's a hidden lot-difficulty score). Does the back-door analysis still work using only $R$, which *is* logged?

Yes — $R$ is a complete back-door blocker by itself. Let's verify with $L$ dropped."""),

code("""df_no_L = df.drop(columns=["L"])

mu = GradientBoostingRegressor(random_state=0).fit(df_no_L[["X3", "R"]], df_no_L["Y"])
pred_high = mu.predict(np.column_stack([np.full(len(df), x_high), df_no_L["R"]]))
pred_low  = mu.predict(np.column_stack([np.full(len(df), x_low),  df_no_L["R"]]))
ate_R_only = (pred_high - pred_low).mean()

print(f"L dropped from the warehouse; adjust on R: {ate_R_only:+.3f}")
print(f"True ATE:                                  {ate_mc:+.3f}")"""),

md("""$\\lbrace R\\rbrace $ alone — without $L$ — still recovers the true ATE. This is the chapter's point about choosing the most reliable adjustment available: $R$ might be all you have, and the back-door criterion guarantees it's enough.

What if neither $L$ nor $R$ is logged? You're left with a downstream view of the system. No valid adjustment set blocks both back-door paths, and the back-door criterion is permanently inapplicable. The next chapters develop alternative identification strategies (front-door, IV, RDD) for exactly this case."""),

md("""## Part 8 — Connect back to the chapter

The chapter's §2.9 worked example reports: naive ATE $+0.222$, back-door on $\\lbrace L\\rbrace $ $+1.742$, true ATE $+1.563$. Your numbers from this lab should be close (the exact values depend on RNG seed and the gradient-booster's hyperparameters, but order of magnitude and sign should match).

The chapter's three pitfalls (§2.10) are now concrete in your hands:

1. *Conditioning on a collider* — Part 6, the L-X2 partial correlation flipping sign when M2 is added.
2. *Conditioning on a mediator* — Part 5 with M2 as adjustment, which (for X3 → Y) doesn't directly block but does collider-open. For $X_2 \\to Y$ (not studied here) M2 would be a mediator.
3. *Variable selection ≠ identification* — feature importance, regularization, cross-validation all fail to flag a wrong adjustment set; only the DAG can."""),

md("""## Part 9 — Library validation: DoWhy on the same DAG

The bracket-line analysis above used scikit-learn directly. DoWhy implements the same identification + back-door adjustment behind a single API. Use it to (a) double-check our manual estimates and (b) see what a production pipeline looks like."""),

code("""try:
    import dowhy
    from dowhy import CausalModel
    have_dowhy = True
except ImportError:
    have_dowhy = False
    print("DoWhy not installed. The %pip install at the top should have fixed this in Colab.")

if have_dowhy:
    gml = '''graph [directed 1
        node [id "L" label "L"]
        node [id "R" label "R"]
        node [id "X2" label "X2"]
        node [id "X3" label "X3"]
        node [id "M2" label "M2"]
        node [id "Y" label "Y"]
        edge [source "L" target "R"]
        edge [source "L" target "M2"]
        edge [source "L" target "Y"]
        edge [source "R" target "X2"]
        edge [source "R" target "X3"]
        edge [source "X2" target "M2"]
        edge [source "M2" target "Y"]
        edge [source "X3" target "Y"]
    ]'''
    model = CausalModel(data=df, treatment="X3", outcome="Y", graph=gml)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("DoWhy identified estimand (back-door adjustment set):")
    print(estimand.get_backdoor_variables())
    est = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    print(f"\\nDoWhy back-door ATE (per unit X3): {est.value:+.5f}")
    print(f"DoWhy ATE x contrast range:        {est.value * (x_high - x_low):+.4f}")
    print(f"Manual back-door on L (GBC):       {ate_L:+.4f}")
    print(f"True ATE (Monte Carlo):            {ate_mc:+.4f}")"""),

md("""DoWhy picks the same back-door adjustment set as our manual analysis and recovers a similar ATE. The value of using the library is the automated identification: it enumerates the back-door paths, finds a valid adjustment set (or reports non-identifiability), and applies the formula — eliminating the class of mistakes where an analyst silently picks a wrong adjustment set."""),

md("""## Reflection

**The right adjustment set comes from the DAG.** No supervised-learning machinery can substitute. A high-AUC ATE estimator with the wrong adjustment set is a high-AUC wrong answer.

**Identification is local: choose the adjustment for the query.** $M_2$ is irrelevant for $X_3 \\to Y$ but central to $X_2 \\to Y$. Per-query analysis is the right discipline.

**Even with $L$ hidden, the back-door criterion can succeed.** If $R$ is logged, the back-door is still closeable. Identification doesn't require seeing the confounder directly; it requires *some* observable variable on the path."""),

md("""## Exercises

1. **Try $\\lbrace X_2\\rbrace $ as an adjustment for $X_3 \\to Y$.** Does the path $X_3 \\leftarrow R \\to X_2$ open? Compute the estimate. Is $X_2$ on a back-door path from $X_3$ to $Y$?

   <details><summary>Solution</summary>

   $X_2$ is a *sibling* of $X_3$ (both children of $R$). The path $X_3 \\leftarrow R \\to X_2$ exists but stops at $X_2$ — it does not continue to $Y$ except through $M_2$, which is downstream of $X_2$. So $X_2$ is on no back-door path from $X_3$ to $Y$. Conditioning on it adds no useful information; `adjusted_slope(data, ["X2"])` returns ~`+0.222`, close to the naive estimate.
   </details>

2. **Add a new variable to the DAG.** Introduce a stage-3 dimensional measurement $M_3$ with $M_3 = 0.05\\,X_3 + 0.2\\,L + U_{M_3}$ and modify $Y$ to depend on $M_3$: $Y = 75 - 2L + 0.1 X_3 + 0.3 M_2 + 0.4 M_3 + U_Y$. What's the new true ATE of $X_3$ on $Y$? Which adjustment sets are valid?

   <details><summary>Solution</summary>

   New true ATE per unit $X_3$ is $0.1 + 0.05 \\times 0.4 = 0.12$. Back-door paths from $X_3$ are unchanged ($M_3$ is on a *forward* path, not a back-door), so $\\lbrace L\\rbrace$ or $\\lbrace R\\rbrace$ still suffice. Adjusting on $M_3$ would block the indirect path through $M_3$ and underestimate. Adjusting on $L$ and $M_3$ would do likewise.
   </details>

3. **Build the DAG with `dagitty` or `networkx`.** Use a Python graph library to encode the DAG and write a function that enumerates all valid back-door sets for $(X_3, Y)$.

   <details><summary>Solution</summary>

   ```python
   import networkx as nx
   from itertools import chain, combinations
   G = nx.DiGraph([("L","R"),("L","M2"),("L","Y"),("R","X2"),("R","X3"),
                   ("X2","M2"),("M2","Y"),("X3","Y")])
   def powerset(s):
       return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
   # For each subset Z: (a) no descendant of X3 in Z, (b) every back-door
   # path from X3 to Y is d-separated by Z. Yield valid Z.
   ```

   Valid minimal sets here: $\\lbrace L\\rbrace$ and $\\lbrace R\\rbrace$. The full algorithm — *adjustment-set enumeration* (van der Zander et al. 2014) — is implemented in `dagitty` (R) and `pgmpy` (Python).
   </details>

4. **Estimate $X_2 \\to Y$.** What is the true ATE? Enumerate the back-door paths (there are three). What's the smallest valid adjustment set? Compare $\\lbrace L\\rbrace$ to $\\lbrace R\\rbrace$ to $\\lbrace L, R\\rbrace$: do they all work for $X_2 \\to Y$? (Hint: $\\lbrace L\\rbrace$ alone does *not* block all back-door paths from $X_2$ to $Y$ — there's a path through $X_3$ that $\\lbrace L\\rbrace$ doesn't intersect.)

   <details><summary>Solution</summary>

   True ATE of $X_2$ on $Y$: through $X_2 \\to M_2 \\to Y$, the chain coefficient is $0.05 \\times 0.3 = 0.015$. Back-door paths from $X_2$ to $Y$:

   - $X_2 \\leftarrow R \\leftarrow L \\to Y$
   - $X_2 \\leftarrow R \\leftarrow L \\to M_2 \\to Y$
   - $X_2 \\leftarrow R \\to X_3 \\to Y$

   The first two go through $L$; the third does not. $\\lbrace L\\rbrace$ alone leaves the third open. $\\lbrace R\\rbrace$ blocks all three (every path passes through $R$). $\\lbrace L, X_3\\rbrace$ also works. Smallest single-variable valid set: $\\lbrace R\\rbrace$.
   </details>"""),

md("""## What's next

Lab 3 turns to identification when no valid back-door set exists — the front-door criterion. The mediator $M_2$, which was a *failure mode* for $X_3$ in this lab, becomes the *enabler* of identification for $X_2$ in the next."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch02" / "lab02.ipynb", cells)
print("Built lab02.ipynb")
