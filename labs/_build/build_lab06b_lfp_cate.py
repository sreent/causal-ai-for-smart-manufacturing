"""Build labs/ch06/lab06b_lfp_cate.ipynb — CATE on LFP cells, with batch as the effect modifier."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 6B — CATE of the Severson Early-Cycle Feature on Cycle Life

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06b_lfp_cate.ipynb)

**Companion to Lab 6A.** Lab 6A built a synthetic SCM with known heterogeneous treatment effects and verified that meta-learners (T-, S-, X-, DR-) and Causal Forest recover the per-subgroup CATE. **Lab 6B asks the same heterogeneity question on the LFP cell dataset: does the marginal effect of the Severson `log_var_deltaQ` feature on cycle life vary across the three collection batches?**

The deliverable is a per-batch CATE table, an honest statement of how stable the heterogeneity is across estimators, and the engineering recommendation that follows (which batch's cells benefit *most* from a given improvement in the early-cycle signature)."""),

md("""## What this lab is *not* doing

- **Treating `log_var_deltaQ` as a randomly-assigned treatment.** It is a *measured early-cycle feature* of each cell, not an intervention. The CATE here is the *predictive* heterogeneity of the cycle-life curve as a function of this feature, conditioned on batch. A genuine intervention CATE would require manipulating cell construction.
- **Per-protocol CATE.** The 72 specific protocols would give finer-grained effect modifiers but require the original `.mat` files. We use batch as a coarse proxy.
- **Causal Forest with thousands of trees.** Lab 6A introduces Causal Forest; here we use a small `CausalForestDML` configuration to keep the lab fast on Colab."""),

code("""%pip install -q numpy pandas matplotlib scikit-learn econml"""),

code("""import os, sys, urllib.request, pathlib

DATA = pathlib.Path("/content")
for name in ("lfp_prep.py", "lfp_cell_summary.csv", "lfp_cell_cycle.csv"):
    p = DATA / name
    if not p.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/{name}",
            p,
        )
sys.path.insert(0, str(DATA))

from lfp_prep import load_lfp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load and inspect"""),

code("""df = load_lfp(chapter=6)
df["batch_idx"] = pd.Categorical(df["batch"], categories=["train", "test1", "test2"]).codes

print(f"Cells: {len(df)}")
print(df.groupby("batch")[["log_var_deltaQ", "cycle_life"]].agg(["mean", "std"]).round(2).to_string())"""),

md("""## Background — the meta-learner zoo in 90 seconds

Chapter 6 introduces several CATE estimators with overlapping names. Here is the taxonomy as a glance-table so the next four Parts have semantic anchors:

| Name | What it fits | What heterogeneity it can capture |
|------|---------------|-----------------------------------|
| **S-learner** | A *single* outcome model $\\hat{\\mu}(T, X)$ that takes the treatment $T$ as one of its features | Heterogeneity only where the *single* model bends across $X$. If we use a linear $\\hat{\\mu}$, S-learner gives a constant CATE (no heterogeneity). |
| **T-learner** | *Two* outcome models, one for treated and one for control: $\\hat{\\mu}_1(X)$, $\\hat{\\mu}_0(X)$. The CATE is the difference. | Heterogeneity in either arm shows up; small-sample arms can overfit independently. |
| **X-learner** | T-learner plus a cross-fit residual correction. Good when treated/control arm sizes are imbalanced. | Same as T-learner but with variance reduction on the smaller arm. |
| **DR-learner / DML** | Fit the propensity $\\hat{e}(X)$ and outcome $\\hat{\\mu}_a(X)$ as nuisances, plug into the doubly-robust score, then regress the score on the heterogeneity features. Cross-fit. | Robust to nuisance misspecification (one of the two has to be right). The chapter's preferred estimator. |
| **Causal Forest** | Honest random-forest split criterion targeting the CATE rather than the outcome | Nonparametric, scales to many heterogeneity features; can find non-linear modifiers. |

For this lab, **batch is the heterogeneity feature**. We compare two estimators: the T-learner (fit one slope per batch) and DML-CATE (combine cross-fit nuisance models with batch as the modifier). Agreement between them is the chapter's preferred robustness signal.

## Part 2 — Marginal (pooled) effect: the ATE baseline

A single linear regression of `cycle_life` on `log_var_deltaQ` gives the pooled effect — the average across all batches. This is the chapter's S-learner with the simplest possible covariate set (none). Since the model is linear and there is no $T \\times X$ interaction, the S-learner here gives a *constant* effect — no heterogeneity. It is our ATE baseline."""),

code("""ols = LinearRegression().fit(df[["log_var_deltaQ"]].values, df["cycle_life"].values)
print(f"Pooled slope dY/dX:  {float(ols.coef_[0]):+.1f} cycles per unit log_var_deltaQ")
print(f"R^2:                 {ols.score(df[['log_var_deltaQ']].values, df['cycle_life'].values):.3f}")"""),

md("""## Part 3 — Per-batch slopes: the T-learner equivalent

Fit one regression *within* each batch. The per-batch slopes are the simplest CATE estimator — the T-learner restricted to the linear case. Different slopes across batches = the heterogeneity we are trying to characterise."""),

code("""per_batch = []
for b, sub in df.groupby("batch"):
    m = LinearRegression().fit(sub[["log_var_deltaQ"]].values, sub["cycle_life"].values)
    per_batch.append({
        "batch": b,
        "n":      len(sub),
        "slope":  float(m.coef_[0]),
        "intercept": float(m.intercept_),
    })
per_batch_df = pd.DataFrame(per_batch)
print(per_batch_df.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
print()
print(f"Max - min slope across batches: {per_batch_df['slope'].max() - per_batch_df['slope'].min():+.1f} cycles per unit")"""),

md("""## Part 4 — DML-CATE via EconML

**Where DML-CATE actually differs from the T-learner.** Both estimators give a per-batch slope. They differ in two ways:

1. **Nuisance handling.** T-learner fits a separate OLS per batch and uses *only* `log_var_deltaQ` as a feature. DML first residualises both $Y$ and $T$ on the other covariates ($W$ = the remaining capacity-summary features) — *removing the variance those covariates explain in both* — and then fits the CATE on the residuals. The Frisch-Waugh-Lovell decomposition is the algebraic identity behind this; the practical effect is that nuisance variance no longer inflates the CATE's standard error.
2. **Cross-fitting.** DML fits nuisances on $K-1$ folds and predicts on the held-out fold, rotating. T-learner uses the same data for fit and prediction. Without cross-fitting, regularisation bias from the nuisance models leaks into the CATE; with it, the CATE remains asymptotically unbiased.

When does this matter? In small samples with many controls. With 124 LFP cells and only two extra controls, DML's variance reduction is modest. With 50,000 wafers and 200 sensor features, it is the difference between a usable estimate and an unstable one.

DML treats `log_var_deltaQ` as the treatment, `batch_idx` as the heterogeneity feature, and any remaining controls (here: the other capacity-summary features) as nuisances. The CATE function returned is conditional on batch."""),

code("""from econml.dml import LinearDML
from sklearn.preprocessing import OneHotEncoder

Y = df["cycle_life"].values
T = df["log_var_deltaQ"].values
X_het = pd.get_dummies(df["batch"], drop_first=False).astype(float).values
W = df[["max_cap_cyc10", "fade_cyc10_to_100"]].values  # other early-cycle controls

dml = LinearDML(
    model_y=GradientBoostingRegressor(random_state=0, n_estimators=50),
    model_t=GradientBoostingRegressor(random_state=0, n_estimators=50),
    discrete_treatment=False,
    cv=5,
    random_state=0,
)
dml.fit(Y=Y, T=T, X=X_het, W=W)

# Predict CATE for a one-hot vector representing each batch.
batches = ["train", "test1", "test2"]
batch_dummy = np.eye(3, dtype=float)
cate = dml.effect(batch_dummy)
ci_low, ci_high = dml.effect_interval(batch_dummy, alpha=0.05)

cate_df = pd.DataFrame({
    "batch":      batches,
    "CATE":       cate.ravel(),
    "ci_low":     np.atleast_1d(ci_low).ravel(),
    "ci_high":    np.atleast_1d(ci_high).ravel(),
})
print(cate_df.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))"""),

md("""## Part 5 — Compare T-learner, DML-CATE, and the pooled ATE"""),

code("""compare = pd.DataFrame({
    "batch":        batches,
    "T_learner":    [next(r["slope"] for r in per_batch if r["batch"] == b) for b in batches],
    "DML_CATE":     cate.ravel(),
})
compare["pooled_ATE"] = float(ols.coef_[0])
compare["disagreement"] = compare["T_learner"] - compare["DML_CATE"]
print(compare.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))"""),

md("""**How to read the comparison — concrete agreement criteria.**

Both estimators target the same per-batch CATE. Use the DML CI from Part 4 as the *scale of meaningful disagreement* and apply these thresholds:

| Pattern | Interpretation |
|---------|----------------|
| T-learner and DML-CATE within **0.5 CI widths** per batch, both differ from pooled by > 1 CI width | Real heterogeneity, well-identified. Report the per-batch CATEs. |
| Within 0.5 CI widths but indistinguishable from pooled | No meaningful batch-level heterogeneity; pooled ATE is fine. CATE machinery was overkill here. |
| Disagreement of 1-2 CI widths | DML's nuisance correction is doing something the within-batch OLS missed. Report DML as the principled estimator and T-learner as the simple-model reference. |
| Disagreement > 2 CI widths | One of the two is misspecified. Most likely culprit: small per-batch sample sizes drive T-learner to high variance, OR the other capacity controls are themselves effect modifiers DML is mis-handling. Don't trust either point estimate; use the DML *bounds* and call out the disagreement. |

The CI in the Part-4 table is the right scale. *Differences relative to it* matter, not absolute slope values."""),

md("""## Part 6 — Decision

Three bullets:

1. **Per-batch CATE estimates** (read from Parts 4-5) tell us how a *unit improvement in `log_var_deltaQ`* (e.g., from a manufacturing-quality change) would change predicted cycle life *per batch* — i.e., per cell-vintage / protocol family. The batch with the largest negative slope is the one where the early-cycle signature is most diagnostic.

2. **Targeting decision:** if you can intervene only on a subset of cells, prioritise the batch with the strongest CATE (largest absolute slope). The pooled ATE smooths over this and misleads if the heterogeneity is real.

3. **Caveat:** the CATE here is *predictive*, not *interventional*. `log_var_deltaQ` is a measured feature, not something we set. The right interpretation is \"for a hypothetical cell with that early-cycle signature in that batch, what is the conditional expected cycle life?\""""),

md("""## Reflection

**CATE estimation amplifies the importance of getting the conditioning right.** A pooled ATE averages across heterogeneity that may not be the relevant feature for the decision. The T-learner reveals heterogeneity but ignores controls; DML formalises both. Disagreements between the two are diagnostic.

**\"Significant heterogeneity\" is a statistical claim about a *function*, not a single number.** A wide CI on the CATE means the heterogeneity exists in the point estimates but the data cannot rule out a constant treatment effect. Reporting per-batch CATE intervals (Part 4) is the deliverable."""),

md("""## What's next

Lab 7B switches to the *per-cycle* slice and applies time-varying-treatment machinery: at each cycle, the capacity-drop is the proxy treatment, and the cell's future cycle life is the outcome."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch06" / "lab06b_lfp_cate.ipynb", cells)
print("Built lab06b_lfp_cate.ipynb")
