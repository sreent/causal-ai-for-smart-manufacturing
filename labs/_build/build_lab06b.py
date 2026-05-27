"""Build labs/ch06/lab06b.ipynb — CATE on LFP cells, with batch as the effect modifier."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 6B — CATE of the Severson Early-Cycle Feature on Cycle Life

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06b.ipynb)

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

md("""## Part 2 — Marginal (pooled) effect: the ATE baseline

A single linear regression of `cycle_life` on `log_var_deltaQ` gives the pooled effect — the average across all batches. This is the chapter's S-learner with no covariates."""),

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

md("""**How to read the comparison.**

- If T-learner and DML-CATE *agree* per batch and *differ from the pooled ATE*, the heterogeneity is real and well-identified.
- If they disagree, the DML version's variance-reducing nuisance correction is doing something the within-batch OLS missed — most likely controlling for the other capacity features acting as confounders.
- If both equal the pooled ATE within sampling error, there is no meaningful batch-level heterogeneity and the chapter's CATE machinery is overkill here."""),

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

write_notebook(pathlib.Path(__file__).parent.parent / "ch06" / "lab06b.ipynb", cells)
print("Built lab06b.ipynb")
