"""Build labs/ch04/lab04b.ipynb — DID across Severson 2019 battery batches."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 4B — DID Across LFP Battery Batches (Severson 2019)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch04/lab04b.ipynb)

**Companion to Lab 4A.** Lab 4A built a synthetic experiment with a clean phased-rollout instrumental variable and verified that IV/DID estimators recover the known effect while naive regression does not. **Lab 4B applies the same machinery to a real natural experiment: three batches of LFP cells in Severson et al. 2019 received overlapping-but-distinct fast-charge protocol families over different collection periods.**

The deliverable is *not* a clean ATE — these batches are not randomised and the protocols are not orthogonal to collection time. The deliverable is **a DID analysis with its assumptions named, the parallel-trends check failed-or-passed explicitly, and an honest statement about which causal claim the data does and does not support**.

**Dataset.** 124 LFP cells from Severson et al. 2019 (replicated by Mattia 2021), in three collection batches (`train`, `test1`, `test2`) totalling 41 + 43 + 40 cells. Per cell we have the cycle-to-end-of-life (`cycle_life`) outcome and the Severson summary feature `log_var_deltaQ` (log variance of the discharge capacity curve difference between cycles 10 and 100), which is the dominant predictor in the original paper."""),

md("""## What this lab is *not* doing

- **Treating batch as a randomised treatment.** The batches were collected at different times with different protocol families and different ambient conditions in the test facility. We use batch *as the treatment* in the DID structure, and explicitly check the parallel-trends assumption knowing it may fail.
- **Per-protocol effect estimation.** Severson's 72 specific fast-charge protocols would let us estimate finer-grained protocol effects, but they require the original `.mat` files (not in this lab's vendored slice). The batch-level analysis is the coarsest version of the IV/DID story; per-protocol is a natural extension.
- **Survival analysis.** Cycle life is right-censored for the longest-lived cells in some original sources; the pre-processed slice ships the un-censored cycle_life values, treated as continuous outcomes."""),

code("""%pip install -q numpy pandas matplotlib scikit-learn statsmodels"""),

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
import statsmodels.api as sm

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load the cell-level summary"""),

code("""df = load_lfp(chapter=4)
print(f"Cells:                 {len(df)}")
print(f"Per-batch counts:\\n{df['batch'].value_counts().to_dict()}")
print(f"Cycle-life range:      {df['cycle_life'].min():.0f} - {df['cycle_life'].max():.0f}")
print(f"log_var_deltaQ range:  {df['log_var_deltaQ'].min():+.2f} - {df['log_var_deltaQ'].max():+.2f}")
print()
print(df.head(3).to_string())"""),

md("""## Part 2 — The trap: naive batch-mean comparison

A naive analyst would compute the mean cycle_life per batch and conclude that the batch with the highest mean has the \"best protocol family\". That ignores any pre-existing differences in cell condition (which we can see via `log_var_deltaQ`)."""),

code("""means = df.groupby("batch")["cycle_life"].agg(["mean", "std", "count"]).round(1)
print(means.to_string())
print()
print("Naive ranking by mean cycle_life:")
for b in means["mean"].sort_values(ascending=False).index:
    print(f"  {b}: {means.loc[b, 'mean']:.0f} cycles (n={int(means.loc[b, 'count'])})")"""),

md("""## Part 3 — The assumed selection-on-observables / DID structure

```
   batch (treatment cohort) ──► protocol-family ──► cycle_life
                            └──► (collection-period unobservables: tester drift,
                                  ambient temperature, lot-to-lot resistance) ──► cycle_life
```

**Identification claim (DID flavour).** If the only batch-to-batch difference *other than the protocol family* is captured by the pre-treatment summary `log_var_deltaQ` (the first-100-cycle capacity-curve signature), then a regression of `cycle_life` on `batch + log_var_deltaQ` recovers the batch-protocol effect as the batch dummies' coefficients.

**Why this is fragile.** That assumption is *exactly* what DID gets wrong if other batch-level unobservables remain — tester calibration drift, supplier resistance shifts, ambient changes. Part 5's parallel-trends-style check tries to surface this."""),

code("""# Use 'train' as the reference batch; estimate batch effects controlling for log_var_deltaQ.
df_design = pd.get_dummies(df[["batch", "log_var_deltaQ"]], columns=["batch"], drop_first=True)
df_design = df_design.astype(float)
df_design.insert(0, "const", 1.0)

ols = sm.OLS(df["cycle_life"].values, df_design).fit()
print(ols.summary().tables[1])"""),

md("""**Read the table.** The `batch_test1` and `batch_test2` coefficients are the DID-style estimates of *how many more (or fewer) cycles* a cell in those batches lives, holding `log_var_deltaQ` fixed at the same value. The `log_var_deltaQ` coefficient itself is the within-batch slope of cycle life on the early-cycle feature — Severson's published result is a strong negative slope, and we should recover the same sign here."""),

md("""## Part 4 — Parallel-trends-style check on the early-cycle feature

A clean DID assumes the treatment cohorts would have followed parallel trajectories absent the treatment. We don't have a pre-treatment period (the cells are randomised to fast-charge protocols at cycle 1), so we use the *pre-treatment covariate distribution* as a proxy: if the batches differ wildly in `log_var_deltaQ` distribution, the cells in each batch were not exchangeable to begin with, and the DID estimates are confounded by selection."""),

code("""fig, ax = plt.subplots(figsize=(7, 4))
for b in ["train", "test1", "test2"]:
    sub = df[df["batch"] == b]
    ax.scatter(sub["log_var_deltaQ"], sub["cycle_life"], label=f"{b} (n={len(sub)})", alpha=0.65)
ax.set_xlabel("log_var_deltaQ (Severson early-cycle feature)")
ax.set_ylabel("cycle_life (cycles to EOL)")
ax.set_title("Per-batch relationship between early-cycle signature and EOL")
ax.legend()
plt.tight_layout()
plt.show()

# Quantitative covariate-overlap check.
overlap = df.groupby("batch")["log_var_deltaQ"].agg(["mean", "std", "min", "max"]).round(3)
print(overlap.to_string())"""),

md("""**The right way to read the plot.** If the three batches occupy roughly the same region of the x-axis and the slope `cycle_life ~ log_var_deltaQ` looks roughly parallel across batches, the DID estimates from Part 3 are at least *consistent* with the parallel-trends assumption. If the batches are in distinct x-regions or have distinct slopes, the assumption is doubtful and the Part-3 numbers need a much larger uncertainty band than the OLS standard errors suggest."""),

md("""## Part 5 — IV interpretation (sketch)

A pure IV reading would treat batch assignment as an *instrument* for the protocol family received: batch causes protocol-family-membership which causes cycle life. This requires:
1. Relevance: batch strongly predicts protocol family. **Yes by design.**
2. Exclusion: batch affects cycle life *only* through protocol family. **Probably not** — collection-period unobservables (calibration drift, ambient) violate exclusion.

Because exclusion is doubtful, the IV reading would inflate the estimate by the same confounding the DID is trying to control for. We report the DID result, name the exclusion violation, and treat the IV reading as inconsistent with the data."""),

md("""## Part 6 — Decision

Three bullets, the deliverable a battery-engineering team would read:

1. **Batch-level effects on cycle life** (read off `batch_test1` and `batch_test2` coefficients from Part 3) are the DID-style estimates *if* `log_var_deltaQ` captures all batch-orthogonal differences in initial cell condition. The point estimates are useful for ranking; the magnitudes carry meaningful uncertainty.

2. **The parallel-trends-style check in Part 4** is either passing or failing in your run — if the batches occupy distinct regions of the feature space, the DID estimate is confounded by selection and should not be reported as a protocol effect. If the regions overlap, the estimate is at least internally consistent.

3. **A defensible follow-up** would be to (a) get the original protocol assignments per cell (from data.matr.io) and re-estimate at the protocol level, (b) add observed cell-level covariates beyond `log_var_deltaQ` (cycle-1 capacity, internal resistance), and (c) consider a propensity-score-weighted version of the DID, since the implicit assumption of common support across batches is what Part 4 stress-tests."""),

md("""## Reflection

**Natural experiments are rarely clean experiments.** The Severson batches *look* like an IV/DID setup — different protocols across periods — but the protocols, the collection periods, and the operating conditions are entangled in ways that the published `cycle_life` numbers alone cannot disentangle. The DID estimator is a tool; whether its assumptions hold is a domain question.

**Reporting the diagnostic IS the deliverable.** A team that reports DID estimates without the Part-4 overlap check is reporting numbers without context. A team that reports the overlap check together with the estimate is reporting an analysis."""),

md("""## What's next

Lab 6B uses the same LFP summary slice with EconML's CATE meta-learners to ask *whether the effect of `log_var_deltaQ` on cycle life is heterogeneous across batches*. Lab 7B switches from the cell-level summary to the per-cycle trajectory and applies time-varying-treatment machinery."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch04" / "lab04b.ipynb", cells)
print("Built lab04b.ipynb")
