"""Build labs/ch02/lab02b.ipynb — back-door adjustment on the AI4I 2020 milling-machine dataset."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 2B — Back-Door Adjustment on Real Milling-Machine Data (AI4I 2020)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch02/lab02b.ipynb)

**Companion to Lab 2A.** Lab 2A built a hand-written SCM, identified the back-door paths from a DAG, applied the back-door criterion to choose an adjustment set, and verified the adjusted estimator recovers the known causal effect. **Lab 2B does the same loop on the AI4I 2020 Predictive Maintenance dataset, where the truth is unknown but the features have *named physical semantics* — unlike the anonymised station codes in many industrial datasets.**

The deliverable is a defensible back-door analysis of *one engineering question*: **does increasing rotational speed cause more machine failures?** We will state the DAG, defend it from physics, apply the adjustment, and produce the decision an engineer would actually act on.

**Dataset.** AI4I 2020 (Matzka, 2020). 10,000 samples from a milling machine with 5 numeric process variables (air temperature, process temperature, rotational speed, torque, tool wear), a product-Type categorical (L/M/H), a binary machine-failure label, and 5 specific failure-mode flags (TWF tool-wear failure, HDF heat-dissipation failure, PWF power failure, OSF overstrain failure, RNF random failure). CC BY 4.0."""),

md("""## What this lab is *not* doing

- **Discovering the DAG from data.** Chapter 9 covers that. Here we state an assumed DAG from physical knowledge of how a milling machine operates and use the back-door criterion to identify the effect under that DAG.
- **Picking the treatment.** We anchor on *rotational speed* because it is the most directly *controllable* process variable a line lead can set. Other choices (torque setpoint, recipe Type) are valid; the back-door logic carries over.
- **Disambiguating the 5 failure modes.** Lab 2B targets the *binary* `failure` label. The per-mode flags are saved for Lab 6B (CATE per failure mode) and Lab 10B (mediation through specific failure mechanisms).
- **Confidence intervals beyond standard logistic CI.** The chapter's bootstrap and influence-function CIs appear in Lab 5B (DML). Here we focus on identification, not inference machinery."""),

code("""# Lab 2B uses only standard scientific-Python + scikit-learn.
%pip install -q numpy pandas matplotlib scikit-learn"""),

code("""import os, sys, urllib.request, pathlib

# Pull the AI4I prep module and the vendored CSV from the repo.
DATA = pathlib.Path("/content")
for name in ("ai4i_prep.py", "ai4i2020.csv"):
    p = DATA / name
    if not p.exists():
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/{name}",
            p,
        )
sys.path.insert(0, str(DATA))

from ai4i_prep import load_ai4i

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load the chapter slice

The Ch 2 slice returns the cleaned AI4I frame with the `Type` categorical one-hot encoded (`type_M`, `type_H`; L is the reference)."""),

code("""df = load_ai4i(chapter=2)
print(f"Shape:        {df.shape}")
print(f"Failure rate: {df['failure'].mean():.3%}")
print()
print("Columns:")
for c in df.columns:
    print(f"  {c}")"""),

md("""## Part 2 — The trap: naive logistic of failure on rotational speed

A predictive-modelling pipeline that ignored the DAG would fit `failure ~ rot_speed_rpm` and read off the coefficient as the \"effect of speed on failure.\" Let us see what that says."""),

code("""def fit_logit_with_const(X, y, max_iter=2000):
    X = np.atleast_2d(X)
    if X.ndim == 1: X = X.reshape(-1, 1)
    m = LogisticRegression(max_iter=max_iter, )
    m.fit(X, y)
    return m

naive = fit_logit_with_const(df[["rot_speed_rpm"]].values, df["failure"].values)
naive_coef = float(naive.coef_[0, 0])

print(f"Naive logistic regression (failure ~ rot_speed_rpm):")
print(f"  Coefficient on rot_speed_rpm: {naive_coef:+.6f}  (log-odds per rpm)")
print(f"  Direction: {'+ (higher speed -> more failure)' if naive_coef > 0 else '- (higher speed -> less failure)'}")"""),

md("""**Stop and ask before continuing.** What does the sign of that coefficient even *mean*? It is the *conditional* association — among wafers that share whatever else the data captures (nothing, in this naive model), higher rpm corresponds to higher or lower failure odds. But: would a line lead who *increased* the rpm setpoint actually see more failures? That is a counterfactual question, and the naive coefficient does not answer it. The chapter says the way to bridge from correlation to counterfactual is to write down a DAG and check the back-door criterion."""),

md("""## Part 3 — The assumed DAG (defended from physics)

```
   type ──────► rot_speed_rpm    (M/H product variants are run at higher speeds)
       │ │
       │ └──► failure             (H variants have tighter tolerances -> higher reject rate)
       │
   tool_wear_min ─► rot_speed_rpm  (operators slow worn tools to avoid catastrophic failure)
            │ │
            │ └──► failure          (worn tools fail more often, all else equal)
            │
   torque_Nm ──► failure            (mechanical overload from torque drives one of the failure modes)
          │
          └─► tool_wear_min          (high torque accelerates wear)

   air_temp_K, process_temp_K ──► failure  (thermal failures HDF/PWF are temperature-driven)
                              │
                              └──► rot_speed_rpm  (operators adjust speed for thermal management)

   rot_speed_rpm ──► failure          (the direct effect we want to estimate)
```

**Defence of each edge from machine knowledge:**
- *Type → rot_speed_rpm, Type → failure*: the AI4I codebook (Matzka 2020) explicitly states that L/M/H variants have different operating envelopes and different per-variant failure rates by design.
- *tool_wear → rot_speed_rpm*: standard operator practice on a milling machine; this is a back-door route through operator behaviour.
- *torque → tool_wear*: physical wear accumulates faster under load.
- *temperatures → rot_speed_rpm*: thermal-management adjustments are real; not all sites do this but it is a plausible confounder.

**Back-door identification — a 30-second refresher.** A set $Z$ satisfies the back-door criterion for $X \\to Y$ if (1) no node in $Z$ is a descendant of $X$, and (2) $Z$ d-separates $X$ from $Y$ in the modified graph where the outgoing edges from $X$ are removed. The intuition is "$Z$ blocks every non-causal path from $X$ to $Y$ that goes 'around the back'". The set $Z = \\{\\text{type, tool\\_wear, torque, air\\_temp, process\\_temp}\\}$ satisfies both conditions for our DAG. None of these is downstream of speed (condition 1); each non-direct path from speed to failure goes through one of them as a chain or collider that conditioning blocks (condition 2)."""),

md("""## Part 4 — Adjusted logistic regression

**Why binarise the treatment at the median.** The chapter's worked examples target a binary $A \\in \\{0, 1\\}$ — a single counterfactual contrast (high vs low rotational speed). Three reasons we choose *median* specifically:

1. **Equal exposure groups by construction.** $P(A=1) \\approx 0.5$ guarantees the propensity $e(z) = P(A=1 \\mid Z=z)$ stays bounded away from 0 and 1 across most of $Z$-space. That keeps the back-door identification numerically stable (positivity holds with margin).
2. **Process-meaningful when no spec is given.** A real engineering analysis would dichotomise at a control-chart specification limit (e.g., "above the upper warning line"). AI4I's CC-BY codebook does not publish a tighter spec, so median is the conservative default — it tests the *direction* of the effect at the centre of the operating envelope, not at a particular limit.
3. **Comparable to the chapter's high/low convention.** Lab 2A uses median splits; using the same convention here keeps the two labs methodologically aligned.

The trade-off: binarising coarsens a continuous treatment. The ATE we estimate is the population-average effect of *crossing the median*, not of a specific RPM increment. A continuous-treatment version (with dose-response surfaces) is Lab 6B's territory."""),

code("""speed_threshold = float(df["rot_speed_rpm"].median())
T = (df["rot_speed_rpm"] >= speed_threshold).astype(int).values
Y = df["failure"].values
Z_cols = ["torque_Nm", "tool_wear_min", "air_temp_K", "process_temp_K", "type_M", "type_H"]
Z = df[Z_cols].values

print(f"Median rotational speed: {speed_threshold:.0f} rpm")
print(f"P(T=1 high speed):       {T.mean():.3f}")
print(f"P(failure | T=1):         {Y[T==1].mean():.3%}")
print(f"P(failure | T=0):         {Y[T==0].mean():.3%}")
print(f"Naive ATE (T=1 vs T=0):   {Y[T==1].mean() - Y[T==0].mean():+.3%}   (no adjustment)")"""),

code("""# Adjusted logistic regression: failure ~ T + Z.
X_full = np.hstack([T.reshape(-1, 1), Z])
adj = LogisticRegression(max_iter=4000, )
adj.fit(X_full, Y)
treatment_coef = float(adj.coef_[0, 0])

# Convert log-odds to a probability-scale ATE by averaging over the empirical Z.
def predict_prob_at_T(model, Z_, t):
    X_ = np.hstack([np.full((len(Z_), 1), t), Z_])
    return model.predict_proba(X_)[:, 1]

p1 = predict_prob_at_T(adj, Z, 1)
p0 = predict_prob_at_T(adj, Z, 0)
ate_adj = float(np.mean(p1 - p0))

print(f"Adjusted logistic coefficient on T: {treatment_coef:+.4f}  (log-odds)")
print(f"Adjusted ATE on probability scale:  {ate_adj:+.3%}   (back-door identified)")
print(f"Naive ATE (no adjustment):          {Y[T==1].mean() - Y[T==0].mean():+.3%}")
print()
shrinkage = 1 - abs(ate_adj) / max(abs(Y[T==1].mean() - Y[T==0].mean()), 1e-9)
print(f"Shrinkage from naive to adjusted:   {100*shrinkage:.1f}%")"""),

md("""**Read the shrinkage.** If the adjusted ATE is substantially smaller (in magnitude) than the naive ATE, the confounders $Z$ accounted for most of the apparent speed–failure relationship — i.e., the naive coefficient was capturing back-door paths, not the direct effect. If the adjusted ATE is comparable to or larger than the naive, the confounders are *not* explaining the relationship away, and the direct effect under the assumed DAG is real.

A sign flip between naive and adjusted (Simpson's paradox in this data) is *also* possible and worth checking — it would mean the unadjusted estimate had the wrong direction.

**A common surprising pattern in AI4I.** Operators *slow down worn tools* to avoid catastrophic failures. That induces a negative naive correlation between RPM and failure: high RPM correlates with healthy tools, low RPM with worn tools. The naive coefficient absorbs that backwards-feeling correlation. After adjusting for tool_wear (which we are doing), the direct RPM → failure effect can swing back toward positive (mechanical stress at high speed). If the adjusted estimate has a *different sign* from the naive estimate, you are seeing exactly this back-door inversion — *the operator's compensating behaviour was confounding the analysis*."""),

md("""## Part 5 — Sensitivity: adjustment-set robustness

The assumed DAG is *one* defensible reading of how the milling machine works. A robust analysis re-runs the back-door estimator under a few plausible alternative adjustment sets and reports the range of estimates."""),

code("""def adjusted_ate(Z_cols, df=df, T=T, Y=Y):
    Z_ = df[Z_cols].values if Z_cols else np.zeros((len(df), 0))
    X_ = np.hstack([T.reshape(-1, 1), Z_])
    m = LogisticRegression(max_iter=4000, )
    m.fit(X_, Y)
    p1 = m.predict_proba(np.hstack([np.ones((len(Z_), 1)), Z_]))[:, 1]
    p0 = m.predict_proba(np.hstack([np.zeros((len(Z_), 1)), Z_]))[:, 1]
    return float(np.mean(p1 - p0))

alternatives = [
    ("Naive (no Z)",                              []),
    ("Type only",                                 ["type_M", "type_H"]),
    ("Tool wear + Type",                          ["tool_wear_min", "type_M", "type_H"]),
    ("Mechanical (torque + tool_wear + Type)",    ["torque_Nm", "tool_wear_min", "type_M", "type_H"]),
    ("Thermal (temps + Type)",                    ["air_temp_K", "process_temp_K", "type_M", "type_H"]),
    ("Full assumed-DAG adjustment",               Z_cols),
]

rows = [{"adjustment_set": name, "ATE": adjusted_ate(cols), "|Z|": len(cols)}
        for name, cols in alternatives]
results = pd.DataFrame(rows)
print(results.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))"""),

md("""**The right way to read the table — concrete criteria for "stable" vs "drifting".**

| Pattern across rows | Interpretation | Action |
|---|---|---|
| All adjusted estimates within ~0.5 percentage points and same sign | Robust. Conclusion does not depend on the specific adjustment-set choice. | Report the full-DAG number; note the agreement. |
| Sign consistent but magnitude varies 2-3× across alternatives | Direction is robust; magnitude is not. | Report a *range*, not a point estimate. The right adjustment set is the one a domain expert defends. |
| Sign flips between alternatives (Simpson-style) | The DAG is doing all the work. | Stop and resolve the DAG question with a process engineer *before* publishing any estimate. |
| Naive row is far from all adjusted rows | Back-door adjustment matters. | Naive coefficient is what a predictive pipeline would have reported; the adjusted rows show what the chapter's machinery contributes. |

The naive row at the top of the table is the **null hypothesis for the value of back-door adjustment**. If the adjusted rows are indistinguishable from it, the confounders weren't doing much. If they are different, the chapter's machinery is the explanation."""),

md("""## Part 6 — Decision

Three bullets, the deliverable a line lead would read:

1. **Adjusted estimate of the effect of (high vs low) rotational speed on machine failure** (read from Part 4 — sign and magnitude in percentage points). The DAG assumption is named and defended; the back-door adjustment set is named and applied.

2. **Robustness across adjustment-set alternatives** (Part 5 table). If the sign and rough magnitude are consistent across the alternatives, the conclusion is robust to plausible DAG variants. If they diverge, name the disagreement and the DAG choice it depends on.

3. **What this estimate is not.** It is not the effect of any *specific* rotational-speed value, only of the (above-median vs below-median) binary contrast on the empirical Z distribution. It is not robust to an unmeasured confounder beyond $Z$ (Lab 13B's sensitivity-value framework applies if you want to bound that). And it does not say *which failure mode* the extra failures fall under — Lab 10B's mediation analysis decomposes that."""),

md("""## Reflection

**Real-data back-door is a DAG-defence exercise.** The estimator (logistic regression with the right covariates) is trivial; the hard part is naming and defending the DAG that says these covariates suffice. The AI4I codebook gives us named physical semantics for every column, which is a luxury — most industrial datasets (Bosch, SECOM) anonymise their features. The same back-door criterion applies in both cases; only the defence-of-the-DAG step gets harder when you cannot point to physics.

**Sensitivity to alternative adjustment sets is the part most analyses skip.** Reporting one adjusted number without showing how it shifts under reasonable DAG variants is the kind of brittle analysis that crashes when a domain expert proposes a different DAG. The cheap version of robustness (Part 5's adjustment-set sweep) is one re-fit per alternative."""),

md("""## What's next

Lab 4B uses the *temporal cohort* structure of LFP battery fast-charge protocols — different protocols introduced at different times across cells — to do a DID / IV analysis on cycle life. Lab 6B revisits AI4I-style heterogeneity with EconML's CATE meta-learners; Lab 10B uses Backblaze SMART telemetry to decompose drive-failure causes by mediation."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch02" / "lab02b.ipynb", cells)
print("Built lab02b.ipynb")
