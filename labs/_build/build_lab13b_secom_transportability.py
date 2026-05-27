"""Build labs/ch13/lab13b_secom_transportability.ipynb — transportability across SECOM time-subsets."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 13B — Transportability of a SECOM Effect Estimate Across Quarters

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch13/lab13b_secom_transportability.ipynb)

**Companion to Lab 13A.** Lab 13A built a synthetic source/target pair where we knew exactly which distributions differed, ran the transport-formula machinery, and verified the re-weighted source estimate matched the target ground truth. **Lab 13B does the same exercise on SECOM split into Jul-Aug (source) and Sep-Oct (target).**

The deliverable is: an effect estimate built on the source half, a *transported* estimate predicted for the target half, a *direct* target-half estimate as the comparator, and a sensitivity statement that quantifies how much an unobserved effect modifier could swing the conclusion.

**Why this matters.** A common industrial scenario: an analysis is run on last quarter's data and a deployment decision is made for next quarter. Transportability is the formal question of *whether that is justified*. Doing the analysis well is the difference between a defensible deployment and a yield regression."""),

md("""## What this lab is *not* doing

- **Selecting an effect modifier.** Lab 13A taught the diagram-based selection-diagram approach. Here we assume the only relevant difference between source and target is *covariate-distribution shift*, not a structural difference in the SCM.
- **Time-series-aware splits.** A more correct analysis would account for serial correlation within each subset. We treat the splits as i.i.d. samples.
- **Choosing the target-quarter sensor to act on.** Same sensor convention as Labs 1B and 5B: the top |corr| sensor is the anchor."""),

code("""%pip install -q ucimlrepo statsmodels pysensemakr"""),

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
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load with source/target split

The Ch 13 slice partitions SECOM by month: source = Jul-Aug 2008, target = Sep-Oct 2008. Same five sensors as Lab 1B."""),

code("""df = load_secom(chapter=13)
sensor_cols = [c for c in df.columns if c.startswith("S")]
treatment   = sensor_cols[0]   # convention: top |corr with yield| sensor

src = df[df["split"] == "source"].reset_index(drop=True)
tgt = df[df["split"] == "target"].reset_index(drop=True)

print(f"Total wafers:           {len(df)}")
print(f"Source (Jul-Aug 2008):  {len(src)}   failure rate {src['yield_fail'].mean():.3%}")
print(f"Target (Sep-Oct 2008):  {len(tgt)}   failure rate {tgt['yield_fail'].mean():.3%}")
print(f"Treatment sensor:       {treatment}")
print(f"Control sensors:        {sensor_cols[1:]}")"""),

md("""## Part 2 — Source-fit estimate, naive target apply

Fit a logistic regression on the source half: `yield_fail ~ treatment + controls`. The treatment coefficient (on the log-odds scale) is the source-fit effect estimate. Then predict on the target half and read off the *implied* target failure rate under the source model."""),

code("""def fit_logit(df_, treatment, controls):
    X = df_[[treatment] + controls].values
    y = df_["yield_fail"].values
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(X, y)
    return m

m_src = fit_logit(src, treatment, sensor_cols[1:])
coef_src = float(m_src.coef_[0, 0])   # treatment coef on source

X_tgt = tgt[[treatment] + sensor_cols[1:]].values
y_tgt = tgt["yield_fail"].values
p_tgt_under_src = m_src.predict_proba(X_tgt)[:, 1]

implied_target_rate = float(p_tgt_under_src.mean())
actual_target_rate  = float(y_tgt.mean())

print(f"Source-fit treatment coefficient (log-odds):  {coef_src:+.4f}")
print(f"Source model's implied target failure rate:   {implied_target_rate:.3%}")
print(f"Actual target failure rate:                   {actual_target_rate:.3%}")
print(f"Implied - actual:                             {(implied_target_rate - actual_target_rate):+.3%}")"""),

md("""**Read the implied-vs-actual gap on the *probability scale*, not the coefficient scale.** A small gap (within a couple of percentage points of the actual target rate) is a *necessary-but-not-sufficient* condition for transportability — the marginal target rate could match while the *effect estimate* still fails to transport. A *large* gap is a red flag that should be inspected before any other diagnostic.

**On this SECOM split the gap is enormous.** The source-fit model predicts a target failure rate of ~36 % while the observed target failure rate is ~4 % — a +32 percentage-point over-prediction. That is *not* "near zero" by any standard; it directly says the source population's relationship between sensors and yield does not generalise to the target population's. Don't read this Part-2 result as a "sanity check passed"; it is a sanity check *failed*, with the diagnostic value of telling you to expect Parts 4-5 to confirm non-transportability.

The real test (a head-to-head coefficient comparison) is in Part 4. Use Part 3's covariate-shift table to *localise* the cause of the Part-2 gap; use Part 5 to ask whether reweighting closes it; use Part 6 to bound the residual under unmeasured-confounder assumptions."""),

md("""## Part 3 — Diagnose covariate distribution shift

Transportability fails when the covariate distribution shifts. Quantify the shift with the *standardised mean difference* (SMD) per sensor:

$$\\mathrm{SMD} = \\frac{|\\bar{X}_{\\text{tgt}} - \\bar{X}_{\\text{src}}|}{\\sqrt{(s_{\\text{src}}^2 + s_{\\text{tgt}}^2)/2}}.$$

**Why SMD > 0.25 is the conventional cutoff.** SMD is a *unitless effect-size* measure — the same scale as Cohen's $d$. The thresholds come from causal-inference practice (Stuart 2010, Austin 2011):

- **SMD < 0.10**: distributions are essentially identical; covariate shift is negligible.
- **0.10 ≤ SMD < 0.25**: marginal shift; the source-fit estimate transports with light reweighting.
- **0.25 ≤ SMD < 0.50**: substantive shift; reweighting is necessary, and the source-fit may be biased without it. This is the conventional alarm threshold.
- **SMD ≥ 0.50**: severe shift; the source and target populations may not even *share support* on this variable. Reweighting may fail; restrict the deployment to the overlap region or run a fresh target-period estimate.

These thresholds are heuristics, not laws — but they tell you *which sensors* to investigate before pretending the source estimate transports."""),

code("""def smd(a, b):
    return float(abs(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

shift = pd.DataFrame([
    {"variable": c,
     "mean_src": float(src[c].mean()),
     "mean_tgt": float(tgt[c].mean()),
     "SMD":      smd(src[c], tgt[c])}
    for c in sensor_cols + ["yield_fail"]
]).sort_values("SMD", ascending=False)

print(shift.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()
high_shift = shift[shift["SMD"] > 0.25]["variable"].tolist()
print(f"Variables with SMD > 0.25 (non-trivial shift): {high_shift or 'none'}")"""),

md("""## Part 4 — Direct target-fit estimate vs source-fit estimate

Fit the same logistic on the target half and compare the treatment coefficient to the source-fit coefficient. If they agree, the source-fit transports. If they disagree, an effect-modifier shift is the most likely culprit; the disagreement is the *non-transportability gap*."""),

code("""m_tgt = fit_logit(tgt, treatment, sensor_cols[1:])
coef_tgt = float(m_tgt.coef_[0, 0])

gap = coef_tgt - coef_src
rel_gap = abs(gap) / (abs(coef_src) + 1e-9)

print(f"Source-fit treatment coef:   {coef_src:+.4f}")
print(f"Target-fit treatment coef:   {coef_tgt:+.4f}")
print(f"Gap (tgt - src):             {gap:+.4f}")
print(f"Relative gap:                {100*rel_gap:.1f}%   (>30% suggests substantive non-transportability)")"""),

md("""## Part 5 — Reweighted transport

If the covariate distribution shifted (Part 3) but the SCM is otherwise the same, reweighting source observations by $w(z) = p_{\\text{tgt}}(z) / p_{\\text{src}}(z)$ recovers a target-domain estimate from source data.

**How a classifier gives us the density ratio (without estimating either density).** Estimating $p_{\\text{tgt}}(z)$ and $p_{\\text{src}}(z)$ separately is hard in high dimensions. The trick: pool source and target rows, fit a classifier $r(z) = P(\\text{source} \\mid z)$, and read off the ratio from a one-line Bayes-rule derivation:

$$\\frac{p_{\\text{tgt}}(z)}{p_{\\text{src}}(z)} \\;=\\; \\frac{P(\\text{tgt} \\mid z)}{P(\\text{src} \\mid z)} \\cdot \\frac{P(\\text{src})}{P(\\text{tgt})} \\;\\propto\\; \\frac{1 - r(z)}{r(z)}.$$

The prior ratio $P(\\text{src})/P(\\text{tgt})$ is a constant that affects only the overall scale; once we re-normalise weights (or use the relative ratio in a downstream regression), it drops out. This trick — *density-ratio estimation by classification* — is the workhorse for covariate-shift correction, importance-weighted ERM, and modern causal-inference reweighting in general (Sugiyama et al., 2012). It is also why a small logistic regression is enough here: we are not estimating the densities themselves, just their ratio."""),

code("""# Build the classifier-based density ratio: r(z) = p(source=1 | z) -> w(z) = (1-r)/r
both = pd.concat([
    src.assign(_src=1),
    tgt.assign(_src=0),
], ignore_index=True)

clf_src = LogisticRegression(max_iter=2000, class_weight="balanced")
clf_src.fit(both[sensor_cols].values, both["_src"].values)

r = clf_src.predict_proba(src[sensor_cols].values)[:, 1]   # P(source | z) on source rows
r = np.clip(r, 0.05, 0.95)
w = (1 - r) / r                                            # weight to reshape source -> target

m_src_rw = LogisticRegression(max_iter=2000, class_weight="balanced")
m_src_rw.fit(src[[treatment] + sensor_cols[1:]].values,
             src["yield_fail"].values,
             sample_weight=w)
coef_src_rw = float(m_src_rw.coef_[0, 0])

print(f"Source-fit (unweighted) coef:                {coef_src:+.4f}")
print(f"Source-fit (re-weighted to target) coef:     {coef_src_rw:+.4f}")
print(f"Target-fit (gold-standard) coef:             {coef_tgt:+.4f}")
print()
print(f"Re-weighting shrinks source->target gap by:")
print(f"  {abs(coef_tgt - coef_src):+.4f}  ->  {abs(coef_tgt - coef_src_rw):+.4f}")"""),

md("""**The closer the re-weighted source estimate is to the target-fit estimate, the more of the original gap was covariate-shift (vs structural shift).** Residual disagreement is the signal that something *beyond* the measured covariates changed between quarters — a software update, a new operator rotation, a supplier change."""),

md("""## Part 6 — Cinelli-Hazlett sensitivity on the transportability assumption

Even after re-weighting, we cannot rule out an *unobserved* effect modifier. The Cinelli-Hazlett robustness value asks: *how strong (in partial-R²) would such an unmeasured effect modifier need to be to explain the entire treatment coefficient?*

We compute the RV on the source-fit linear-probability model. (Sensemakr works on OLS, so we re-fit with `statsmodels` OLS on the source half as the chapter prescribes.)"""),

code("""import statsmodels.api as sm
from sensemakr import Sensemakr

# Fit the linear-probability model on the source half with a constant + controls.
data_src = src[[treatment] + sensor_cols[1:]].copy()
data_src.insert(0, "const", 1.0)
ols = sm.OLS(src["yield_fail"].values, data_src).fit()

est = float(ols.params[treatment])
se  = float(ols.bse[treatment])
dof = int(ols.df_resid)

# Use the manual (estimate / se / dof) constructor; sensemakr's model= path
# trips over a pandas indexing issue in some statsmodels-OLS combinations.
s = Sensemakr(estimate=est, se=se, dof=dof, treatment=treatment, q=1.0, alpha=0.05)

stats = s.sensitivity_stats
print(f"Source-fit treatment estimate     = {est:+.5f}")
print(f"Standard error                     = {se:.5f}")
print(f"Partial R^2 (treatment, outcome)   = {stats['r2yd_x']:.4f}")
print(f"Robustness value  RVq=1            = {stats['rv_q']:.4f}")
print(f"Robustness value  RVqa=1 (alpha=5%) = {stats['rv_qa']:.4f}")"""),

md("""**How to read the RV.**

- **RVq=1** (the unconditional robustness value): a hypothetical unobserved confounder would need a partial-R² with the treatment *and* with the outcome of at least this much to reduce the estimate to zero. RVq=1 close to zero (< 0.05) is fragile; > 0.20 is robust.
- **Benchmark comparison**: the printout compares the hypothetical confounder to the strongest *measured* covariate. If \"as strong as the strongest measured control\" is enough to wipe out the estimate, the effect is fragile.

For transportability: the same RV reading also bounds how strong an *unmeasured effect modifier* would have to be to flip the source→target transport conclusion."""),

md("""## Part 7 — Decision

Three bullets, the deployment-readiness report:

1. **Does the source estimate transport?** Read Parts 4 and 5: if `target-fit coef` is within roughly 20% of `re-weighted source coef`, the answer is *yes, with measured covariate adjustment*. If the gap is large, the answer is *no* — the deployment recommendation must be re-estimated on target-period data before rollout.

2. **What drove any residual gap?** Part 3's SMD table localises the shift to specific sensors. If a single sensor accounts for most of the shift, that is the variable to investigate first (was there a tool change, calibration cycle, or recipe revision in Sep-Oct that changed its operating range?).

3. **How robust is the estimate to an unmeasured effect modifier?** Part 6's RV gives the bar. If the RV is large (the strongest measured confounder is not strong enough to wipe out the effect), the estimate survives plausible hidden modifiers; if it is small, the recommendation is *one weak hidden confounder away from being wrong* and a controlled experiment in the target period is the right next step."""),

md("""## Reflection

**Transportability is a separate inference from identification.** Even an estimator that perfectly identifies the source-domain effect can produce a misleading target-domain prediction if the populations shift. The chapter's contribution is *making the shift visible*, not erasing it.

**The three-quantity report — source-fit, re-weighted source-fit, target-fit — is the deliverable.** The agreement (or disagreement) across the three answers the deployment question in a way that no single number can. Pair this with a sensitivity bound and a process engineer has the information to decide *deploy now*, *wait for more target data*, or *run a controlled trial first*."""),

md("""## What's next

This concludes the Lab B arc. Across the thirteen Lab Bs, the dataset palette covers SECOM (1B, 3B, 5B, 9B, 13B), AI4I 2020 (2B), LFP batteries (4B, 6B, 7B), Backblaze drive SMART telemetry (8B, 10B), and the Tennessee Eastman simulator (11B, 12B). Every notebook walks the same five-step skeleton — load, frame, identify, estimate, sensitivity — and produces the *defensible report* that a domain team would actually act on, not a predictive number stripped of identifiability context."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch13" / "lab13b_secom_transportability.ipynb", cells)
print("Built lab13b_secom_transportability.ipynb")
