"""Build labs/ch14/lab14h_multisite.ipynb — guided capstone Starter F on multi-site synthetic data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402


def reveal(content: str) -> dict:
    return md("<details>\n<summary><b>Click to reveal sample answer</b></summary>\n\n"
              + content + "\n\n</details>")


cells = [

md("""# Lab 14H — Guided Capstone (Starter F): Transportability with Effect-Modifier Shift Across Plants

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14h_multisite.ipynb)

**Starter F** of the §14.7 capstone. This is the **deployment-flavoured verifiable** starter: real plants face transportability questions all the time (does my Plant A analysis apply to Plant B?), but on real data the ground truth is unknown. This capstone uses the `multisite_synthetic.py` generator where the *true* ATE at each plant is analytically known, so the student can validate their reweighting against the ground truth.

**The capstone question.** *Plant A's controlled trial estimates an ATE of about +0.21 from the intervention. Plant B is about to deploy. Should they trust Plant A's estimate, or re-run the trial at B?*

**Companion labs.** Lab 13B (transportability across SECOM quarters) is the methodological reference; the `multisite_synthetic.py` ground-truth helper is the validation tool.

**Why this starter is "verifiable + deployment-flavoured".** The synthetic data documents an effect-modifier shift between the two plants (Plant A's raw_grade distribution is Beta(2,5) — low-grade-heavy; Plant B's is Beta(5,2) — high-grade-heavy). The treatment effect scales with raw_grade, so the per-plant ATEs differ by construction. The student's reweighting must recover Plant B's true ATE from Plant A's data alone. If it does, the transportability machinery is working; if it doesn't, the shift is more complex than effect-modifier-only and the analysis must escalate."""),

md("""## Setup"""),

code("""%pip install -q numpy pandas matplotlib scikit-learn statsmodels"""),

code("""import os, sys, urllib.request, pathlib

DATA = pathlib.Path("/content")
if not (DATA / "multisite_synthetic.py").exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sreent/causal-ai-for-smart-manufacturing/main/labs/data/multisite_synthetic.py",
        DATA / "multisite_synthetic.py",
    )
sys.path.insert(0, str(DATA))

from multisite_synthetic import load_multisite, true_ate_per_site
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression

rng = np.random.default_rng(0)
df = load_multisite(n_per_site=1000, seed=0)
print(f'Generated {len(df)} units; site distribution:\\n{df[\"site\"].value_counts().to_dict()}')
print(f'\\nGround truth from SCM:')
for k, v in true_ate_per_site().items():
    print(f'  {k}: {v:+.4f}')"""),

md("""## Artifact 1 — Problem Statement and DAG

### Q1.1 Write the problem statement.

*Hint.* The decision involves a deployment risk: if Plant A's estimate transports cleanly, Plant B can adopt the intervention without a separate trial — saving 6 months of evaluation. If it doesn't, Plant B's pre-deployment investment is justified.

*Your turn.*"""),

md("""*[Problem statement.]*"""),

reveal("""Plant A ran a controlled trial of a process intervention with ATE ≈ +0.21 (on a continuous yield-related outcome). Plant B is identical to A in equipment and SCM structure but receives a different raw-material grade distribution (Plant B sources higher-grade material on average). The operations leadership team wants to know whether Plant A's ATE estimate transports to Plant B's population without a separate trial, or whether the raw-grade distribution shift invalidates the source-to-target generalisation. If the transported estimate falls within ~20% of Plant B's direct estimate, transport is supported and Plant B can adopt the intervention. If the gap is larger, Plant B must run its own trial before deployment."""),

md("""### Q1.2 Draw the transportability DAG.

*Hint.* The classic selection-diagram form: a "site" node points at the variables that differ across sites. Here, site → raw_grade (the effect modifier), but NOT site → treatment effect mechanism.

*Your turn.*"""),

md("""*[DAG.]*"""),

reveal("""```
   site ──► raw_grade (effect modifier)   [Beta(2,5) at A; Beta(5,2) at B]

   raw_grade ──┬──► outcome (baseline shift)
               └──► tau (treatment-effect scales with raw_grade)

   treatment ──► outcome  (randomised; ATE = tau(raw_grade))

   tau, treatment ──► outcome  (effect modification)
```

The defining feature: `site` affects the *distribution* of `raw_grade`, but the SCM equations (how raw_grade affects outcome, how treatment affects outcome) are *identical* across sites. This is the "covariate shift only" scenario — the cleanest case for transportability."""),

md("""### Q1.3 What would break the transportability assumption?

*Your turn.*"""),

md("""*[Statement of what would break the assumption.]*"""),

reveal("""**Site → treatment-effect mechanism directly.** If Plant B's machines respond to the intervention *differently* than Plant A's (e.g., older equipment, different operator training), then $\\tau(\\text{raw\\_grade}, \\text{site} = A) \\neq \\tau(\\text{raw\\_grade}, \\text{site} = B)$ and no reweighting on observed covariates can recover Plant B's ATE from Plant A's data.

In the synthetic dataset, this is *not* the case — the SCM is identical across sites — so the reweighted estimate should match the ground truth. On real data, a structural difference of this kind would manifest as the reweighted estimate *still* disagreeing with the direct Plant-B estimate, even after accounting for raw_grade."""),

md("""## Artifact 2 — Estimand

### Q2.1 Write the transported-ATE estimand.

*Your turn.*"""),

md("""*[Estimand.]*"""),

reveal("""$$\\text{ATE}_B^{\\text{transported}} \\;=\\; E_{\\text{raw\\_grade} \\sim p_B(\\text{raw\\_grade})}\\big[\\tau(\\text{raw\\_grade})\\big],$$

where $\\tau(\\text{raw\\_grade})$ is the conditional-on-raw-grade treatment effect (estimable from Plant A) and $p_B$ is Plant B's marginal distribution of raw_grade (observable from Plant B without running a treatment trial)."""),

md("""### Q2.2 Couple to a decision threshold.

*Your turn.*"""),

md("""*[Decision rule.]*"""),

reveal("""- $|\\text{ATE}_B^{\\text{transported}} - \\text{ATE}_B^{\\text{direct}}| < 0.05$: transport is **supported**; Plant B can adopt without a separate trial.
- $0.05 \\leq |\\text{gap}| < 0.15$: transport is **partial**; Plant B should validate on a pilot of 100-200 units before full deployment.
- $|\\text{gap}| \\geq 0.15$: transport **fails**; Plant B must run its own controlled trial before any deployment."""),

md("""## Artifact 3 — Identification

### Q3.1 Name the transport strategy.

*Hint.* For "covariate shift only" — the SCM is the same; only the marginal distribution of effect modifiers differs — the strategy is *reweighting* by the source-to-target density ratio.

*Your turn.*"""),

md("""*[Strategy + assumptions.]*"""),

reveal("""**Strategy.** Reweighted source-domain estimation. Estimate the conditional treatment effect $\\tau(z)$ on Plant A, then average over Plant B's covariate distribution by weighting Plant A's units by $w(z) = p_B(z) / p_A(z)$. Identification requires:

1. *S-admissibility:* the SCM equations are the same across sites; only covariate marginals differ.
2. *Positivity of the density ratio:* every region of raw_grade that has support at Plant B also has support at Plant A. With Beta(2,5) and Beta(5,2) this is satisfied.
3. *Identification at the source:* Plant A's $\\tau(z)$ is itself causally identified (here, by randomisation within Plant A)."""),

md("""### Q3.2 What would invalidate S-admissibility, in observed data?

*Your turn.*"""),

md("""*[How to detect S-admissibility violation.]*"""),

reveal("""The cleanest detection: compute *direct* Plant-B ATE (which requires Plant B to have its own randomised arm) and compare to the *reweighted* Plant-A estimate. If they agree, S-admissibility holds; if they disagree, either there is a structural difference between the two plants OR an effect modifier shift we did not account for. In production, the Plant-B direct estimate is what the team would build *before* relying on Plant A's reweighted version — exactly the kind of pre-deployment validation Artifact 6's checklist would specify."""),

md("""## Artifact 4 — Estimator

### Q4.1 Estimate $\\tau(z)$ on Plant A.

*Hint.* Fit `outcome ~ treatment * raw_grade` on Plant A's data. The interaction coefficient is $\\partial\\tau/\\partial z$.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
src = df[df['site'] == 'A'].copy()
src['tx_grade'] = src['treatment'] * src['raw_grade']
X = sm.add_constant(src[['treatment', 'raw_grade', 'tx_grade']])
ols_A = sm.OLS(src['outcome'].values, X).fit()
print(ols_A.summary().tables[1])
# tau(z) = tau_intercept + tau_slope * z, read off the 'treatment' + 'tx_grade' coefficients
tau_intercept_est = float(ols_A.params['treatment'])
tau_slope_est     = float(ols_A.params['tx_grade'])
print(f'\\nLearned tau(z) = {tau_intercept_est:+.4f} + {tau_slope_est:+.4f} * z')
print(f'True tau(z)    = +0.10 + +0.40 * z (from the SCM)')
```
"""),

md("""### Q4.2 Compute the density ratio $p_B(z) / p_A(z)$ via classifier.

*Hint.* The classifier trick from Lab 13B: pool A and B; fit $r(z) = P(\\text{source} = A \\mid z)$; the density ratio is $(1 - r) / r$.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
# Pool A and B for the classifier.
pooled = df.copy()
pooled['is_A'] = (pooled['site'] == 'A').astype(int)
clf = LogisticRegression(max_iter=1000).fit(pooled[['raw_grade']].values, pooled['is_A'].values)

# Density ratio for each Plant-A unit (weight A units to look like B).
r = clf.predict_proba(src[['raw_grade']].values)[:, 1]  # P(A | z) for source units
r = np.clip(r, 0.05, 0.95)
w = (1 - r) / r                                          # p_B(z) / p_A(z) up to normalisation
w = w / w.mean()                                         # Hajek normalisation
print(f'Density-ratio weight stats: min {w.min():.3f}, mean {w.mean():.3f}, max {w.max():.3f}')
print(f'Effective sample size (Kish): {(w.sum()**2 / (w**2).sum()):.0f} of {len(w)}')
```
"""),

md("""### Q4.3 Compute the transported ATE: reweighted Plant-A vs direct Plant-B.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
# Reweighted Plant-A: fit tau(z) on A, average over Plant-B's marginal via the weights.
src['tau_hat'] = tau_intercept_est + tau_slope_est * src['raw_grade'].values
ate_A_transported = float(np.sum(w * src['tau_hat']) / np.sum(w))

# Direct Plant-B: fit the same model on B.
tgt = df[df['site'] == 'B'].copy()
tgt['tx_grade'] = tgt['treatment'] * tgt['raw_grade']
ols_B = sm.OLS(tgt['outcome'].values, sm.add_constant(tgt[['treatment', 'raw_grade', 'tx_grade']])).fit()
tau_B_int = float(ols_B.params['treatment'])
tau_B_sl  = float(ols_B.params['tx_grade'])
ate_B_direct = tau_B_int + tau_B_sl * tgt['raw_grade'].mean()

# Naive: use Plant-A's ATE directly, ignoring grade distribution shift.
ate_A_direct = tau_intercept_est + tau_slope_est * src['raw_grade'].mean()

truth = true_ate_per_site()
print(f'Plant-A naive (apply A to B):           {ate_A_direct:+.4f}  (truth ATE_A = {truth[\"ATE_A\"]:+.4f})')
print(f'Plant-A reweighted to B (transported):  {ate_A_transported:+.4f}')
print(f'Plant-B direct (gold standard):         {ate_B_direct:+.4f}  (truth ATE_B = {truth[\"ATE_B\"]:+.4f})')
print(f'\\nGap |reweighted - direct|: {abs(ate_A_transported - ate_B_direct):.4f}')
```
"""),

md("""**Discussion.** The reweighting moves the source estimate from ATE_A (~0.21) toward ATE_B (~0.39) by accounting for Plant B's higher raw_grade distribution. If the reweighted estimate lands within 0.05 of the direct Plant-B estimate, the transportability is **clean** — Plant A's data was sufficient. If the gap exceeds 0.15, the structural assumptions failed somewhere; investigate which."""),

md("""## Artifact 5 — Sensitivity Analysis

### Q5.1 Bound the impact of an unmeasured effect modifier.

*Hint.* The reweighted estimate accounts for `raw_grade`. If there is a *different* effect modifier (operator skill, ambient conditions) that ALSO shifts across plants but is unmeasured, the reweighted estimate is biased. Simulate a modifier of varying strength and report how the reweighted estimate moves.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
# Add a synthetic hidden modifier 'hidden' that ALSO shifts treatment effect, with
# different marginal distribution across sites.
rng_sens = np.random.default_rng(2)
for hidden_strength in [0.0, 0.2, 0.4]:
    df_sens = df.copy()
    df_sens['hidden'] = np.where(df_sens['site'] == 'A',
                                  rng_sens.beta(2, 5, len(df_sens)),
                                  rng_sens.beta(5, 2, len(df_sens)))
    df_sens['outcome'] += hidden_strength * df_sens['hidden'] * df_sens['treatment']

    src_s = df_sens[df_sens['site'] == 'A'].copy()
    tgt_s = df_sens[df_sens['site'] == 'B'].copy()
    src_s['tx_grade'] = src_s['treatment'] * src_s['raw_grade']
    ols_sens = sm.OLS(src_s['outcome'].values,
                       sm.add_constant(src_s[['treatment', 'raw_grade', 'tx_grade']])).fit()
    src_s['tau_hat'] = ols_sens.params['treatment'] + ols_sens.params['tx_grade'] * src_s['raw_grade']
    transported = float(np.sum(w * src_s['tau_hat'].values) / np.sum(w))
    direct_B = (tgt_s['outcome'][tgt_s['treatment']==1].mean() - tgt_s['outcome'][tgt_s['treatment']==0].mean())
    print(f'hidden_strength = {hidden_strength}: transported {transported:+.4f}, direct_B {direct_B:+.4f}, gap {abs(transported - direct_B):.4f}')
```
"""),

md("""### Q5.2 Verdict.

*Your turn.*"""),

md("""*[Verdict on transport reliability.]*"""),

reveal("""If the gap stays under 0.05 across `hidden_strength` $\\in [0, 0.4]$, the transport is **robust** to unmeasured effect-modifier shifts of moderate strength. If the gap grows above 0.10 at strength 0.2, the transport is **fragile**; Plant B should run its own trial.

For this synthetic dataset (no hidden effect modifier in the SCM at strength=0), the reweighted estimate matches the direct estimate to within ~0.01 by construction. The sensitivity sweep simulates what would happen on real data where a hidden modifier might exist."""),

md("""## Artifact 6 — Deployment-Readiness Checklist

### Q6.1 Target population + deployment scope.

*Your turn.*"""),

md("""*[Target population + scope.]*"""),

reveal("""**Target population.** Production units at Plant B, processed under Plant B's raw-material sourcing in the current quarter. Deployment to a different quarter (or a third plant) requires re-running the transport step with that target's covariate distribution.

**Scope.** This deployment relies on the assumption that Plant B's machines respond to the intervention identically to Plant A's. If Plant B has a *different* generation of equipment, a separate trial at Plant B is necessary before deployment."""),

md("""### Q6.2 Three monitors with thresholds.

*Your turn.*"""),

md("""*[Three monitors.]*"""),

reveal("""1. **Weekly raw_grade SMD at Plant B.** Alarm if SMD of Plant B's raw_grade distribution from the deployment-baseline exceeds 0.25 (Cohen-d threshold). The reweighting was calibrated on a specific Plant-B raw_grade distribution; if it shifts, recompute.
2. **Monthly Plant-A re-estimation.** Re-fit $\\tau(z)$ on the most recent month of Plant A. Alarm if the slope coefficient changes sign (would invalidate the transport assumption).
3. **Quarterly mini-trial at Plant B.** Run a 100-200 unit randomised trial at Plant B; compare the Plant-B-direct ATE to the transported estimate. Alarm if the gap exceeds 0.05.

Rollback: revert to the pre-intervention baseline at Plant B via a single config update. Authority: Plant B operations lead with quality manager concurring."""),

md("""## Closing

Variants to extend:
- Add a **third plant C** with a more extreme effect-modifier shift; demonstrate when transport breaks.
- Replace the linear $\\tau(z)$ model with a **non-parametric CATE** (Causal Forest); compare transported vs direct.
- Apply the same machinery to **SECOM quarters** (Lab 13B's territory) as a real-data validation of the synthetic-data lessons.

The deployment-readiness focus of this starter — explicit rollback criteria, named monitors, mini-trial validation — is the part of causal AI that operates teams care about most. Methods chapters do the identification; this chapter and the capstone do the deployment. Refer to [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the per-artifact standards."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14h_multisite.ipynb", cells)
print("Built lab14h_multisite.ipynb")
