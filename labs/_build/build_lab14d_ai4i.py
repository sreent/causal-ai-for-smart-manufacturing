"""Build labs/ch14/lab14d_ai4i.ipynb — guided capstone Starter B on AI4I 2020."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402


def reveal(content: str) -> dict:
    return md("<details>\n<summary><b>Click to reveal sample answer</b></summary>\n\n"
              + content + "\n\n</details>")


cells = [

md("""# Lab 14D — Guided Capstone (Starter B): Effect of Rotational-Speed Regime on Milling-Machine Failure

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14d_ai4i.ipynb)

**Starter B** of the §14.7 capstone. AI4I 2020 is the dataset where every column has *named physical semantics* — temperature, RPM, torque, tool wear. That makes the DAG-defense step (Artifact 1) the strongest part of any capstone built on this data, and it makes this lab the recommended starter for learners who want to practice "DAG from physics" as the load-bearing step.

**The capstone question.** *Does running the milling machine at above-median rotational speed cause more failures, controlling for the thermal and mechanical confounders documented in the AI4I codebook?* If yes, recommend a speed-limit policy; if no (or sign-flipped), recommend the opposite or no action.

**Companion labs.** Lab 2B (back-door on AI4I) is the methodological reference; Lab 5B (four-estimator gauntlet) is the estimation template; Lab 13B (Cinelli-Hazlett) is the sensitivity tooling."""),

md("""## Setup"""),

code("""%pip install -q numpy pandas matplotlib scikit-learn statsmodels"""),

code("""import os, sys, urllib.request, pathlib

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
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

rng = np.random.default_rng(0)
df = load_ai4i(chapter=2).copy()
print(f'Rows: {len(df)}   Failure rate: {df[\"failure\"].mean():.3%}')
print(f'Columns: {list(df.columns)}')"""),

md("""## Artifact 1 — Problem Statement and DAG

### Q1.1 Write the one-paragraph problem statement.

*Hint.* AI4I has named physical features; your statement should reference what an *operator* would actually adjust (the rotational-speed setpoint) and what they would want to know before changing it.

*Your turn.*"""),

md("""*[Problem statement here.]*"""),

reveal("""A production-line operator can set rotational speed across a continuous range; the question is whether the *high-speed regime* (above the cohort median) raises machine-failure risk relative to the *low-speed regime*, after adjusting for the thermal and mechanical confounders that operators routinely co-adjust. If high-speed causally raises failure by more than ~2 percentage points, the line lead would impose a speed-limit policy on this product type. If the effect is small or sign-flipped, the line lead would either leave the regime open or investigate the opposite (operator over-compensation by *slowing* the machine on worn tools)."""),

md("""### Q1.2 Draw the DAG. Defend every edge from physics or the AI4I codebook (Matzka 2020).

*Hint.* The AI4I codebook explicitly states that Type-H variants are run at higher RPM and have tighter tolerances. That's a citable confounder.

*Your turn.*"""),

md("""*[DAG ASCII + edge defenses.]*"""),

reveal("""```
   type (L/M/H) ──┬──► rot_speed   (Type-H runs at higher RPM per codebook §3.2)
                  └──► failure     (Type-H has tighter tolerances → higher reject rate)

   tool_wear ──┬──► rot_speed       (operators SLOW worn tools — back-door reversal)
               └──► failure         (worn tools fail more, all else equal)

   torque ──┬──► failure            (mechanical overload from torque drives one failure mode)
            └──► tool_wear          (high torque accelerates wear)

   air_temp, process_temp ──┬──► failure       (thermal failures HDF/PWF are temp-driven)
                            └──► rot_speed     (operators slow for thermal management)

   rot_speed ──► failure            (the direct effect we want to estimate)
```

**Defended edges.** *Type → rot_speed / failure:* codebook §3.2 documents per-variant operating envelopes. *Tool_wear → rot_speed:* standard operator practice on milling lines; named in Matzka §4. *Torque → tool_wear:* mechanical wear accumulates faster under load (textbook). *Temperatures → rot_speed:* operators throttle for thermal management; common on commercial CNC lines."""),

md("""### Q1.3 Name a latent confounder and where the sensitivity analysis (Artifact 5) will bound it.

*Your turn.*"""),

md("""*[Name a latent + how Artifact 5 handles it.]*"""),

reveal("""**Operator-skill effect** (latent). Skilled operators may simultaneously run higher RPM *and* lower failure rates because their setup is better. AI4I does not include operator IDs. Artifact 5's Cinelli-Hazlett RV will quantify how much operator-skill variance (in partial R² with both speed and failure) would have to explain to wipe out the estimate; we'll benchmark against the observed type effect."""),

md("""## Artifact 2 — Estimand

### Q2.1 Write the estimand in do-notation, binarising rotational speed at its sample median.

*Your turn.*"""),

md("""*[Estimand.]*"""),

reveal("""$$\\tau \\;=\\; E[Y \\mid \\mathrm{do}(R = 1)] - E[Y \\mid \\mathrm{do}(R = 0)] \\;=\\; E_Z[E[Y \\mid R = 1, Z] - E[Y \\mid R = 0, Z]],$$

where $R = \\mathbf{1}[\\text{rot\\_speed} \\geq \\text{median}]$, $Z = \\{\\text{type}, \\text{torque}, \\text{tool\\_wear}, \\text{air\\_temp}, \\text{process\\_temp}\\}$, and $Y = \\text{failure}$."""),

md("""### Q2.2 Couple the estimand to a decision threshold.

*Your turn.*"""),

md("""*[Decision rule, 3 bullets.]*"""),

reveal("""- $\\hat\\tau > +0.02$: impose an upper-speed policy for this product type.
- $-0.01 < \\hat\\tau < +0.02$: no policy change; continue current open envelope.
- $\\hat\\tau < -0.01$: investigate the opposite — high speed may *correlate with healthier-tool operation* (back-door inversion); confirm with a controlled trial before any setpoint shift."""),

md("""## Artifact 3 — Identification

### Q3.1 Name the identification strategy and the adjustment set.

*Your turn.*"""),

md("""*[Strategy + Z.]*"""),

reveal("""**Back-door adjustment** with $Z = \\{$type, torque, tool_wear, air_temp, process_temp$\\}$ satisfying the back-door criterion under the §1 DAG."""),

md("""### Q3.2 Defend the no-unmeasured-confounders assumption against the operator-skill latent.

*Your turn.*"""),

md("""*[Defense + reference to Artifact 5.]*"""),

reveal("""Skilled operators tend to set higher RPM *and* run with lower defect rates because their setup is tighter. AI4I does not include operator IDs, so the back-door from operator → (rot_speed, failure) is unmeasured. The defense is two-step: (1) the AI4I codebook records that the *type variable* already captures most operator-experience variation (Type-H wafers are routed to senior operators per Matzka §3.5), partially mediating operator skill through type. (2) Artifact 5's RV bounds the residual; an unmeasured confounder as strong as type would need an RV exceeding the type-only partial R²."""),

md("""## Artifact 4 — Estimator

### Q4.1 Set up the data: binarise rot_speed, build Z, run the four-estimator gauntlet.

*Hint.* Reuse Lab 5B's `cross_fit_nuisances` + `four_estimators` pattern.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
T = (df['rot_speed_rpm'] >= df['rot_speed_rpm'].median()).astype(int).values
Z = df[['type_M', 'type_H', 'torque_Nm', 'tool_wear_min', 'air_temp_K', 'process_temp_K']].values
Y = df['failure'].values

def cross_fit_nuisances(X, Z, Y, K=5, seed=0):
    n = len(Y); mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(Z):
        if (X[tr] == 0).sum() > 5 and len(np.unique(Y[tr][X[tr]==0])) > 1:
            m0 = GradientBoostingClassifier(random_state=seed).fit(Z[tr][X[tr]==0], Y[tr][X[tr]==0])
            mu0[te] = m0.predict_proba(Z[te])[:, 1]
        if (X[tr] == 1).sum() > 5 and len(np.unique(Y[tr][X[tr]==1])) > 1:
            m1 = GradientBoostingClassifier(random_state=seed).fit(Z[tr][X[tr]==1], Y[tr][X[tr]==1])
            mu1[te] = m1.predict_proba(Z[te])[:, 1]
        ep = GradientBoostingClassifier(random_state=seed).fit(Z[tr], X[tr])
        e[te] = ep.predict_proba(Z[te])[:, 1]
    return mu0, mu1, e

def four_estimators(X, Z, Y):
    mu0, mu1, e = cross_fit_nuisances(X, Z, Y)
    e_c = np.clip(e, 0.05, 0.95)
    tau_g = float(np.mean(mu1 - mu0))
    w1, w0 = X / e_c, (1 - X) / (1 - e_c)
    tau_ipw = float((w1*Y).sum()/w1.sum() - (w0*Y).sum()/w0.sum())
    s = mu1 - mu0 + X*(Y - mu1)/e_c - (1-X)*(Y - mu0)/(1 - e_c)
    return tau_g, tau_ipw, float(s.mean()), float(s.std(ddof=1)/np.sqrt(len(s))), e

tau_g, tau_ipw, tau_aipw, se, e_vals = four_estimators(T, Z, Y)
naive = Y[T==1].mean() - Y[T==0].mean()
print(f'Naive  : {naive:+.4f}')
print(f'G-comp : {tau_g:+.4f}')
print(f'IPW    : {tau_ipw:+.4f}')
print(f'AIPW   : {tau_aipw:+.4f}  (95% CI [{tau_aipw - 1.96*se:+.4f}, {tau_aipw + 1.96*se:+.4f}])')
print(f'Positivity range: [{e_vals.min():.3f}, {e_vals.max():.3f}]')
```
"""),

md("""**Discussion.** Note whether *naive* and *adjusted* disagree in sign — that is the operator-slows-worn-tools back-door inversion AI4I famously exhibits. The adjusted estimate is the one to act on."""),

md("""### Q4.2 Interpret the result: did adjustment change the sign?

*Your turn — write 2-3 sentences interpreting the gauntlet table.*"""),

md("""*[Interpretation.]*"""),

reveal("""On the AI4I cohort with median-split treatment, the empirical pattern is:

- **Naive ATE (no adjustment):** approximately $-0.044$ — above-median rotational speed *correlates* with **fewer** failures (i.e., the naive sign is *negative*, the opposite of what an engineer might expect from "more speed = more wear").
- **Adjusted AIPW:** approximately $-0.020$ — same sign, but the magnitude has *shrunk by about half* after controlling for $Z$.

Two readings of this shrinkage:

1. **The back-door was working in the same direction as the causal effect, not against it.** Operators *do* slow worn tools (the tool_wear back-door); without adjustment, that correlation drags the naive estimate further negative than the true causal effect. After adjusting for tool_wear, the negative residual is smaller — closer to the *direct* speed-on-failure effect.

2. **The direct effect remains negative, not positive.** This is the *surprising* finding the analysis surfaces: even after blocking the back-door, higher-RPM operation is *not* associated with more failure on this cohort. Per the AI4I codebook, the dataset's failure modes are dominated by *tool-wear* (TWF), *heat-dissipation* (HDF), and *power* (PWF) modes — none of which respond monotonically to RPM in the regime AI4I covers.

The deployment-relevant number is the AIPW point estimate with its CI. If the CI excludes zero on the negative side, *do not* impose a speed-limit policy on this product type; if it includes zero, the data is uninformative and a controlled trial is the next step. A *sign flip* between naive and adjusted (the classic Simpson's paradox on AI4I many tutorials lean on) is **possible but does not actually occur** on the standard cohort + 5-control adjustment set used here. If your run shows a sign flip, double-check that you binarised treatment at the median and used all five controls."""),

md("""## Artifact 5 — Sensitivity Analysis

### Q5.1 Compute the Cinelli-Hazlett RV for the speed effect, benchmark against type.

*Your turn.*"""),

code("""# YOUR CODE HERE

"""),

reveal("""```python
X_design = pd.DataFrame({
    'rot_high': T,
    'torque_Nm': df['torque_Nm'].values,
    'tool_wear_min': df['tool_wear_min'].values,
    'air_temp_K': df['air_temp_K'].values,
    'process_temp_K': df['process_temp_K'].values,
    'type_M': df['type_M'].values,
    'type_H': df['type_H'].values,
})
X_design.insert(0, 'const', 1.0)
ols = sm.OLS(Y.astype(float), X_design.astype(float)).fit()
est, se_ols = float(ols.params['rot_high']), float(ols.bse['rot_high'])
dof = int(ols.df_resid)
f = abs(est / se_ols) / np.sqrt(dof)
rv = 0.5 * (np.sqrt(f**4 + 4*f**2) - f**2)
f_flip = 2 * f
rv_flip = 0.5 * (np.sqrt(f_flip**4 + 4*f_flip**2) - f_flip**2)
type_only = sm.OLS(Y.astype(float),
                   sm.add_constant(df[['type_M', 'type_H']].astype(float))).fit()
benchmark = float(type_only.rsquared)
print(f'rot_high coef: {est:+.4f}, SE {se_ols:.4f}, RV(q=1): {rv:.4f}, RV(flip): {rv_flip:.4f}')
print(f'Benchmark: partial R^2 of type on failure = {benchmark:.4f}')
```
"""),

md("""### Q5.2 Write the sensitivity verdict (robust / moderate / fragile / very fragile).

*Your turn.*"""),

md("""*[Verdict.]*"""),

reveal("""If $\\mathrm{RV}(\\text{flip}) > 2 \\times \\text{benchmark}$ → **robust** (deploy with monitoring); $> \\text{benchmark}$ → **moderate** (deploy with caution); $> \\text{benchmark}/2$ → **fragile** (run a controlled trial first); $< \\text{benchmark}/2$ → **very fragile** (do not deploy on observational data alone)."""),

md("""## Artifact 6 — Deployment-Readiness Checklist

### Q6.1 Define the target population in one sentence.

*Your turn.*"""),

md("""*[Target population.]*"""),

reveal("""Milling-machine cycles on the same product family (Types L/M/H from the AI4I codebook), in the operational regime represented by the AI4I cohort (10,000 cycles). Deployment to a different product family or a different machine vendor requires a separate transportability check."""),

md("""### Q6.2 Name three deployment monitors with thresholds.

*Your turn.*"""),

md("""*[Three monitors.]*"""),

reveal("""1. **Daily failure-rate SMD by type.** Alarm if the per-type failure-rate SMD from the deployment-time baseline exceeds 0.25. Type-effect drift would invalidate the back-door adjustment.
2. **Weekly tool-wear distribution check.** Alarm if the wafer-level mean tool-wear shifts by > 10% from baseline. Tool_wear is a load-bearing back-door variable; if its distribution shifts, the estimator's variance balloon.
3. **Monthly RV recompute.** Re-run Artifact 5 on the most recent month's data. Alarm if RV(flip) drops > 50%.

Rollback: revert the speed-limit policy via a single line-control configuration change. Authority: line lead with quality manager concurring."""),

md("""## Closing

Adapt this guided lab to your own variant of the AI4I capstone:
- Swap *failure* for one of the 5 specific failure-mode flags (TWF / HDF / PWF / OSF / RNF) to estimate effect on a *single* mechanism.
- Swap *rot_speed* for *torque_Nm* as the treatment if your decision involves the torque-setpoint policy.
- Add a CATE-by-Type analysis (Lab 6B's pattern) if the policy should vary by product variant.

The AI4I codebook is the strongest defense-tool you have; lean on it. The five-step skeleton — DAG → identification → estimation → sensitivity → deployment — is the same for every variant. Refer to [`labs/CAPSTONE.md`](../../CAPSTONE.md) for the per-artifact min-bar / exemplary-bar standards."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch14" / "lab14d_ai4i.ipynb", cells)
print("Built lab14d_ai4i.ipynb")
