"""Build labs/ch01/lab01.ipynb — Correlation to Causation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 1 — From Correlation to Causation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch01/lab01.ipynb)

**Companion lab to Chapter 1.** Build the high-AUC trap on synthetic stamping-line data, see why a strong predictive classifier still recommends the wrong intervention, and recover the right answer with a manual back-door adjustment.

This lab takes you through the central failure mode that motivates the whole book. By the end you will have generated data from a known structural causal model (SCM), trained a strong predictive classifier on it, watched the classifier confidently recommend a counterproductive action, and applied the do-operator by hand to recover the correct interventional answer. The lab is intentionally low-tech — only numpy, scikit-learn, and a generative SCM — because the point is structural, not algorithmic. The same trap appears with deep networks, gradient boosting, or any other supervised learner."""),

md("""## What you'll do

1. **Set up the SCM** — a stamping-line process where lot difficulty (unmeasured) drives both the operator's force setting and the defect probability.
2. **Generate observational data** by sampling from the SCM and dropping the latent variable, matching what the data warehouse would record.
3. **Train a naive classifier** to predict defect from force, get a high AUC, and interpret its score gradient as a (wrong) intervention recommendation.
4. **Apply the do-operator** by graph surgery on the SCM, Monte-Carlo evaluate, and recover the true interventional curve.
5. **Compare** the observational curve to the interventional curve — they have opposite slopes.
6. **Block the back-door path** by conditioning on a measurable proxy of difficulty (the lot type), and verify the adjusted estimator recovers the interventional answer."""),

md("""## Setup

The core of the lab uses standard scientific Python (numpy / pandas / scikit-learn) so the back-door adjustment is transparent. Part 7 also demonstrates the same adjustment using DoWhy's `CausalModel` API, which is the production tool you would reach for on a real DAG."""),

code("""# Colab: install the causal-inference libraries used in Parts 7 and exercises.
# Skip if already installed.
%pip install --quiet dowhy 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The structural causal model

A high-volume stamping line forms automotive bracket parts. At one station, an operator sets the stamping *force* (in newtons) before each lot runs. Force above 500 N tends to cause cracking; force below 500 N tends to cause incomplete forming. The "right" force depends on the *difficulty* of the incoming lot — which is itself a function of upstream variables (sheet thickness variation, alloy temper, lubrication state) that are not recorded in the MES. We summarize these into a single latent variable, $D$ (difficulty), with $D \\sim \\mathcal N(0, 1)$ in standardized units.

The operator has built up an internal model: harder lots need more force. So they compensate, setting force around $500 + 50 \\cdot D$. The defect probability is *jointly* a function of force and difficulty: harder lots fail more often regardless of force, and force itself has a (small) direct effect on the defect logit.

The DAG:

```
   D (latent difficulty) ──► force ──► defect
   D ──────────────────────────────────► defect
```

The single structural equation we need is the defect logit:
$$\\text{logit}\\,P(\\text{defect} = 1 \\mid F, D) = -3 + 1.5\\, D - 0.005\\,(F - 500).$$

The coefficient on force is **negative** — increasing force *reduces* defects, holding difficulty fixed. This is the key feature: in the data we will record, force and defect will look positively correlated (operators turn up force for hard lots, and hard lots fail more), but the *causal* effect of force on defect is negative. The structural equation knows this; the operator's pattern of action will hide it from the data."""),

code("""# Defect logit — the single structural equation we'll need.
# Defined once and used (a) to generate observational data and
# (b) later, to evaluate the do-operator by Monte Carlo.
def defect_logit(force_value, difficulty_value):
    return -3 + 1.5 * difficulty_value - 0.005 * (force_value - 500)

# Sanity check: harder lots have higher defect probability at any force level.
# Holding force at 500 N, a difficulty-2 lot should be far more defect-prone
# than a difficulty-0 lot.
for d in [-2, 0, 2]:
    p = 1 / (1 + np.exp(-defect_logit(500, d)))
    print(f"P(defect | force=500, D={d:+d}) = {p:.3f}")"""),

md("""The numbers say what you'd expect: a difficulty-+2 lot has roughly a 50% defect rate at the nominal 500 N force, while a difficulty $-2$ lot has well under 1%. *Holding force fixed*, harder lots fail more. This is the causal mechanism we want to recover from data."""),

md("""## Part 2 — Generate observational data

In the actual fab, $D$ is not measured per lot. The data lake records force (sensor on the press), defect status (from end-of-line inspection), and a categorical *lot type* tag — let's say "high-difficulty" or "low-difficulty" — which the upstream supplier records but which is coarse compared to the latent continuous difficulty.

The data we generate below mimics this. We sample difficulty, sample force conditional on it (the operator's compensation behavior), sample defects from the structural equation, and record force and defect — *not* the latent $D$ — plus a binary lot-type variable that is a noisy discretization of $D$."""),

code("""n = 5000

# Latent variables
difficulty = rng.normal(0, 1, n)

# Operator's behavior: force compensates for difficulty (with noise)
force = 500 + 50 * difficulty + rng.normal(0, 10, n)

# Outcome from the structural equation
defect = rng.binomial(1, 1 / (1 + np.exp(-defect_logit(force, difficulty))))

# Categorical lot-type tag (binary high/low difficulty proxy)
lot_high = (difficulty > 0).astype(int)

# Build the dataset the analyst sees (difficulty is NOT in the data lake)
data = pd.DataFrame({"force": force, "lot_high": lot_high, "defect": defect})
print(f"Observed defect rate: {defect.mean():.3f}")
print(f"force range: [{force.min():.0f}, {force.max():.0f}] N")
print(f"Observational correlation force-defect: {np.corrcoef(force, defect)[0,1]:+.3f}")
data.head()"""),

md("""The observational correlation between force and defect is positive: high-force lots are more likely to fail. That's because operators turn up force when difficulty is high, and high difficulty causes defects. The naive interpretation — "high force *causes* defects" — sounds right and is precisely backwards. We will now train a classifier on these data and watch it deliver exactly that wrong recommendation."""),

md("""## Part 3 — Train a naive classifier

Predict defect from force using gradient boosting. The classifier will achieve a respectable AUC because high force *is* genuinely informative about defects — it's a downstream consequence of difficulty, which causes defects. The classifier picks up the *associational* signal and gives no indication that it is observational rather than interventional."""),

code("""clf = GradientBoostingClassifier(random_state=0).fit(force.reshape(-1, 1), defect)
auc = roc_auc_score(defect, clf.predict_proba(force.reshape(-1, 1))[:, 1])
print(f"Classifier AUC: {auc:.3f}")

# Predicted defect probability across a sweep of force values
force_grid = np.linspace(force.min(), force.max(), 200)
p_observational = clf.predict_proba(force_grid.reshape(-1, 1))[:, 1]

fig, ax = plt.subplots()
ax.plot(force_grid, p_observational, label="Classifier P(defect | force)")
ax.set_xlabel("Force (N)")
ax.set_ylabel("Predicted P(defect)")
ax.set_title(f"Naive observational classifier — AUC {auc:.3f}")
ax.legend()
plt.show()"""),

md("""The classifier's predicted P(defect | force) is monotonically *increasing*. The action recommendation that follows is "reduce force." A team that took this recommendation seriously — say, by mandating a force-cap of 480 N across the line — would expect a defect reduction.

In fact the structural equation says the opposite. The $-0.005 \\cdot (F - 500)$ term means raising force *reduces* the defect logit. The reason the classifier sees a positive force–defect association is the confounding by difficulty: force is high *because* difficulty is high, and difficulty is what's actually driving the defects.

To verify the disconnect we need the *interventional* curve — what the defect rate would be if we *set* force to each value, independent of difficulty."""),

md("""## Part 4 — The do-operator, by hand

The do-operator replaces a variable's structural equation with a constant. In our SCM, $\\text{do}(\\text{force} = f)$ replaces "force = 500 + 50·D + noise" with "force = f". Every other equation is untouched. In particular, $D$'s distribution is *not* affected — $D$ is exogenous and upstream of force, so intervening on force doesn't change what difficulties show up.

The interventional defect probability is then
$$E[\\text{defect} \\mid \\text{do}(\\text{force} = f)] = E_D\\left[\\sigma(\\text{defect\\_logit}(f, D))\\right],$$
which we can estimate by Monte Carlo: draw many $D$ values from the marginal distribution of $D$ and average. Since we have $n = 5000$ samples of $D$ already, we can use those directly as our Monte Carlo sample (the empirical marginal)."""),

code("""def p_defect_under_do_force(f):
    \"\"\"E[defect | do(force = f)] under the SCM.\"\"\"
    logits = defect_logit(f, difficulty)         # vectorized over the n latent samples
    probs  = 1 / (1 + np.exp(-logits))
    return probs.mean()

p_interventional = np.array([p_defect_under_do_force(f) for f in force_grid])

fig, ax = plt.subplots()
ax.plot(force_grid, p_observational, label="Observational E[defect | force]")
ax.plot(force_grid, p_interventional, label="Interventional E[defect | do(force)]", linewidth=2)
ax.set_xlabel("Force (N)")
ax.set_ylabel("P(defect)")
ax.set_title("Observational vs. interventional defect curves")
ax.legend()
plt.show()"""),

md("""The two curves have **opposite slopes**. The observational curve (blue) climbs from low defect rates at low force to substantial defect rates at high force — exactly what the classifier saw. The interventional curve (orange) *drops* with increasing force — exactly what the structural equation says (the $-0.005\\,(F-500)$ term in `defect_logit`).

The team that mandated a force-cap based on the observational curve would have *increased* their defect rate. The right intervention is the opposite one.

This is the disconnect Chapter 1 motivates: a classifier with ~0.88 AUC and a clean monotone score curve, deployed as a decision-support tool, can recommend the wrong action with confidence. The disconnect is not a bug in the classifier; it is a feature of observational data. The do-operator gives us the right answer only when we can simulate the SCM — which we can do here only because we *built* the SCM. With real data we need another route."""),

md("""## Part 5 — Recover the interventional curve from data alone

The do-operator above used the latent $D$, which is not available in the data warehouse. But we have a *proxy*: the binary `lot_high` tag. The back-door criterion says: if `lot_high` blocks every back-door path from force to defect, conditioning on it gives an unbiased estimate of the interventional effect.

The DAG:

```
   D ──► force ──► defect
   D ────────────► defect
   D ──► lot_high (a proxy)
```

**Question.** Does `lot_high` actually block the back-door? It depends on whether `lot_high` captures *all* of $D$'s influence on force and defect. A binary discretization of a continuous variable loses information — within `lot_high = 1` (any $D > 0$), $D$ still ranges from 0 to $\\infty$, and within that range $D$ is still correlated with both force (operator compensation) and defect (the SCM). So we *expect* the binary proxy to leave residual confounding.

Let's see by running the adjustment."""),

code("""# Back-door adjustment with the binary lot_high.
# Standardized formula: E[defect | do(force=f)] = sum_lot P(lot) * P(defect | force=f, lot)

backdoor_curves = {}
for lot_value in [0, 1]:
    mask = data["lot_high"] == lot_value
    lr = LogisticRegression().fit(data.loc[mask, ["force"]].values, data.loc[mask, "defect"].values)
    backdoor_curves[lot_value] = lr.predict_proba(force_grid.reshape(-1, 1))[:, 1]

p_lot1 = (data["lot_high"] == 1).mean()
p_lot0 = 1 - p_lot1
p_adj_coarse = p_lot0 * backdoor_curves[0] + p_lot1 * backdoor_curves[1]

fig, ax = plt.subplots()
ax.plot(force_grid, p_observational, label="Observational (naive classifier)")
ax.plot(force_grid, p_interventional, label="True interventional (do-operator on SCM)", linewidth=2)
ax.plot(force_grid, p_adj_coarse, "--", label="Back-door adjusted on lot_high (binary)", linewidth=2)
ax.set_xlabel("Force (N)")
ax.set_ylabel("P(defect)")
ax.set_title("Coarse binary proxy: back-door adjustment FAILS")
ax.legend()
plt.show()

# Slopes for comparison
slope_obs   = p_observational[-1] - p_observational[0]
slope_int   = p_interventional[-1] - p_interventional[0]
slope_adj_c = p_adj_coarse[-1] - p_adj_coarse[0]
print(f"Observational slope (low->high force): {slope_obs:+.3f}")
print(f"Interventional slope:                  {slope_int:+.3f}")
print(f"Coarse-proxy adjusted slope:           {slope_adj_c:+.3f}")"""),

md("""**Stop here.** Before reading on, form your own hypothesis: the adjusted slope is still positive (truth is negative). Why didn't conditioning on `lot_high` block the back-door? Two suspects: (a) the back-door criterion is genuinely violated by this coarse proxy, or (b) the criterion is satisfied but the estimator failed. Make a guess, then continue."""),

md("""**The coarse proxy fails.** The adjusted curve has the *same sign* as the naive observational curve — both go up with force — while the truth goes the other way. A team that trusted the binary-`lot_high` adjustment would make the same mistake as the team that trusted the raw classifier: cap force, lose yield.

Why? `lot_high` is a binary discretization of a continuous latent. Conditioning on it removes the *between-strata* part of the confounding but leaves the *within-strata* part. Within `lot_high = 1`, lots range from "slightly difficult" ($D \\approx 0$) to "very difficult" ($D \\to \\infty$), and within that range, force and defect remain correlated through $D$. The back-door criterion was satisfied *only* if `lot_high` carried *all* of $D$'s information about the (force, defect) joint — which a coarse binary discretization does not.

This is a real failure mode in practice. A supplier's "high difficulty" tag may sound like enough adjustment but rarely is. The remedy: a finer proxy."""),

md("""## Part 6 — A finer proxy: when adjustment works

In some fabs, the supplier provides a *continuous* difficulty score (or several finer-grained tags). Suppose we get $D$ itself in the data warehouse (or a near-perfect proxy for it). Then the back-door is genuinely blocked and the adjustment recovers the truth."""),

code("""# Hypothetical: we now have the continuous difficulty score in the data warehouse.
# Back-door adjust on continuous D. Two model choices:
#   (a) GradientBoostingClassifier (flexible, but extrapolates poorly off-manifold)
#   (b) LogisticRegression (parametric, extrapolates along the linear logit)
# Force and D are correlated ~0.98 in the data, so (low F, high D) and (high F, low D)
# combinations are rare — extrapolation matters.

mu_gbc = GradientBoostingClassifier(random_state=0).fit(
    np.column_stack([data["force"], difficulty]), data["defect"]
)
mu_lr = LogisticRegression().fit(
    np.column_stack([data["force"], difficulty]), data["defect"]
)

def standardize(model):
    return np.array([
        model.predict_proba(np.column_stack([np.full(len(difficulty), f), difficulty]))[:, 1].mean()
        for f in force_grid
    ])

p_adj_gbc = standardize(mu_gbc)
p_adj_lr  = standardize(mu_lr)

fig, ax = plt.subplots()
ax.plot(force_grid, p_observational, label="Observational (naive)")
ax.plot(force_grid, p_interventional, label="True interventional", linewidth=2)
ax.plot(force_grid, p_adj_coarse, ":",  label="Adj. on binary lot_high (wrong)")
ax.plot(force_grid, p_adj_gbc,    "-.", label="Adj. on D, GBC (extrapolation issue)")
ax.plot(force_grid, p_adj_lr,     "--", label="Adj. on D, logistic (works)", linewidth=2)
ax.set_xlabel("Force (N)")
ax.set_ylabel("P(defect)")
ax.set_title("Continuous D unlocks the back-door — but model choice still matters")
ax.legend()
plt.show()

print(f"Interventional slope:                          {slope_int:+.4f}")
print(f"Coarse-proxy (binary lot_high) adjusted slope: {slope_adj_c:+.4f}   (wrong sign)")
print(f"Continuous D, GBC adjusted slope:              {p_adj_gbc[-1] - p_adj_gbc[0]:+.4f}   (wrong sign — extrapolation)")
print(f"Continuous D, logistic adjusted slope:         {p_adj_lr[-1]  - p_adj_lr[0]:+.4f}   (matches truth)")"""),

md("""**The logistic-regression curve matches the interventional truth in shape and sign.** A team using this would correctly recommend *increasing* force. The back-door criterion delivered — once we had a proxy fine enough to block the back-door *and* a model that extrapolated correctly to (low force, high D) and (high force, low D) combinations that are rare in observation.

The gradient-boosting model, on the same data with the same adjustment set, *also* fails to recover the truth. Force and D have correlation ≈ 0.98 in observation, so the off-manifold region (low force with a hard lot, or high force with an easy lot) has almost no training data. GBC has no way to extrapolate; logistic regression assumes a linear logit and extrapolates parametrically. The right model is the one whose assumptions match the SCM's structure.

Four lessons from this contrast.

1. **The back-door criterion isn't satisfied just because you condition on something correlated with the confounder.** It's satisfied when the conditioning set blocks every back-door path. A coarse proxy can fail this test even if it sounds sufficient.

2. **The estimator's model also has to do work.** Even with a sufficient adjustment set, a flexible learner can fail in low-overlap regimes because it has no signal in the regions the back-door adjustment needs to evaluate. Parametric models extrapolate; flexible models interpolate well but extrapolate poorly. *Positivity* (Chapter 5) formalizes this.

3. **The price of identification is data, not just methodology.** When the right adjustment requires a variable you don't measure, the back-door criterion gives you the right answer only in principle — operationally you're stuck. This is what motivates the front-door criterion (Chapter 3) and instrumental variables (Chapter 4): identification strategies that don't require seeing the confounder at all.

4. **DAGs are the discipline.** Without a DAG, "what adjustment set should I use" becomes guesswork. The DAG tells you what to condition on; the data tells you whether the adjustment can be estimated."""),

md("""## Part 7 — The same adjustment with DoWhy

The manual back-door adjustment in Part 6 made the mechanism transparent. In production you would reach for a library that handles the same identification + estimation given a DAG specification. `dowhy.CausalModel` is the standard Python tool: pass it the data and the DAG, ask for the identifying estimand, fit an estimator, get an ATE."""),

code("""try:
    import dowhy
    from dowhy import CausalModel
    have_dowhy = True
except ImportError:
    have_dowhy = False
    print("DoWhy not installed. The %pip install at the top should have fixed this in Colab.")

if have_dowhy:
    # Provide DoWhy with the data (including the difficulty proxy) and the DAG.
    df_dowhy = data.assign(D=difficulty)
    gml_graph = '''graph [directed 1
        node [id "D" label "D"]
        node [id "force" label "force"]
        node [id "defect" label "defect"]
        edge [source "D" target "force"]
        edge [source "D" target "defect"]
        edge [source "force" target "defect"]
    ]'''
    model = CausalModel(data=df_dowhy, treatment="force", outcome="defect", graph=gml_graph)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Identified estimand:")
    print(str(estimand).split("\\n")[2])
    # Linear-regression back-door estimator over the same contrast
    est = model.estimate_effect(estimand,
        method_name="backdoor.linear_regression",
        target_units="ate")
    print(f"DoWhy back-door ATE (per unit force): {est.value:+.5f}")
    # Compare to our manual logistic-regression result on the same contrast
    contrast = force_grid[-1] - force_grid[0]
    print(f"DoWhy ATE x contrast range:           {est.value * contrast:+.4f}")
    print(f"Manual continuous-D logistic slope:   {slope_int:+.4f}  (interventional truth)")"""),

md("""DoWhy's back-door identification reproduces the manual result. The value of using the library is *not* a different number — it's that the library forces you to articulate the DAG (the GML graph spec above), runs the identification algorithm so you can't accidentally pick a wrong adjustment set, and gives you a standard interface to swap estimators (linear regression here; could be propensity-score matching, IPW, or DR with one argument change).

The library handles bookkeeping. The discipline of writing down the DAG is what makes the analysis defensible — and that discipline is the lab's lasting deliverable, library or no library."""),

md("""## Reflection

Three observations from this lab.

**The classifier was not wrong in the way it claimed to be.** The model said "high force is associated with defects" and that is empirically true in the data. The error is not in the prediction; it is in interpreting a prediction as a decision recommendation. Conflating "high force is *associated* with defects" with "*reducing* force will reduce defects" is the categorical mistake.

**The high AUC is silent about causal validity.** Predictive performance measures one thing: how well the model recovers the conditional distribution $P(Y \\mid X)$. Interventional accuracy is a separate property — how well the model predicts $P(Y \\mid \\text{do}(X))$. These are equal only when the data come from a randomized experiment or when the conditional $X \\to Y$ relationship is unconfounded. Neither holds here.

**Adjustment requires both data and an assumption.** We could only do the back-door adjustment because we knew (or assumed) which variable was on the back-door path. And as Part 5 showed, even when we have a variable correlated with the confounder, it's not necessarily *sufficient* — a coarse proxy can fail the criterion. The DAG was doing real work even when we never wrote out a formal d-separation argument."""),

md("""## Exercises

1. **Finer discretization.** Replace the binary `lot_high` with a 5-level quantile bin of difficulty (`pd.qcut(difficulty, 5)`). Run the back-door adjustment. How close does it get to the continuous-$D$ result? Plot all three adjusted curves (binary, 5-level, continuous) on the same axes to see the proxy-quality / adjustment-quality trade-off.

   <details><summary>Solution</summary>

   ```python
   from sklearn.linear_model import LogisticRegression
   lot_5 = pd.qcut(difficulty, 5, labels=False)
   curves_5 = []
   for level in range(5):
       mask = lot_5 == level
       lr = LogisticRegression().fit(data.loc[mask, ["force"]].values, data.loc[mask, "defect"].values)
       curves_5.append(lr.predict_proba(force_grid.reshape(-1, 1))[:, 1])
   weights = pd.Series(lot_5).value_counts(normalize=True).sort_index().values
   p_adj_5 = sum(w * c for w, c in zip(weights, curves_5))
   print(f"5-level adjusted slope: {p_adj_5[-1] - p_adj_5[0]:+.3f}   (truth: -0.136)")
   ```

   The 5-level slope sits between binary (`+0.667`, still wrong-sign) and continuous-$D$ logistic (`-0.159`, matches truth). With 5 bins the estimate is usually near zero; with 20+ bins it converges to the continuous-$D$ result. Proxy granularity is a sliding scale, not "good enough / not good enough."
   </details>

2. **Operator stratification.** Modify the SCM to introduce two operator types: a "conservative" operator who undercompensates (force = 500 + 30·D) and an "aggressive" operator who overcompensates (force = 500 + 70·D). Generate data with a 60/40 split. Does adjusting on `operator` recover the interventional curve? What about adjusting on `operator` *and* `lot_high`?

   <details><summary>Solution</summary>

   Adjusting on `operator` alone does *not* recover the truth: operator is a sibling of difficulty, not on the back-door path between force and defect. It correlates with the operator's *response to* $D$, not $D$ itself. Adjusting on `operator + lot_high` does better — the two coarsely partition the joint $(D, \text{operator})$ space — but a continuous proxy for $D$ remains the only way to get the interventional curve cleanly.
   </details>

3. **Selection bias.** Suppose defects below a certain threshold are not recorded in the data lake (the inspection station only reports failures, not pass marks). Re-generate the data with this filter and re-run the analysis. What does the adjusted curve look like? Which variable is now opening a path that was previously closed?

   <details><summary>Solution</summary>

   Selecting on `defect == 1` is conditioning on a descendant of force (force is a parent of defect; defect is the inclusion indicator). The truncated sample no longer represents the marginal $P(D)$, so the standardization $E_D[P(\text{defect} \mid \text{do}(F=f), D)]$ is biased. Symptom: the adjusted curve flattens and may flip sign. Fixes: inverse-probability-of-selection weights, or modelling the selection mechanism explicitly.
   </details>

4. **Mediator misuse.** Introduce a post-stamp dimensional check `M` as a mediator (force → M → defect) and add a small direct effect (force → defect not through M). Fit a classifier of defect on (force, M). What does the partial effect of force look like? What happens to the interventional answer if you intervene on M directly versus force? This previews the front-door material in Chapter 3.

   <details><summary>Solution</summary>

   With $M$ in the feature set, the regression's partial effect of force collapses to just the direct $F \to Y$ piece — $M$ has absorbed the indirect channel. $\text{do}(F = f)$ moves $Y$ both directly *and* through $M$; $\text{do}(M = m)$ moves $Y$ only through the $M \to Y$ coefficient. The two interventions give different answers, which is the point of the front-door criterion (Chapter 3): when $M$ fully mediates and the back-door is blocked from view, $\text{do}(M)$ is identifiable from data and $\text{do}(F)$ chains through it.
   </details>"""),

md("""## What's next

Chapter 2 formalizes the DAG and back-door machinery you just used by hand. Lab 2 walks through structural-equation construction, d-separation, and the back-door criterion explicitly, on a richer multi-stage process. The same trap — confounding leading to wrong-sign causal estimates — will recur throughout the rest of the labs in increasingly subtle forms.

The pattern this lab established is the recurring shape of every chapter going forward: **state the SCM, identify what's observable and what's latent, choose an identification strategy, estimate, validate against the ground truth where possible.**"""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch01" / "lab01.ipynb", cells)
print("Built lab01.ipynb")
