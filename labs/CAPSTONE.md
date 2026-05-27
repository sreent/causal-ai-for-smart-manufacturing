# Capstone Handbook

A self-contained handbook for the Chapter 14 §14.7 capstone project. This
document closes the gap from "specification" (what §14.7 lists) to "action"
(what a learner does at the desk on Monday morning). The companion worked
example is in [`labs/ch14/lab14c_capstone.ipynb`](ch14/lab14c_capstone.ipynb).

## 1. What the capstone is, and what it isn't

The capstone is **a single defensible causal analysis on one industrial
dataset**, applying the full pipeline from Chapters 1-13: DAG →
identification → estimation → sensitivity → deployment. It is not a survey,
not a literature review, not a feature-engineering exercise, and not a
predictive-modelling competition. The deliverable is **the report you would
hand to a process engineering team before they act on the recommendation**.

Three things a capstone *is*:
- A causal question stated precisely (in do-notation).
- A set of identifying assumptions named and defended.
- A numerical estimate plus the sensitivity statement that bounds its
  reliability.

Three things a capstone *isn't*:
- A predictive-model demo (a high AUC is not the deliverable).
- A multi-question survey (one question, well-defended, beats five
  questions superficially answered).
- An exhaustive DAG with 30 nodes and no edge defense (noise dressed up
  as rigour).

## 2. What success looks like

A capstone that would survive a meeting with a domain expert who disagrees
with your DAG. That is the bar.

Concretely: a reader who reads only your deliverable should be able to
- restate your causal question in their own words within 60 seconds,
- name the strongest identifying assumption and what would break it,
- describe how to monitor for that assumption breaking in production,
- decide whether to deploy your recommendation, defer pending more data,
  or escalate for a controlled experiment.

If a reader cannot do all four after reading your deliverable, the work
isn't done.

## 3. The six artifacts in detail

Each subsection below gives: *what it covers*, the *minimum bar* (the floor
a submission must clear), the *exemplary bar* (what an A-grade submission
looks like), and the *anti-patterns* (things that look defensible but
aren't). Use these as a checklist before submitting.

### Artifact 1 — Problem statement and DAG

**What it covers.** One paragraph stating the industrial decision the
analysis informs. A DAG over the relevant variables, with *each edge
defended* by either process knowledge or a citation. Latent variables
explicitly marked.

**Minimum bar.**
- One paragraph naming the manufacturing decision the analysis would
  inform.
- A DAG (drawn as ASCII art, mermaid, or a figure) with ≥ 4 variables.
- A two-sentence defense of every edge from physics, instrumentation,
  operator behaviour, or a published reference.

**Exemplary bar.**
- The decision the analysis informs is sharp enough that you can name
  the *team* in the organisation who would act on it.
- Every edge in the DAG cites a specific data source, paper, or piece
  of process documentation — not "the process engineers said so" but
  "the AI4I codebook (Matzka 2020, §3.2) states that Type-H variants
  run at higher RPM by design".
- Latent confounders are *enumerated*, not just acknowledged. Each
  latent variable has a sensitivity entry later (Artifact 5).
- The DAG includes the variables required for *every other artifact*
  to be defensible. If Artifact 3 invokes the back-door criterion,
  the back-door variables are in the DAG.

**Anti-patterns.**
- A "kitchen-sink DAG" with 20 nodes and no defense. Two well-defended
  nodes are worth more than twenty asserted ones.
- A DAG that drops latent confounders the literature mentions.
  Anonymised sensor data has confounders even if you don't name them;
  pretending otherwise is fragility.
- A DAG that visually contradicts the estimand in Artifact 2. If the
  estimand involves a mediator, the DAG must show it.

### Artifact 2 — Estimand

**What it covers.** The specific causal quantity (ATE, CATE, NDE/NIE,
policy value, controlled direct effect, transported ATE) the project
targets, written in do-notation. The connection from the problem
statement to the estimand should be unambiguous.

**Minimum bar.**
- The estimand on one line in do-notation, e.g.,
  `tau = E[Y | do(X = 1)] - E[Y | do(X = 0)]`.
- A one-sentence justification: why is *this* estimand the right summary
  of the decision in Artifact 1?

**Exemplary bar.**
- The estimand is *coupled* to a decision threshold. "If `tau > 0.05`,
  recommend a controlled trial on the intervention; if `0.0 < tau < 0.05`,
  recommend continuous monitoring; if `tau ≤ 0`, recommend status quo."
- Conditional and population estimands are distinguished. If you want
  CATE, say CATE; if you want population ATE, say ATE; do not blur
  the two.
- For sequential or multi-stage problems, the estimand is per-stage
  (Q-function, policy value) rather than a single number that hides
  the dynamic structure.

**Anti-patterns.**
- "We want to know if X causes Y" — no estimand. The course's running
  pitfall: a causal question is a *specific* counterfactual, not a
  hand-wave.
- An estimand that doesn't match the data structure (e.g., asking for
  a per-period CATE when only one period is in the dataset).
- An estimand stated only in plain English. The do-notation forces
  clarity; rely on it.

### Artifact 3 — Identification

**What it covers.** The identifying assumptions required for the
estimand (back-door, front-door, IV, mediation formula, transportability),
each defended in writing against the dataset's known limitations.

**Minimum bar.**
- The identification strategy named (back-door, front-door, IV, RD, DID,
  G-formula, IPTW, mediation formula, OPE, transportability).
- The set of variables involved in identification listed.
- A one-paragraph defense of each assumption.

**Exemplary bar.**
- For each assumption, the worst plausible violation is named explicitly
  ("if the supplier rotation introduces a confounder we didn't measure,
  the back-door adjustment is incomplete; the Cinelli-Hazlett sensitivity
  in Artifact 5 bounds the impact"). Each named violation links forward
  to Artifact 5.
- Alternative identification strategies considered and rejected with a
  reason. ("Front-door considered; rejected because we cannot defend the
  no-leak-into-M assumption on this data.")
- The identification claim is *testable* somewhere: either a placebo
  test, a falsification check, or a sub-sample re-estimation.

**Anti-patterns.**
- Asserting back-door without naming Z. The set Z is the entire content
  of the back-door identification.
- Invoking IV without checking the exclusion restriction or relevance.
  Each IV assumption is a separate defensibility step.
- "We assume no unmeasured confounders." This is the most common
  unfalsifiable assertion in applied causal inference; the entire point
  of Artifact 5 is to make this assertion *quantitative*.

### Artifact 4 — Estimator

**What it covers.** The choice of estimator (G-computation, IPW, AIPW,
DML, IV/2SLS, RDD, DID, doubly-robust mediation, OPE/IPS/SNIPS/DR, etc.)
with the reasoning for that choice over alternatives. A working
implementation that produces a numerical estimate.

**Minimum bar.**
- The estimator named.
- The implementation runs end-to-end and produces a numerical estimate.
- A 95% CI (or equivalent uncertainty) accompanies the point estimate.
- Standard diagnostics for the estimator (positivity histogram for IPW,
  pre-trend check for DID, weak-IV F-statistic for IV, etc.).

**Exemplary bar.**
- Multiple estimators run on the same data, with their agreement
  reported. Disagreement is diagnostic, not an embarrassment.
- The nuisance models (outcome model, propensity model, instrument-stage
  models) are fit with cross-fitting where the asymptotic theory
  requires it.
- Bootstrap or influence-function CIs reported alongside model-based CIs;
  when they diverge, the diagnosis is named.
- Code is reproducible: a `requirements.txt`, fixed random seeds,
  one-command rebuild.

**Anti-patterns.**
- A single estimator with no comparator. If your DML estimate looks
  good but you didn't try AIPW or G-comp, you don't know whether the
  DML number is signal or artefact.
- Hyperparameter-tuning the nuisance models on the outcome of interest.
  Cross-fit; do not let the test set leak into the nuisance fit.
- Reporting an estimate from an unstable estimator without flagging the
  instability. If ESS < 10% for IPS, say so; do not just report the
  IPS number.

### Artifact 5 — Sensitivity analysis

**What it covers.** At minimum, a Cinelli-Hazlett robustness value (or
equivalent) for the strongest identifying assumption from Artifact 3.
Where multiple assumptions matter (e.g., IV's relevance + exclusion +
monotonicity), a separate sensitivity for each.

**Minimum bar.**
- A robustness value (RV) for the no-unmeasured-confounders assumption.
- A benchmark comparing the hypothetical confounder to the strongest
  *measured* confounder (e.g., "an unmeasured confounder would need to
  be as strong as period to wipe out the estimate; period explains 2.6%
  of yield variance, so the bar is moderate but not high").

**Exemplary bar.**
- Sensitivity for *every* identifying assumption that could plausibly
  fail in production. IV's exclusion restriction, DID's parallel-trends,
  front-door's no-leak-into-M each get their own sensitivity entry.
- A *sign-flip threshold*: how strong would the violation have to be
  for the *direction* of the conclusion to reverse? An estimate whose
  *direction* is robust is much more deployable than one whose
  magnitude is barely positive.
- Sensitivity to estimator choice itself. Bootstrap re-runs across
  3-5 estimators; report the dispersion as a kind of model-class
  uncertainty.

**Anti-patterns.**
- "The estimate is robust." Not a claim; a vibe. Robust to what, by how
  much?
- Computing the RV but not benchmarking it. The number is meaningless
  in isolation; it needs a measured-confounder reference.
- Sensitivity sweeps that range from 0 to a number nobody could defend.
  γ = 4 in a logit sensitivity is "the unmeasured confounder explains
  >99% of the M-Y excess"; this is implausibly strong on most industrial
  data. Use a domain-defensible upper bound.

### Artifact 6 — Deployment-readiness checklist

**What it covers.** A short description of (a) the population the
estimate applies to, (b) the transportability scenario for the intended
deployment, (c) the distribution-shift and performance-drift monitors,
(d) rollback criteria.

**Minimum bar.**
- (a) One sentence describing the target population (e.g., "ST4000DM000
  drives in datacentre US-East, first 30 days of service").
- (b) One sentence on transportability (does the estimate generalise to
  other drive models, other datacentres, other time periods?).
- (c) Two-three sentences describing the monitors a production deployment
  would need.
- (d) One sentence on rollback (under what observed condition would the
  intervention be reversed?).

**Exemplary bar.**
- The target population is bounded by *all four* of: unit type, time
  window, geographic scope, and operational regime.
- Transportability is *quantified* — re-estimating on a held-out
  population subset and reporting the source→target gap, in the style
  of Lab 13B.
- The monitors are *named*, not generic. "Daily SMD on the top three
  sensors; per-week recompute of the propensity histogram; alarm on
  ESS < 25% or RV trajectory < 0.10."
- Rollback criteria are *executable* by an on-call rota — a process
  engineer reading the checklist should know exactly what to do.

**Anti-patterns.**
- "We will monitor for drift." How? On what cadence? Triggering what?
- A deployment-readiness section that's a generic MLOps boilerplate
  rather than the specific monitors this *causal* analysis needs. The
  RV is not in standard MLOps; the propensity histogram is not in
  standard MLOps; the back-door variable's variance shifts are not in
  standard MLOps.
- Rollback criteria that depend on rerunning the whole capstone. The
  rollback must be a fast diagnostic.

## 4. Dataset + question starters

§14.7 lists datasets but no causal questions. Below are six paired
starters across the course's palette — four on real industrial data
(SECOM, AI4I, LFP, Backblaze) and two on curated synthetic datasets
(OEE, multi-site) that expose analytic ground truth for validation.
Each is *one* defensible question; a learner can adopt it as-is, modify
it, or use it as a template for a different dataset/question.

### Starter A — SECOM: Causal effect of a top sensor on yield

- **Dataset.** SECOM (UCI 179), full 590-sensor frame.
- **Question.** Among the five sensors with the largest unadjusted
  correlation with yield_fail, which one has the *largest causal
  effect* on yield_fail after blocking the back-door through `period`?
- **Methods.** Same as Lab 1B for identification + Lab 5B for the
  four-estimator gauntlet + Lab 13B for sensitivity. The novelty over
  Lab 1B is reporting one *winner* across estimators rather than five
  candidates with shrinkage.
- **Why this starter works.** Five labs in the course already used
  SECOM, so the data is familiar; the causal question is sharp;
  identification uses one chapter's machinery and sensitivity uses
  another.

### Starter B — AI4I: Effect of rotational-speed regime on machine failure

- **Dataset.** AI4I 2020 (UCI 601), 10,000 milling-machine cycles.
- **Question.** What is the causal effect of running at *above the median
  rotational speed* on the binary failure outcome, controlling for the
  thermal and mechanical confounders documented in the AI4I codebook?
- **Methods.** Lab 2B's back-door logic + Lab 5B's four-estimator
  gauntlet + Lab 6B's CATE-by-Type heterogeneity check + Lab 13B
  sensitivity.
- **Why this starter works.** AI4I has *named physical features*, so
  every edge in the DAG can be defended from the codebook. This is
  the cleanest dataset for a capstone where the DAG-defense step is
  expected to be the strongest part of the submission.

### Starter C — LFP: Cumulative early-cycle high-drop exposure and battery cycle life

- **Dataset.** Severson 2019 LFP cells, per-cycle slice.
- **Question.** What is the causal effect of *cumulative early-cycle
  high-drop exposure* (cycles 2-50) on eventual `cycle_life`, accounting
  for the time-varying confounding of the previous cycle's capacity?
- **Methods.** Lab 7B's g-formula (one-shot + sequential ICE) + IPTW
  MSM + Lab 13B sensitivity for the no-unmeasured-confounders
  assumption.
- **Why this starter works.** Time-varying machinery on real data is
  rare; LFP gives it without requiring a synthetic SCM. The estimand
  is sharp and the chapter-7 machinery applies directly. Recommended
  for learners who want a *methodologically deep* capstone.

### Starter D — Backblaze: Optimal pre-emptive-replacement threshold under cost asymmetry

- **Dataset.** Backblaze Drive Stats, ST4000DM000 30-day subset.
- **Question.** What is the optimal SMART-based replacement policy at
  decision points day-7 and day-14, under a cost ratio of `c_R = 1`
  for pre-emptive replacement vs `c_F = 100` for in-service failure?
  (Re-run the analysis at `c_F = 10` and `c_F = 1000` and report the
  decision boundary's movement.)
- **Methods.** Lab 8B's DTR Q-learning + Lab 11B's OPE for evaluating
  the learned policy under a non-uniform behaviour policy + Lab 13B
  sensitivity.
- **Why this starter works.** A two-lab synthesis (DTR + OPE) that
  Lab 8B explicitly pointed forward to. The cost-ratio sensitivity is
  the kind of operational deliverable an industrial team would actually
  ask for. Recommended for learners who want a *production-flavoured*
  capstone.

### Starter E — OEE synthetic: multi-mediator root-cause decomposition

- **Dataset.** `labs/data/oee_synthetic.py` (curated synthetic; the SCM is
  in the module docstring).
- **Question.** A maintenance program was rolled out unevenly across four
  production lines. The mean OEE rose by ~4 percentage points in the
  treated shifts; *how much* of that rise is attributable to each of the
  three OEE drivers (Availability, Performance, Quality)?
- **Methods.** Lab 10B's NDE/NIE plug-in for *each* mediator separately,
  combined into a multi-mediator decomposition. Cinelli-Hazlett-style
  sensitivity on the no-unmeasured-confounders assumption (line is the
  observed confounder; supplier or shift-pattern would be unobserved
  analogs).
- **Why this starter works.** The capstone's *only* synthetic-data
  validation: the SCM exposes a `true_oee_decomposition()` helper that
  returns the analytic ground-truth NDE / NIE per driver, so the
  student can verify their estimators recover the truth (which is
  impossible on every other starter). Strongly recommended for
  learners who want a *methodologically tight* capstone where they
  know exactly what success looks like numerically.

### Starter F — Multi-site synthetic: transportability with effect modifier shift

- **Dataset.** `labs/data/multisite_synthetic.py` (curated synthetic; SCM
  in the module docstring).
- **Question.** Plant A's controlled trial estimates ATE ≈ +0.21 from the
  intervention. Plant B is about to deploy. Should they trust Plant A's
  estimate, or re-run the trial?
- **Methods.** Lab 13B's transportability framework: estimate ATE at A
  (DML or AIPW), apply naive to B, re-weight source A by target B's
  `raw_grade` distribution (the documented effect modifier), compare
  to the direct ATE on B. Show the source→target gap before and after
  reweighting.
- **Why this starter works.** Like Starter E, this dataset has a known
  analytic ground truth (`true_ate_per_site()` returns the true ATE_A,
  ATE_B, and what a correctly executed transport should recover).
  Plant A's ATE is +0.21; Plant B's is +0.39; the naive transport
  under-estimates by 45%. A learner whose reweighting recovers ATE_B
  has done transportability correctly. Recommended for learners who
  want a *deployment-flavoured* capstone with verifiable success.

### Optional Starter G — Your own dataset

Under instructor pre-approval, a learner may use a dataset they have
legal access to. The criteria for approval:

- The dataset has a clear industrial-decision context.
- The dataset is large enough for the chosen estimator's asymptotics
  to apply (≥ 500 units for cross-sectional ATEs; ≥ 50 cells / 1000
  cell-days for panel; ≥ 500 time-steps for sequential).
- The student can hand-write at least one DAG edge from process
  knowledge before the analysis begins.
- The data sharing is legally clean — no NDA violations, no
  re-identifiable personal data.

## 5. Deliverable format

A capstone submission has three pieces:

1. **A written report (8-15 pages).** PDF or Markdown. One page per
   artifact is the median; some artifacts (DAG, sensitivity) may need
   two pages, others (estimand, deployment checklist) less than one.
   The report is *the* deliverable; everything else supports it.

2. **A reproducible code repository.** Either a single Jupyter notebook
   that runs end-to-end (`capstone.ipynb`) or a directory of Python
   files with a top-level `make` or `python run.py` target. Include
   a `requirements.txt` (pinned) and fixed random seeds. The grader
   should be able to reproduce every number in the report by running
   one command.

3. **A 5-slide executive summary.** PDF. Audience: a director who has
   not read the report. Slides cover: (1) the question, (2) the
   identifying assumptions, (3) the estimate + CI, (4) the
   sensitivity bound, (5) the deployment recommendation. This forces
   you to compress the report into the parts that matter for a
   decision.

## 6. Time budget

Rough budget for a defensible mid-level submission, ~30-50 hours total:

| Artifact | Hours | Notes |
|---|---|---|
| 1. Problem + DAG | 4-6 | DAG-defense is the highest-leverage time investment. |
| 2. Estimand | 1-2 | If this takes more than 2 hours, the DAG is wrong. |
| 3. Identification | 3-4 | Includes the alternative-rejected paragraph. |
| 4. Estimator | 8-12 | Code, debugging, comparator runs, diagnostics. |
| 5. Sensitivity | 4-6 | Per-assumption sensitivity; benchmarks; sign-flip check. |
| 6. Deployment-readiness | 3-4 | Named monitors, rollback criteria, transportability check. |
| Report writing | 6-10 | Two-three passes; the third pass cuts 30% of the second. |
| Slide deck | 2-3 | After the report is final, not before. |

If you find yourself spending 20 hours on Artifact 4 and 0 hours on
Artifact 5, that is a signal: the capstone rewards *breadth-of-rigour*
across artifacts, not depth in any one. A 4-hour-per-artifact submission
across all six beats a 24-hour-on-estimator submission with the rest
missing.

## 7. Rubric expansion

§14.7 lists five evaluation criteria in priority order. The rough
weighting (treat as guidance, not policy):

1. **Correctness of identification — 30%.** The estimand is identified
   under the stated assumptions; the estimator is consistent for that
   estimand. *Fail this and the rest does not matter.*
2. **Defensibility of assumptions — 25%.** Each assumption has either
   a domain-grounded defense or a sensitivity analysis that survives
   a reasonable adversarial probe.
3. **Implementation quality — 20%.** Code runs end-to-end; numbers in
   the report match the code; reproducibility is one command.
4. **Deployment thinking — 15%.** The deployment checklist is
   concrete, not generic; the rollback criteria are executable.
5. **Manufacturing connection — 10%.** The analysis answers a question
   that an industrial team would ask. A statistically tidy paper that
   has nothing to do with manufacturing loses points here.

Within each criterion, the bar is "would survive a meeting with a
domain expert who disagrees." If your DAG would fall apart under
adversarial questioning, you have not met criterion 2 — no matter how
beautiful the rest is.

## 8. Common failure modes

The patterns that consistently cost points in mid-submissions:

- **The estimand drifts.** The introduction says ATE; the estimator
  produces CATE; the deployment recommendation is about a specific
  subgroup. Pick one and stick to it.
- **The DAG is decorative.** It appears in §1 of the report and is
  never referenced again. A capstone DAG must be cited in §3
  (identification), §5 (sensitivity), and §6 (deployment) at minimum.
- **Diagnostics are computed but not interpreted.** A propensity
  histogram is reproduced from Lab 5B without a comment on what it
  reveals about *this* dataset.
- **The sensitivity range is non-credible.** A γ sweep from 0 to 10
  on a logit scale is "the unmeasured confounder is the dominant
  cause of everything"; this is rarely a defensible upper bound on
  industrial data.
- **The deployment-readiness section is generic.** Generic monitors
  ("we will monitor for drift"); generic rollback ("we will revert
  if it underperforms"). Replace with named, executable specifics.
- **The slide deck contradicts the report.** Slide 3's headline
  number differs by 20% from the report's headline number. Always
  re-derive the slides from the report's final tables.
- **The README isn't a README.** The grader cannot rebuild because
  `pip install -r requirements.txt` fails. Run a fresh-environment
  rebuild before submitting.

## 9. Where the worked example helps

[`labs/ch14/lab14c_capstone.ipynb`](ch14/lab14c_capstone.ipynb) walks
through *one* full capstone on SECOM, demonstrating every artifact in
this handbook. Read it once to see what "defensible" looks like in
practice on the actual data, then close it and write your own capstone
on a *different* question (or, with instructor approval, the same
question on a different dataset).

The worked example deliberately uses the dataset most learners have
already seen (SECOM, used in Labs 1B/3B/5B/9B/13B/14B). The reason: by
the time a learner reaches the capstone, SECOM should feel like
home — the worked example shows them what a serious analysis on a
familiar dataset looks like. A capstone using SECOM is a perfectly
acceptable submission; one that picks a different dataset and matches
the worked example's depth is even better.

## 10. Closing

The capstone exists because the methods chapters did the hard
methodological work, but real industrial causal inference is not a
methods exercise — it is an *assumptions exercise*. The capstone tests
whether you can defend an analysis end-to-end: pick a question, choose
identifying assumptions, run the estimator, bound the sensitivity, name
the deployment conditions. None of those steps is hard in isolation;
doing all six well on the same question, on real data, is the skill
the course teaches.

Pick a problem you actually care about. Causal inference rewards depth
over breadth. Good luck.
