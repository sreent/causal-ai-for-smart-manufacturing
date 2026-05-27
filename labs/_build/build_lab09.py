"""Build labs/lab09.ipynb — Causal discovery on tabular data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook

cells = [
md("""# Lab 9 — Causal Discovery on Tabular Process Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab09.ipynb)

**Companion lab to Chapter 9.** Run constraint-based discovery (PC, FCI), score-based discovery (GES), and continuous-optimization discovery (NOTEARS) on the chapter's manufacturing-flow SCM. Diagnose how each handles latent confounders, varying sample sizes, and alpha-sensitivity. Reproduce the chapter §9.8 result where FCI's PAG marks the latent confounder while PC confabulates a direct edge."""),

md("""## What you'll do

1. **Build the chapter's manufacturing-flow SCM** with a latent variable $L$ that confounds $T$ and $P$.
2. **Run PC** with $L$ observed (should recover most of the CPDAG) and with $L$ latent (should produce a spurious $P - T$ edge).
3. **Run FCI** with $L$ latent. Verify the PAG correctly marks the latent-confounder ambiguity as $P \\;o\\!\\to T$.
4. **Sensitivity to alpha**: rerun PC at several alpha levels and watch the edge set change.
5. **Run GES** (score-based) and compare.
6. **Run NOTEARS** (continuous optimization) and observe the varsortability pitfall."""),

md("""## Setup"""),

code("""# Colab: install causal-learn (provides PC, FCI, GES, etc.). NOTEARS in Part 7
# uses our own didactic implementation — for production, install the official
# `notears` or `gcastle` package instead.
%pip install --quiet causal-learn 2>&1 | tail -2"""),

code("""import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (8, 5)"""),

md("""## Part 1 — The manufacturing-flow SCM (Chapter 9 §9.8)

Six variables: $A$ (upstream actor), $L$ (a latent process state), $T$ (a measurement upstream of D), $P$ (a parallel measurement), $D$ (a downstream metric), $Y$ (final outcome).

DAG:
```
   A ───► T ───► D ───► Y
   │      ▲      ▲      ▲
   ▼      │      │      │
   L ─────┘      P ─────┘
   │             ▲
   └─────────────┘ (L → P)
```

In the chapter's notation:
- $A \\sim \\mathcal N(0, 1)$
- $L \\sim \\mathcal N(0, 1)$ — latent in the FCI experiments
- $T = 0.7 A + 0.6 L + \\varepsilon_T$
- $P = 0.8 L + \\varepsilon_P$
- $D = 0.6 T + 0.5 P + \\varepsilon_D$
- $Y = 0.7 D + 0.4 A + \\varepsilon_Y$

V-structures: $A \\to T \\leftarrow L$, $T \\to D \\leftarrow P$, $A \\to Y \\leftarrow D$."""),

code("""def gen_data(n, rng):
    A = rng.normal(0, 1, n)
    L = rng.normal(0, 1, n)
    T = 0.7 * A + 0.6 * L + rng.normal(0, 0.3, n)
    P = 0.8 * L         + rng.normal(0, 0.3, n)
    D = 0.6 * T + 0.5 * P + rng.normal(0, 0.3, n)
    Y = 0.7 * D + 0.4 * A + rng.normal(0, 0.3, n)
    return pd.DataFrame({"A": A, "L": L, "T": T, "P": P, "D": D, "Y": Y})

n = 10_000
df_full = gen_data(n, rng)
df_obs  = df_full.drop(columns=["L"])
df_full.head()"""),

md("""## Part 2 — PC algorithm with $L$ observed

With all six variables visible, PC should recover most of the CPDAG. The true CPDAG has $A \\to T$, $L \\to T$, $T \\to D$, $P \\to D$, $A \\to Y$, $D \\to Y$ directed (forced by v-structures) and $L - P$ undirected (no v-structure to orient it)."""),

code("""try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    have_causallearn = True
except ImportError:
    have_causallearn = False
    print("causal-learn not installed. Install with: %pip install causal-learn")

if have_causallearn:
    cg_full = pc(df_full.values, alpha=0.01, indep_test="fisherz",
                 node_names=list(df_full.columns), show_progress=False)
    print("PC with L observed (edges):")
    for e in sorted({str(e) for e in cg_full.G.get_graph_edges()}):
        print(" ", e)"""),

md("""Output should show: $A \\to T$, $L \\to T$, $T \\to D$, $P \\to D$, $D \\to Y$, $L - P$, and (at this sample size) one missing v-structure: $A - Y$ stays undirected because PC's CI test for $A \\perp D \\mid \\lbrace T\\rbrace $ found a separating set that included $Y$, so the v-structure $A \\to Y \\leftarrow D$ was missed. This is the chapter's "finite-sample CI-test error" point — discussed in detail in §9.8."""),

md("""## Part 3 — PC with $L$ latent (causal-sufficiency violation)

Drop $L$ and rerun PC. Without $L$, the variables $T$ and $P$ are still dependent (shared parent), but PC has no way to encode "they have a hidden common cause." Forced by its causal-sufficiency assumption, it draws a *direct* edge between $T$ and $P$ — a spurious edge that doesn't exist in the true DAG."""),

code("""if have_causallearn:
    cg_obs = pc(df_obs.values, alpha=0.01, indep_test="fisherz",
                node_names=list(df_obs.columns), show_progress=False)
    print("PC with L latent (edges):")
    for e in sorted({str(e) for e in cg_obs.G.get_graph_edges()}):
        print(" ", e)"""),

md("""The spurious $P \\to T$ edge (or $T \\to P$, orientation is essentially arbitrary) appears. The *existence* of this edge is the problem; its direction is a side effect.

This is the failure mode causal-sufficiency violations cause. A downstream analyst using PC's output would assume $P$ is a cause of $T$ (or vice versa) and might recommend an intervention based on a non-existent edge."""),

md("""## Part 4 — FCI: the correct handling of latent confounders

FCI returns a Partial Ancestral Graph (PAG) that explicitly marks latent-confounder ambiguity. Where PC drew $P \\to T$, FCI draws $P \\;o\\!\\to T$ — circle at $P$, arrowhead at $T$. The interpretation: "the data say $T$ is not the cause of $P$, but cannot distinguish 'P causes T' from 'P and T share a latent common cause'."

The honest report. Tools downstream of FCI should not treat $P \\;o\\!\\to T$ as a directed edge."""),

code("""if have_causallearn:
    fci_g, _ = fci(df_obs.values, alpha=0.01, independence_test_method="fisherz",
                   node_names=list(df_obs.columns), show_progress=False)
    print("FCI with L latent (PAG edges):")
    for e in sorted({str(e) for e in fci_g.get_graph_edges()}):
        print(" ", e)"""),

md("""You should see $P \\;o\\!\\to T$ explicitly — FCI's circle-arrow notation for the latent-confounder ambiguity. Other edges in the PAG are correctly oriented (where v-structures forced them) or correctly marked as ambiguous (where the data is genuinely uninformative)."""),

md("""## Part 5 — Sensitivity to alpha

PC and FCI's edge decisions depend on the CI-test significance threshold $\\alpha$. Smaller alpha (stricter test) keeps more edges and is conservative; larger alpha removes more edges. Let's see the L-observed PC output at several alpha values."""),

code("""if have_causallearn:
    for alpha in [0.001, 0.01, 0.05, 0.1]:
        cg = pc(df_full.values, alpha=alpha, indep_test="fisherz",
                node_names=list(df_full.columns), show_progress=False)
        n_edges = len(cg.G.get_graph_edges())
        edges = sorted({str(e) for e in cg.G.get_graph_edges()})
        print(f"alpha = {alpha:.3f}:  {n_edges} edges")
        for e in edges:
            print(f"    {e}")
        print()"""),

md("""Sensitivity is significant — different alphas can change orientation and even edge presence. Reporting a single PC output without an alpha sweep gives a false sense of precision."""),

md("""## Part 6 — Sample-size effect

With fewer observations, CI tests are noisier and the PC output degrades. The chapter chose $n = 10{,}000$ for stable results. At $n = 500$ the output is much less reliable."""),

code("""if have_causallearn:
    for n_test in [500, 2000, 10_000]:
        df_test = gen_data(n_test, np.random.default_rng(0))
        cg = pc(df_test.values, alpha=0.01, indep_test="fisherz",
                node_names=list(df_test.columns), show_progress=False)
        print(f"n = {n_test:>5}:  {len(cg.G.get_graph_edges())} edges")
        for e in sorted({str(e) for e in cg.G.get_graph_edges()})[:8]:
            print(f"    {e}")
        print()"""),

md("""## Part 7 — NOTEARS (continuous-optimization), with caveats

NOTEARS recasts DAG learning as continuous optimization with a smooth acyclicity constraint. The full method uses an augmented-Lagrangian schedule that progressively tightens the acyclicity penalty until the recovered $W$ is exactly a DAG; we implement only a *didactic* version here that uses a single heavy penalty plus post-processing. For production use, install the `notears` package (or `gcastle`, which wraps several NOTEARS-family methods).

The chapter also warns about *varsortability*: NOTEARS-family methods can sometimes recover the topological order through variance ordering of variables rather than through causal structure. Standardizing the data is the standard defense."""),

code("""import scipy.optimize, scipy.linalg

def notears_didactic(X, lambda1=0.05, rho=10.0):
    \"\"\"Simplified linear NOTEARS. Production NOTEARS uses an augmented-Lagrangian
    schedule (Zheng et al. 2018) that we omit; this version uses a single fixed
    penalty and post-processes to enforce DAG structure.\"\"\"
    n, d = X.shape

    def loss_func(w):
        W = w.reshape(d, d)
        # Zero the diagonal (no self-loops)
        W = W - np.diag(np.diag(W))
        recon = 0.5 / n * ((X - X @ W) ** 2).sum()
        M = W * W
        h = np.trace(scipy.linalg.expm(M)) - d
        l1 = lambda1 * np.abs(W).sum()
        return recon + rho * h + l1

    w0 = np.zeros(d * d)
    result = scipy.optimize.minimize(loss_func, w0, method="L-BFGS-B",
                                      options={"maxiter": 200, "disp": False})
    W = result.x.reshape(d, d)
    W = W - np.diag(np.diag(W))   # enforce no self-loop
    return W

def post_process_to_dag(W, threshold=0.1):
    \"\"\"Zero out small entries and, for each bidirected pair (i,j),(j,i),
    keep only the larger-magnitude direction. Cuts cycles down to a DAG.\"\"\"
    W = np.where(np.abs(W) > threshold, W, 0)
    d = W.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            if W[i, j] != 0 and W[j, i] != 0:
                if abs(W[i, j]) >= abs(W[j, i]):
                    W[j, i] = 0
                else:
                    W[i, j] = 0
    return W

# Standardize first (the varsortability defense)
X_std = (df_full.values - df_full.values.mean(0)) / df_full.values.std(0)
W_raw = notears_didactic(X_std, lambda1=0.05, rho=10.0)
W_dag = post_process_to_dag(W_raw, threshold=0.1)

adj = pd.DataFrame(W_dag.round(2), columns=df_full.columns, index=df_full.columns)
print("Didactic NOTEARS adjacency (post-processed to a DAG):")
print("Rows = source, columns = target. Non-zero entry means edge.")
print(adj)
print()

# Compare to truth
truth_edges = {("A","T"), ("L","T"), ("L","P"), ("L","Y"), ("A","Y"),
               ("T","D"), ("P","D"), ("D","Y")}
found_edges = {(c1, c2) for i, c1 in enumerate(df_full.columns)
                       for j, c2 in enumerate(df_full.columns) if W_dag[i, j] != 0}
tp = found_edges & truth_edges
fp = found_edges - truth_edges
fn = truth_edges - found_edges
print(f"True positives ({len(tp)}):  {sorted(tp)}")
print(f"False positives ({len(fp)}): {sorted(fp)}")
print(f"False negatives ({len(fn)}): {sorted(fn)}")"""),

md("""The didactic NOTEARS recovers some of the true edges and misses or hallucinates others — typical for a single-pass implementation. The chapter §9.5 explains why: the loss surface is highly non-convex, the acyclicity constraint is brittle without the augmented-Lagrangian schedule, and the L1 penalty has to be tuned per dataset. *Use the published implementation, not your own toy version* unless you have a specific reason."""),

md("""## Part 8 — Diagnostic: varsortability

The standard varsortability test: compare the variable variance order to the true topological order. If they match, NOTEARS may be recovering the topological order through variance rather than through causal structure — and the "DAG" you get is partly an artifact of how the data were scaled."""),

code("""# Variance order vs true topological order
var_order = df_full.var().sort_values().index.tolist()
print(f"True topological order:        A, L, P, T, D, Y")
print(f"Variance order (ascending):   {' '.join(var_order)}")
match = (var_order == ['A', 'L', 'P', 'T', 'D', 'Y'])
print(f"Match? {match}")"""),

md("""Inspect the output above: in our SCM the variance order does **not** match the topological order, so varsortability is *not* the driver of any recovery NOTEARS achieves here. In datasets where the orders *do* match (which happens by accident in many real-world scalings), you should either randomize the variances or use a varsortability-robust method (e.g., `gcastle`'s `VarSortRegress` baseline)."""),

md("""## Part 9 — GES (score-based discovery)

GES searches over DAGs (or, equivalently, CPDAGs) to maximize a BIC-like score. It's the score-based counterpart to PC's constraint-based search."""),

code("""if have_causallearn:
    from causallearn.search.ScoreBased.GES import ges
    ges_result = ges(df_full.values, score_func="local_score_BIC", maxP=None,
                     parameters=None)
    print("GES with L observed (edges):")
    for e in sorted({str(e) for e in ges_result['G'].get_graph_edges()}):
        print(" ", e)"""),

md("""GES typically recovers a similar CPDAG to PC at the same sample size, but the *failure modes* differ: PC's edges depend on the CI test's α threshold, while GES's depend on the score function and its penalty term. When PC and GES agree, you have stronger evidence than either alone. When they disagree, the disagreement diagnoses which assumption is doing the work."""),

md("""## Reflection

**No discovery algorithm is bulletproof.** Each method has assumptions and failure modes: PC assumes causal sufficiency (no latents), FCI relaxes that but is slower and harder to interpret, GES requires a score-equivalent score, NOTEARS can exploit varsortability.

**FCI is the safer default when latents are plausible.** Manufacturing settings almost always have latents (unmeasured operator skill, ambient drift, untracked recipe parameters). Use FCI; document any PC output as "assuming causal sufficiency."

**Sensitivity analysis is mandatory.** Different $\\alpha$, different CI tests, different sample sizes give different graphs. Report a sensitivity range, not a point estimate of "the DAG."

**Discovery and identification are not the same.** Even a perfectly recovered DAG doesn't guarantee identification — some queries are non-identified for the true graph. Discovery is a *step* toward identification, not a substitute."""),

md("""## Exercises

1. **Try a different CI test.** Replace `fisherz` with a kernel-based test (`kci`) or chi-squared for discrete data. How does the recovered graph change?

   <details><summary>Solution</summary>

   On this linear-Gaussian SCM, Fisher-$z$ is the right test and other tests should agree at large $n$. Kernel CI test (`kci`) is more flexible but much slower; on continuous-Gaussian data it returns the same CPDAG. Chi-squared assumes discrete data and will reject more independencies on continuous data (because of discretization noise), producing a denser graph. Lesson: match the CI test to the data type.
   </details>

2. **GES.** Run causal-learn's `ges` on the same data. Compare to PC. Where do they agree, and where do they differ?

   <details><summary>Solution</summary>

   ```python
   from causallearn.search.ScoreBased.GES import ges
   res = ges(df_full.values, score_func="local_score_BIC")
   ```

   GES typically returns a similar CPDAG to PC at $n = 10{,}000$. The disagreement is most informative: where PC sees an unoriented edge and GES orients it (or vice versa), the orientation depends on the score-vs-CI-test trade-off. If PC and GES agree, the structure is robust; if they disagree, treat the disagreement as a sensitivity bound.
   </details>

3. **Add a chain of latent confounders.** $L_1 \\to L_2 \\to T, P$. Run FCI. How does the PAG report this richer structure?

   <details><summary>Solution</summary>

   FCI still marks the $T$–$P$ relationship with a circle-arrow endpoint (latent-confounder ambiguity). The PAG cannot distinguish "single latent $L$ causes both" from "chain $L_1 \\to L_2$ where $L_2$ causes both" — both are captured by the same bidirected/circle-arrow structure. To resolve the chain you'd need additional context (e.g., temporal precedence) or interventional data.
   </details>

4. **Real data.** Apply PC and FCI to the UCI SECOM dataset (high-dimensional semiconductor manufacturing data). With many sensors, what does the algorithm return? Is the output interpretable?

   <details><summary>Solution</summary>

   SECOM has 591 sensors per wafer × 1567 wafers. PC at $\\alpha = 0.01$ will return a very sparse graph (most pairs pass the conditional-independence test); FCI returns mostly bidirected edges (the algorithm sees common-cause structure everywhere because the conditional-independence patterns require it). At $\\alpha = 0.05$ both become densely connected. The output is *not* directly interpretable as a process DAG without a process-engineering pre-screen of which sensors are physically related; treat the algorithm output as a structure-discovery hint, not an authoritative DAG.
   </details>"""),

md("""## What's next

Lab 10 turns from estimating effects under a discovered DAG to *decomposing* those effects along the DAG's paths — mediation analysis applied to root-cause attribution in FDC."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "lab09.ipynb", cells)
print("Built lab09.ipynb")
