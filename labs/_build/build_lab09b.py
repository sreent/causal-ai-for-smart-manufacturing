"""Build labs/ch09/lab09b.ipynb — causal discovery on a 15-sensor SECOM slice."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from _nb import md, code, write_notebook  # noqa: E402

cells = [

md("""# Lab 9B — Causal Discovery on Real SECOM Sensors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch09/lab09b.ipynb)

**Companion to Lab 9A.** Lab 9A ran the PC algorithm on a synthetic SCM and verified the recovered CPDAG matches the true Markov equivalence class. **Lab 9B asks the harder question on real data: what does PC return when there is no known truth, and how confident can we be in its output?**

The deliverable is a *discovered CPDAG with explicitly named caveats*: the CI test we chose, the significance level, the assumptions PC requires that SECOM almost certainly violates (causal sufficiency, i.i.d.), and a comparison against the *assumed* DAGs from Labs 1B and 5B.

**Dataset.** SECOM, restricted to 15 sensors (top |corr with yield|, same selection rule as Labs 1B and 5B) plus an ordinal `period` index and `yield_fail`. 17 variables is tractable for PC on a laptop; the full 590-sensor matrix is not."""),

md("""## What this lab is *not* doing

- **Discovering on all 590 sensors.** PC scales poorly with $|V|$; with 590 nodes the conditional-independence tests blow up combinatorially. The 15-sensor restriction is pedagogical, not principled.
- **Time-series-aware discovery.** SECOM is sampled sequentially in time; PCMCI / Granger-PC would be the correct family of algorithms. We use vanilla PC (i.i.d. assumption) so the lab focuses on the chapter's algorithm, and flag this as a known violation in Part 6.
- **Latent-confounder discovery (FCI).** Period is the *known* confounder. The lab includes period as an observed ordinal variable so PC has a chance to discover its role; running FCI under the assumption that period is latent would be a separate lab.
- **Score-based or continuous-optimization methods.** GES, NOTEARS, and DAGMA are introduced in §9.5–9.6; this lab focuses on the constraint-based reference (PC).
- **Hyperparameter tuning.** We use defaults from `causal-learn` (`alpha=0.05`, Fisher Z) and then vary alpha as a sensitivity check."""),

code("""%pip install -q ucimlrepo causal-learn"""),

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

rng = np.random.default_rng(0)"""),

md("""## Part 1 — Load the 15-sensor slice

The Ch 9 slice returns the 15 top-by-correlation sensors plus `period` and `yield_fail`. We convert `period` to an ordinal month index (0..3) so the Fisher-Z CI test treats it as continuous; this is an approximation we acknowledge in the decision section."""),

code("""df = load_secom(chapter=9)
sensor_cols = [c for c in df.columns if c.startswith("S")]

# Encode period as ordinal month index for fisherz.
df["period_ord"] = pd.Categorical(df["period"]).codes.astype(float)

variables = ["period_ord"] + sensor_cols + ["yield_fail"]
data = df[variables].astype(float).values

print(f"Shape:        {data.shape}")
print(f"Variables:    {variables}")
print(f"Periods:      {sorted(df['period'].unique())} -> ord {sorted(df['period_ord'].unique())}")"""),

md("""## Part 2 — PC with Fisher Z, alpha = 0.05

Fisher Z is the standard CI test for continuous Gaussian-ish data. The output is a CPDAG: directed edges where PC could orient them (from v-structures + Meek rules), undirected edges where it could not."""),

code("""from causallearn.search.ConstraintBased.PC import pc

cg = pc(data, alpha=0.05, indep_test="fisherz", show_progress=False)
G = cg.G  # GeneralGraph

def edges_from_cpdag(G, var_names):
    \"\"\"Return (directed_edges, undirected_edges) as lists of (parent_idx, child_idx) tuples.

    causal-learn's GeneralGraph encodes edges via a node-pair matrix; we read them
    via .get_graph_edges() for clarity.
    \"\"\"
    directed, undirected = [], []
    for e in G.get_graph_edges():
        # Endpoints are Endpoint enum values; -1=tail, 1=arrow in many encodings,
        # but causal-learn uses the same convention as Tetrad: TAIL and ARROW.
        from causallearn.graph.Endpoint import Endpoint
        n1, n2 = e.get_node1(), e.get_node2()
        i = G.get_node_map()[n1]
        j = G.get_node_map()[n2]
        ep1, ep2 = e.get_endpoint1(), e.get_endpoint2()
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            directed.append((var_names[i], var_names[j]))
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            directed.append((var_names[j], var_names[i]))
        elif ep1 == Endpoint.TAIL and ep2 == Endpoint.TAIL:
            undirected.append((var_names[i], var_names[j]))
        # ARROW-ARROW would be a bidirected edge (latent confounder); shouldn't appear in PC.
    return directed, undirected

directed, undirected = edges_from_cpdag(G, variables)
print(f"Directed edges (oriented by PC):    {len(directed)}")
print(f"Undirected edges (Markov class):    {len(undirected)}")
print(f"Total edges in CPDAG:               {len(directed) + len(undirected)}")"""),

md("""## Part 3 — What did PC say about `yield_fail`?

The key engineering question: *which variables are direct causes of yield failure, according to the discovered CPDAG?*"""),

code("""def neighbours_of(target, directed, undirected):
    parents  = [u for (u, v) in directed if v == target]
    children = [v for (u, v) in directed if u == target]
    undir_neigh = [w for (u, v) in undirected for w in (u, v) if w != target and target in (u, v)]
    return parents, children, undir_neigh

parents_y, children_y, undir_y = neighbours_of("yield_fail", directed, undirected)

print(f"Variables PC oriented as causes of yield_fail (direct parents):")
for p in parents_y:
    print(f"  - {p}  (direct cause)")
print()
print(f"Variables yield_fail orients an arrow into (direct effects):")
for c in children_y:
    print(f"  - {c}")
print()
print(f"Adjacent but un-oriented (Markov-class ambiguous):")
for u in undir_y:
    print(f"  - {u}")"""),

md("""**Read these lists with care.** A name appearing as a *parent of yield_fail* means PC, given the data and the CI test, could not distinguish this variable from a direct cause. It does *not* prove causation: PC's guarantees are asymptotic, the data violates several PC assumptions, and the CI tests have finite-sample error.

If `period_ord` appears as a parent of yield_fail or of many sensors, PC has *re-discovered* the confounder we identified by domain knowledge in Lab 1B. That is a non-trivial sanity check on the discovery."""),

md("""## Part 4 — Sensitivity to the significance level

PC's behaviour at finite samples depends strongly on alpha. Smaller alpha → stricter independence (fewer edges); larger alpha → more edges. The chapter recommends comparing CPDAGs at alpha = 0.01, 0.05, 0.10 as a default sensitivity check."""),

code("""def run_pc_summary(data, alpha, indep_test="fisherz"):
    cg_ = pc(data, alpha=alpha, indep_test=indep_test, show_progress=False)
    d, u = edges_from_cpdag(cg_.G, variables)
    p, c, un = neighbours_of("yield_fail", d, u)
    return {
        "alpha": alpha,
        "n_directed": len(d),
        "n_undirected": len(u),
        "y_parents": p,
        "y_children": c,
        "y_undirected_neighbors": un,
    }

results = [run_pc_summary(data, a) for a in [0.01, 0.05, 0.10]]
summary = pd.DataFrame([
    {"alpha":       r["alpha"],
     "directed":    r["n_directed"],
     "undirected":  r["n_undirected"],
     "parents_of_yield":  ", ".join(r["y_parents"])    or "—",
     "neighbours_of_yield": ", ".join(r["y_undirected_neighbors"]) or "—"}
    for r in results
])
print(summary.to_string(index=False))"""),

md("""**The right way to read the alpha sweep.**

- A variable that appears as a parent of `yield_fail` *at all three* alpha values is a robust candidate.
- A variable that appears *only at the loosest alpha* is suspect — PC is grasping at finite-sample noise.
- The total edge count grows with alpha; if even at alpha = 0.10 PC produces a thinly connected graph, the data is genuinely uninformative for these CI tests.

This is the same logic as a stability-selection wrapper around any regularization-tuned procedure."""),

md("""## Part 5 — Compare to the *assumed* DAGs from Labs 1B and 5B

The assumed DAG in Lab 1B was:

```
   period ──┬──► all sensors
            └──► yield_fail
   sensors ──► yield_fail
```

Three concrete predictions we can check against the discovered CPDAG:

1. **`period_ord` is adjacent to many sensors.** If true, PC re-discovers the time-driven confounder structure.
2. **`period_ord` is adjacent to `yield_fail`.** If true, period is on the assumed-DAG path to yield.
3. **Some sensors are adjacent to `yield_fail`.** If none are, either the sensors carry no marginal causal signal, or the CI test missed it."""),

code("""# Use alpha = 0.05 as the canonical reading.
canonical = next(r for r in results if r["alpha"] == 0.05)
period_in_adjacency = ("period_ord" in canonical["y_parents"]
                       + canonical["y_children"]
                       + canonical["y_undirected_neighbors"])

# Count how many sensors period_ord is adjacent to in the directed+undirected graph.
d05, u05 = edges_from_cpdag(pc(data, alpha=0.05, indep_test="fisherz", show_progress=False).G, variables)
adj_to_period = set()
for u, v in d05 + u05:
    if u == "period_ord":
        adj_to_period.add(v)
    if v == "period_ord":
        adj_to_period.add(u)

sensor_adj_to_period = [s for s in adj_to_period if s.startswith("S")]
sensor_adj_to_yield  = [n for n in canonical["y_parents"] + canonical["y_undirected_neighbors"]
                        if n.startswith("S")]

print(f"Prediction 1 — period adjacent to many sensors:")
print(f"  Discovered: period_ord adjacent to {len(sensor_adj_to_period)} of {len(sensor_cols)} sensors")
print(f"  ({sensor_adj_to_period})")
print()
print(f"Prediction 2 — period adjacent to yield_fail:")
print(f"  Discovered: {'YES' if period_in_adjacency else 'NO'}")
print()
print(f"Prediction 3 — some sensors adjacent to yield_fail:")
print(f"  Discovered: {len(sensor_adj_to_yield)} sensors")
print(f"  ({sensor_adj_to_yield})")"""),

md("""## Part 6 — Decision

Three bullets, the deliverable for a process engineer:

1. **The discovered CPDAG broadly agrees / disagrees with the Lab 1B assumed DAG** (read the Part 5 output to fill in which). If period reappears as a connector to many sensors *and* to yield, the back-door reasoning in Lab 1B has independent algorithmic support — not proof, but a second line of evidence.

2. **The sensors PC orients into `yield_fail` are the candidate causal drivers.** They overlap partially with the |corr|-ranked sensors that 1B and 5B treated as primary suspects, but PC's reasoning is structural (separating sets) rather than correlational. Disagreements between the two lists are the most interesting cases — sensors with high marginal correlation that PC ruled out, and sensors with low marginal correlation that PC kept.

3. **The known violations are non-trivial.** SECOM data is *not* i.i.d. (time-series), Fisher Z assumes Gaussianity (sensors are not), and we excluded 575 sensors from the analysis. Treat the discovered CPDAG as a *hypothesis-generating* tool for follow-on work — interventional experiments, FCI under suspected latency, time-series-aware variants — not as a final causal model."""),

md("""## Reflection

**Discovery on real data returns an equivalence class, not a DAG.** The CPDAG has *un*directed edges by construction; those are not bugs, they are PC honestly reporting which orientations the data cannot distinguish. Resolving them requires additional input — interventions, temporal order, domain priors — and that's the natural bridge to interventional design (which is the back-half of any causal-AI roadmap).

**Algorithm choice is itself an assumption.** PC under Fisher Z assumes Gaussian continuous data, causal sufficiency, and i.i.d. samples. SECOM violates all three to some degree. The right defensive move is to compare across CI tests, alpha levels, and algorithm families (PC vs FCI vs NOTEARS) and report only the edges that survive the comparison. A single PC run is a starting point, not a deliverable."""),

md("""## What's next

Lab 10B uses mediation analysis to attribute a *specific yield drop* to a specific upstream station — a different question than \"which sensors cause yield?\" but the same algorithmic family (path identification under an assumed DAG)."""),
]

write_notebook(pathlib.Path(__file__).parent.parent / "ch09" / "lab09b.ipynb", cells)
print("Built lab09b.ipynb")
