# Labs

Each chapter has **two labs**:

- **Lab A** (synthetic): build a hand-written SCM where the ground truth is known, run the chapter's estimator, verify it recovers the truth. Teaches *the method*.
- **Lab B** (real data): apply the same estimator to a community-standard industrial dataset where the truth is *not* known, and produce a defensible analysis report. Teaches *the deployment of the method under uncertainty*.

Chapter 14 is the capstone — `lab14b_secom_counterfactual.ipynb` covers §14.5's per-unit counterfactual attribution, and six guided notebooks (`lab14[c-h]_*_capstone.ipynb`) walk the §14.7 capstone artifacts on six different starter datasets.

## Open in Google Colab

Click any **Open In Colab** badge to launch the notebook.

| # | Lab A (synthetic — method) | Lab B (real data — report) | Dataset |
|---|---|---|---|
| 1  | [Lab 1](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch01/lab01_high_auc_trap.ipynb)   — high-AUC trap | [Lab 1B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch01/lab01b_secom_backdoor.ipynb)  — naive ML vs back-door | SECOM |
| 2  | [Lab 2](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch02/lab02_backdoor.ipynb)   — SCMs + back-door criterion | [Lab 2B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch02/lab02b_ai4i_backdoor.ipynb)  — back-door on milling-machine failure | AI4I 2020 |
| 3  | [Lab 3](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03_frontdoor.ipynb)   — front-door + do-calculus | [Lab 3B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03b_secom_frontdoor.ipynb)  — front-door on a stipulated SECOM chain | SECOM |
| 4  | [Lab 4](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch04/lab04_iv_did_rdd.ipynb)   — IV / DID / RDD | [Lab 4B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch04/lab04b_lfp_did.ipynb)  — DID across Severson 2019 LFP batches | LFP batteries |
| 5  | [Lab 5](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch05/lab05_dml.ipynb)   — G-comp / IPW / AIPW / DML | [Lab 5B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch05/lab05b_secom_dml.ipynb)  — four-estimator gauntlet | SECOM |
| 6  | [Lab 6](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06_cate.ipynb)   — CATE + meta-learners | [Lab 6B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06b_lfp_cate.ipynb)  — Severson-feature CATE by batch | LFP batteries |
| 7  | [Lab 7](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch07/lab07_time_varying.ipynb)   — time-varying treatments | [Lab 7B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch07/lab07b_lfp_time_varying.ipynb)  — sequential g-formula on cycle life | LFP batteries |
| 8  | [Lab 8](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch08/lab08_dtr.ipynb)   — dynamic treatment regimes | [Lab 8B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch08/lab08b_backblaze_dtr.ipynb)  — DTR Q-learning under cost asymmetry | Backblaze |
| 9  | [Lab 9](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch09/lab09_discovery.ipynb)   — causal discovery (PC / FCI) | [Lab 9B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch09/lab09b_secom_discovery.ipynb)  — PC on 15-sensor SECOM slice | SECOM |
| 10 | [Lab 10](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch10/lab10_mediation.ipynb) — mediation NDE / NIE | [Lab 10B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch10/lab10b_backblaze_mediation.ipynb)  — SMART_5 → SMART_197 → failure mediation | Backblaze |
| 11 | [Lab 11](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch11/lab11_ope.ipynb) — off-policy evaluation | [Lab 11B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch11/lab11b_te_ope.ipynb)  — IPS / SNIPS / DR + DR double-robustness demo | Tennessee Eastman |
| 12 | [Lab 12](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch12/lab12_twin.ipynb) — causal RL + digital twins | [Lab 12B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch12/lab12b_te_twin.ipynb)  — learned-twin validation | Tennessee Eastman |
| 13 | [Lab 13](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch13/lab13_sensitivity.ipynb) — sensitivity + transportability | [Lab 13B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch13/lab13b_secom_transportability.ipynb)  — transport across SECOM quarters | SECOM |
| 14 | *(no synthetic A lab — capstone chapter)* | [Lab 14B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14b_secom_counterfactual.ipynb)  — per-wafer counterfactual attribution | SECOM |

## Chapter 14 capstone — six guided notebooks

Each maps to one starter in [`CAPSTONE.md`](CAPSTONE.md). Pick one and walk it artifact-by-artifact.

| Starter | Guided notebook | Dataset | Methodological focus |
|---|---|---|---|
| A | [lab14c_secom_capstone](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14c_secom_capstone.ipynb) | SECOM | Back-door ATE on top yield-correlated sensor; four-estimator gauntlet; Cinelli-Hazlett sensitivity |
| B | [lab14d_ai4i_capstone](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14d_ai4i_capstone.ipynb) | AI4I 2020 | Back-door on milling-machine rotational speed; codebook-defended DAG |
| C | [lab14e_lfp_capstone](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14e_lfp_capstone.ipynb) | LFP batteries | Sequential ICE g-formula vs one-shot g-comp on cycle life |
| D | [lab14f_backblaze_capstone](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14f_backblaze_capstone.ipynb) | Backblaze | DTR Q-learning under $c_F/c_R \in \{10, 100, 1000\}$ cost-ratio sweep |
| E | [lab14g_oee_capstone](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14g_oee_capstone.ipynb) | OEE synthetic | Multi-mediator A × P × Q decomposition; verifiable against `true_oee_decomposition()` |
| F | [lab14h_multisite_capstone](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch14/lab14h_multisite_capstone.ipynb) | Multi-site synthetic | Reweighted transport across plants; verifiable against `true_ate_per_site()` |

## Lab B dataset palette

Five real datasets + two curated-synthetic generators cover all 14 chapters; most appear across 2-6 Lab Bs so students develop fluency through repetition on shared data.

| Dataset | Type | Used in | Source |
|---|---|---|---|
| **SECOM** | Real measurements | 1B, 3B, 5B, 9B, 13B, 14B, 14C | UCI ML Repo (semiconductor fab) |
| **AI4I 2020** | Real (CC BY 4.0) | 2B, 14D | Matzka 2020 / UCI ML Repo 601 (milling machine, 10k samples, 5 failure modes) |
| **LFP batteries** | Real measurements | 4B, 6B, 7B, 14E | Severson et al. 2019, pre-processed via Mattia 2021's MIT-licensed [revisit-severson-et-al](https://github.com/petermattia/revisit-severson-et-al) |
| **Backblaze Drive Stats** | Real measurements | 8B, 10B, 14F | backblaze.com (200k+ drives, daily SMART telemetry, observed failures) |
| **Tennessee Eastman** | Simulator (canonical digital twin) | 11B, 12B | Downs & Vogel 1993 via vendored `tep2py` |
| **OEE synthetic** | Curated synthetic | 14G | Generated by `labs/data/oee_synthetic.py` with documented SCM; analytic ground truth via `true_oee_decomposition()` |
| **Multi-site synthetic** | Curated synthetic | 14H | Generated by `labs/data/multisite_synthetic.py`; analytic ground truth via `true_ate_per_site()` |

Each dataset is named honestly in the Lab B intro — students learn to distinguish "real measurements" from "industry-standard simulator output" and "curated synthetic with documented SCM," which is itself an epistemic skill the course wants to develop.

## Lab B design principles

To stop real-data Labs B from becoming data-engineering exercises, every Lab B follows the same five-step skeleton:

1. **Load** (one line). Pre-cleaned data comes from a per-dataset prep module in `labs/data/` — no EDA, no missingness imputation, no feature selection in the notebook itself.
2. **The trap** (~10 lines). Apply the chapter's naive baseline; observe what it points at.
3. **The assumed DAG** (markdown only). State the DAG, justify each edge from domain knowledge.
4. **Identification + estimation** (~20 lines). Run the chapter's estimator under the DAG; compare to the naive baseline.
5. **Sensitivity + decision** (~10 lines + 3 bullets). Quantify robustness, write the deployment recommendation.

Each Lab B opens with a **"what this lab is NOT doing"** cell that names the tangents we deliberately skip. The grading rubric is *report quality*, not estimate accuracy (which can't be checked without an oracle).

## Run locally

```bash
git clone https://github.com/sreent/causal-ai-for-smart-manufacturing.git
cd causal-ai-for-smart-manufacturing
pip install -r labs/requirements.txt
jupyter notebook labs/
```

Lab Bs load their dataset from the vendored CSVs in `labs/data/`. No external downloads required for any of the 33 lab notebooks.

## Layout

```
labs/
├── README.md                                # this file
├── CAPSTONE.md                              # capstone handbook (§14.7 artifacts + starter prompts)
├── requirements.txt                         # union of all dependencies
├── ch01/
│   ├── lab01_high_auc_trap.ipynb            # Lab A
│   └── lab01b_secom_backdoor.ipynb          # Lab B
├── ch02/
│   ├── lab02_backdoor.ipynb
│   └── lab02b_ai4i_backdoor.ipynb
│   ...
├── ch13/
│   ├── lab13_sensitivity.ipynb
│   └── lab13b_secom_transportability.ipynb
├── ch14/                                    # capstone chapter
│   ├── lab14b_secom_counterfactual.ipynb    # §14.5 vision-QC counterfactual attribution
│   ├── lab14c_secom_capstone.ipynb          # guided capstone Starter A
│   ├── lab14d_ai4i_capstone.ipynb           # Starter B
│   ├── lab14e_lfp_capstone.ipynb            # Starter C
│   ├── lab14f_backblaze_capstone.ipynb      # Starter D
│   ├── lab14g_oee_capstone.ipynb            # Starter E
│   └── lab14h_multisite_capstone.ipynb      # Starter F
├── _nb.py                                   # tiny helper that builds .ipynb from md/code cell lists
├── data/
│   ├── README.md                            # dataset palette overview
│   ├── secom_prep.py, secom.zip
│   ├── ai4i_prep.py, ai4i2020.csv
│   ├── lfp_prep.py, lfp_cell_summary.csv, lfp_cell_cycle.csv, lfp_preprocess.py
│   ├── backblaze_prep.py, backblaze_subset.csv, backblaze_preprocess.py
│   ├── te_prep.py, te_logged.csv, te_candidate.csv, te_simulator/
│   ├── oee_synthetic.py                     # curated synthetic OEE log generator
│   ├── multisite_synthetic.py               # curated synthetic two-plant generator
│   └── cache/                               # local parse-cache, gitignored
└── _build/
    ├── build_lab01_high_auc_trap.py ... build_lab13_sensitivity.py        # Lab A sources
    ├── build_lab01b_secom_backdoor.py ... build_lab14b_secom_counterfactual.py  # Lab B sources
    └── build_lab14c_secom_capstone.py ... build_lab14h_multisite_capstone.py    # guided capstone sources
```

To regenerate a notebook after editing its build script:

```bash
python3 labs/_build/build_lab01b_secom_backdoor.py
```

## What each lab does

**Lab A** (synthetic, method-teaching):
1. Build the SCM for the chapter's worked example.
2. Reproduce the chapter's numerical result so you can see the estimator behaves as advertised.
3. Stress the assumptions — break a model deliberately, watch the estimator degrade, recover with the right method.
4. Cross-check with the production library (DoWhy, EconML, CausalPy, causal-learn, etc.).

**Lab B** (real data, report-teaching):
1. Load a pre-cleaned slice of the chapter's anchor dataset (one line).
2. Run the naive baseline — what would standard ML point at?
3. State the assumed DAG and justify it from domain knowledge.
4. Apply the chapter's identification strategy + estimator on real data.
5. Quantify sensitivity. Write a three-bullet deployment recommendation.

**Lab 14B + 14C-H** (capstone): a single applied analysis on a chosen dataset, walking the six §14.7 artifacts (problem+DAG → estimand → identification → estimator → sensitivity → deployment).

Each Lab B's solution blocks contain a worked example and the interpretation — the *report quality*, not the answer key.
