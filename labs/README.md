# Labs

Each chapter has **two labs**:

- **Lab A** (synthetic): build a hand-written SCM where the ground truth is known, run the chapter's estimator, verify it recovers the truth. Teaches *the method*.
- **Lab B** (real data): apply the same estimator to a community-standard industrial dataset where the truth is *not* known, and produce a defensible analysis report. Teaches *the deployment of the method under uncertainty*.

The two labs together cover the full arc from "does the math work?" to "what would I hand to a process engineer?"

## Open in Google Colab

Click any **Open In Colab** badge to launch the notebook.

| Chapter | Lab A (synthetic — method) | Lab B (real data — report) | Lab B dataset |
|---|---|---|---|
| 1 | [Lab 1](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab01.ipynb)   — high-AUC trap | [Lab 1B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab01b.ipynb)  — naive ML vs back-door on SECOM | SECOM |
| 2 | [Lab 2](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab02.ipynb)   — SCMs and back-door | Lab 2B (planned)  — multi-stage back-door on Bosch PLP | Bosch PLP |
| 3 | [Lab 3](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab03.ipynb)   — front-door and do-calculus | Lab 3B (planned)  — front-door on SECOM chemical pipeline | SECOM |
| 4 | [Lab 4](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab04.ipynb)   — IV, DID, RDD | Lab 4B (planned)  — phased station rollout on Bosch PLP | Bosch PLP |
| 5 | [Lab 5](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab05.ipynb)   — G-comp, IPW, AIPW, DML | Lab 5B (planned)  — DML on SECOM with diagnostics | SECOM |
| 6 | [Lab 6](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab06.ipynb)   — CATE and meta-learners | Lab 6B (planned)  — CATE per station on Bosch PLP | Bosch PLP |
| 7 | [Lab 7](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab07.ipynb)   — time-varying treatments | Lab 7B (planned)  — fast-charge policy effect on cycle life | LFP batteries (Severson et al. 2019) |
| 8 | [Lab 8](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab08.ipynb)   — dynamic regimes | Lab 8B (planned)  — replacement / migration DTR under degradation | Backblaze Drive Stats |
| 9 | [Lab 9](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab09.ipynb)   — causal discovery | Lab 9B (planned)  — PC/FCI on SECOM high-dim sensors | SECOM |
| 10 | [Lab 10](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab10.ipynb) — mediation, RCA, FDC | Lab 10B (planned)  — multi-stage FDC on Bosch PLP | Bosch PLP |
| 11 | [Lab 11](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab11.ipynb) — off-policy evaluation | Lab 11B (planned)  — OPE on Tennessee Eastman | Tennessee Eastman |
| 12 | [Lab 12](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab12.ipynb) — causal RL and digital twins | Lab 12B (planned)  — causal RL on Tennessee Eastman | Tennessee Eastman |
| 13 | [Lab 13](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/lab13.ipynb) — transportability, sensitivity, deployment | Lab 13B (planned)  — transportability across SECOM subsets | SECOM |

## Lab B dataset palette

Five datasets cover all 13 chapters; most appear across 2-6 Lab Bs so students develop fluency through repetition on shared data.

| Dataset | Type | Lab Bs | Source |
|---|---|---|---|
| **SECOM** | Real measurements | 1B, 3B, 5B, 9B, 13B | UCI ML Repo (semiconductor fab) |
| **Bosch PLP** | Real measurements | 2B, 4B, 6B, 10B | Kaggle competition (Bosch production line) |
| **LFP batteries** | Real measurements | 7B | Severson et al. 2019 / `data.matr.io/1/` (124 Li-ion cells, 72 fast-charge protocols, RUL = cycles-to-EOL) |
| **Backblaze Drive Stats** | Real measurements | 8B | backblaze.com/cloud-storage/resources/hard-drive-test-data (200k+ drives, daily SMART telemetry, observed failures) |
| **Tennessee Eastman** | Simulator (canonical digital twin) | 11B, 12B | Downs & Vogel (chemical-process control) |
| **MVTec AD** | Real images | 14B (frontier chapter) | MVTec research (industrial visual inspection) |

Each dataset is named honestly in the Lab B intro — students learn to distinguish "real measurements" from "industry-standard simulator output," which is itself an epistemic skill the course wants to develop.

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

Lab Bs trigger a one-time download of their dataset on first run (~3 MB for SECOM, larger for others). Downloaded data is cached under `labs/data/cache/` and gitignored.

## Layout

```
labs/
├── README.md                  # this file
├── requirements.txt           # union of all dependencies
├── lab01.ipynb ... lab13.ipynb       # Lab A (synthetic)
├── lab01b.ipynb ... lab13b.ipynb     # Lab B (real data) — being added incrementally
├── _nb.py                     # tiny helper that builds .ipynb from md/code cell lists
├── data/
│   ├── secom_prep.py          # load_secom(chapter=N) returns the pre-cleaned slice for that lab
│   ├── bosch_prep.py          # (planned)
│   ├── cmapss_prep.py         # (planned)
│   ├── te_prep.py             # (planned)
│   ├── mvtec_prep.py          # (planned)
│   └── cache/                 # downloaded data, gitignored
└── _build/
    ├── build_lab01.py ... build_lab13.py      # Lab A source-of-truth
    ├── build_lab01b.py ... build_lab13b.py    # Lab B source-of-truth (incremental)
    └── verify_all.py          # runs every Lab A's code cells (skipping %pip and Lab B for now)
```

To regenerate a notebook after editing its build script:

```bash
python3 labs/_build/build_lab01b.py
```

## What each lab does

**Lab A** (synthetic):
1. Build the SCM for the chapter's worked example.
2. Reproduce the chapter's numerical result so you can see the estimator behaves as advertised.
3. Stress the assumptions — break a model deliberately, watch the estimator degrade, recover with the right method.
4. Cross-check with the production library (DoWhy, EconML, CausalPy, causal-learn, etc.).

**Lab B** (real data):
1. Load a pre-cleaned slice of the chapter's anchor dataset (one line).
2. Run the naive baseline — what would standard ML point at?
3. State the assumed DAG and justify it from domain knowledge.
4. Apply the chapter's identification strategy + estimator on real data.
5. Quantify sensitivity. Write a three-bullet deployment recommendation.

Each Lab B's solution blocks contain a worked example and the interpretation — the *report quality*, not the answer key.
