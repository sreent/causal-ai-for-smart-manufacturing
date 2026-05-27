# Labs

Each chapter has **two labs**:

- **Lab A** (synthetic): build a hand-written SCM where the ground truth is known, run the chapter's estimator, verify it recovers the truth. Teaches *the method*.
- **Lab B** (real data): apply the same estimator to a community-standard industrial dataset where the truth is *not* known, and produce a defensible analysis report. Teaches *the deployment of the method under uncertainty*.

The two labs together cover the full arc from "does the math work?" to "what would I hand to a process engineer?"

## Open in Google Colab

Click any **Open In Colab** badge to launch the notebook.

| Chapter | Lab A (synthetic — method) | Lab B (real data — report) | Lab B dataset |
|---|---|---|---|
| 1 | [Lab 1](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch01/lab01.ipynb)   — high-AUC trap | [Lab 1B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch01/lab01b.ipynb)  — naive ML vs back-door on SECOM | SECOM |
| 2 | [Lab 2](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch02/lab02.ipynb)   — SCMs and back-door | [Lab 2B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch02/lab02b.ipynb)  — back-door on AI4I milling machine | AI4I 2020 |
| 3 | [Lab 3](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03.ipynb)   — front-door and do-calculus | [Lab 3B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch03/lab03b.ipynb)  — front-door on a stipulated SECOM chain | SECOM |
| 4 | [Lab 4](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch04/lab04.ipynb)   — IV, DID, RDD | [Lab 4B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch04/lab04b.ipynb)  — DID across Severson 2019 LFP batches | LFP batteries |
| 5 | [Lab 5](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch05/lab05.ipynb)   — G-comp, IPW, AIPW, DML | [Lab 5B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch05/lab05b.ipynb)  — DML on SECOM with four-estimator diagnostics | SECOM |
| 6 | [Lab 6](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06.ipynb)   — CATE and meta-learners | [Lab 6B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch06/lab06b.ipynb)  — CATE of the Severson feature, batch as effect modifier | LFP batteries |
| 7 | [Lab 7](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch07/lab07.ipynb)   — time-varying treatments | [Lab 7B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch07/lab07b.ipynb)  — time-varying capacity-drop exposure on LFP cycle life | LFP batteries |
| 8 | [Lab 8](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch08/lab08.ipynb)   — dynamic regimes | Lab 8B (planned)  — replacement / migration DTR under degradation | Backblaze Drive Stats |
| 9 | [Lab 9](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch09/lab09.ipynb)   — causal discovery | [Lab 9B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch09/lab09b.ipynb)  — PC on a 15-sensor SECOM slice | SECOM |
| 10 | [Lab 10](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch10/lab10.ipynb) — mediation, RCA, FDC | Lab 10B (planned)  — multi-stage FDC on Bosch PLP | Bosch PLP |
| 11 | [Lab 11](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch11/lab11.ipynb) — off-policy evaluation | [Lab 11B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch11/lab11b.ipynb)  — IPS/SNIPS/DR on Tennessee Eastman | Tennessee Eastman |
| 12 | [Lab 12](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch12/lab12.ipynb) — causal RL and digital twins | [Lab 12B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch12/lab12b.ipynb)  — learned-twin validation on Tennessee Eastman | Tennessee Eastman |
| 13 | [Lab 13](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch13/lab13.ipynb) — transportability, sensitivity, deployment | [Lab 13B](https://colab.research.google.com/github/sreent/causal-ai-for-smart-manufacturing/blob/main/labs/ch13/lab13b.ipynb)  — transport SECOM Jul-Aug -> Sep-Oct | SECOM |

## Lab B dataset palette

Five datasets cover all 13 chapters; most appear across 2-6 Lab Bs so students develop fluency through repetition on shared data.

| Dataset | Type | Lab Bs | Source |
|---|---|---|---|
| **SECOM** | Real measurements | 1B, 3B, 5B, 9B, 13B | UCI ML Repo (semiconductor fab) |
| **AI4I 2020** | Synthetic-but-named-semantics | 2B | Matzka 2020 / UCI ML Repo 601 (milling machine, 10k samples, 5 failure modes, CC BY 4.0). Vendored in `labs/data/ai4i2020.csv` |
| **LFP batteries** | Real measurements | 4B, 6B, 7B | Severson et al. 2019 (124 LFP cells), pre-processed via Mattia 2021's MIT-licensed [revisit-severson-et-al](https://github.com/petermattia/revisit-severson-et-al) repo into ~560 KB of summary + per-cycle CSVs vendored in `labs/data/` |
| **Backblaze Drive Stats** | Real measurements | 8B, 10B (planned) | backblaze.com/cloud-storage/resources/hard-drive-test-data (200k+ drives, daily SMART telemetry, observed failures) |
| **Bosch PLP** *(dropped)* | Real measurements | — | Kaggle competition (Bosch production line); 14GB + Kaggle gate. Replaced by AI4I/LFP/Backblaze. |
| **Tennessee Eastman** | Simulator (canonical digital twin) | 11B, 12B | Downs & Vogel via [`tep2py`](https://github.com/camaramm/tep2py); two trajectories pre-generated in `labs/data/`, Fortran source vendored under `labs/data/te_simulator/` for re-simulation |
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
├── ch01/
│   ├── lab01.ipynb            # Lab A (synthetic)
│   └── lab01b.ipynb           # Lab B (real data)
├── ch02/
│   └── lab02.ipynb            # Lab 2B planned
├── ...
├── ch13/
│   └── lab13.ipynb            # Lab 13B planned
├── _nb.py                     # tiny helper that builds .ipynb from md/code cell lists
├── data/
│   ├── secom_prep.py          # load_secom(chapter=N) returns the pre-cleaned slice for that lab
│   ├── secom.zip              # vendored UCI SECOM data
│   ├── ai4i_prep.py           # load_ai4i(chapter=N) for Lab 2B (milling machine)
│   ├── ai4i2020.csv           # vendored AI4I 2020 (CC BY 4.0, 520 KB)
│   ├── te_prep.py             # load_te(scenario) for Labs 11B/12B
│   ├── te_logged.csv          # pre-generated TE trajectory under Bernoulli(0.5) IDV(1)
│   ├── te_candidate.csv       # pre-generated TE trajectory under IDV(1) = 0 always
│   ├── te_simulator/          # vendored tep2py (Fortran source + Python wrapper) for re-simulation
│   ├── lfp_prep.py            # load_lfp(chapter=N) for Labs 4B/6B/7B
│   ├── lfp_cell_summary.csv   # 124 cells × 7 cols (cycle_life, log_var_deltaQ, ...)
│   ├── lfp_cell_cycle.csv     # 12,276 rows × 6 cols (per-cycle capacity trajectories)
│   ├── backblaze_prep.py      # (planned) load_backblaze for Labs 8B/10B
│   ├── mvtec_prep.py          # (planned) Lab 14B vision
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
