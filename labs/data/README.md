# Dataset palette for the Lab-B series

Each Lab B in this repo runs on a real (or simulator-generated) dataset rather
than a toy DAG, so chapter-specific causal-inference concepts land on
production-style mess. This document records, for every vendored dataset:

- where the raw data comes from and its license
- what (if any) preprocessing was applied to produce the vendored slice
- the schema the lab loaders expose
- which Lab B (or labs) use it

| Dataset             | Domain                       | Vendored as                                   | Preprocessing script             | Used by             |
|---------------------|------------------------------|-----------------------------------------------|----------------------------------|---------------------|
| SECOM               | Semiconductor wafer yield    | `secom.zip`                                   | none (cleaning inside loader)    | 1B, 3B, 5B, 9B, 13B, 14B |
| AI4I 2020           | Milling machine PdM          | `ai4i2020.csv`                                | none (UCI as-is)                 | 2B                  |
| LFP batteries       | Lithium-ion cycle life       | `lfp_cell_summary.csv`, `lfp_cell_cycle.csv`  | **`lfp_preprocess.py`**          | 4B, 6B, 7B          |
| Tennessee Eastman   | Chemical-process control     | `te_logged.csv`, `te_candidate.csv` + simulator | recipe below                   | 11B, 12B            |
| Backblaze           | Hard-drive failure (SMART)   | `backblaze_subset.csv`                        | **`backblaze_preprocess.py`**    | 8B, 10B             |
| OEE synthetic       | A x P x Q manufacturing KPI  | none (generator)                              | **`oee_synthetic.py`**           | capstone Starter E  |
| Multi-site synthetic | Cross-plant transportability | none (generator)                              | **`multisite_synthetic.py`**     | capstone Starter F  |

The two **curated synthetic** generators at the bottom were planned in
`book/COURSE_PLAN.md` §2 to expand the capstone palette where no public
real dataset exposes the structure (a manipulable A x P x Q OEE
decomposition; clean cross-site transportability with documented effect
modifiers). Each ships an analytic ground truth in a `true_*()` helper
so a capstone submission can validate its NDE/NIE or transported ATE
against the SCM the data was generated from.

Loaders (`*_prep.py`, `*_synthetic.py`) all expose a single public function
(e.g. `load_secom`, `load_lfp`, `load_te`, `load_ai4i`, `load_backblaze`,
`load_oee`, `load_multisite`) returning a chapter- or capstone-shaped
DataFrame. Open the loader for the chapter-by-chapter slice definitions.

---

## SECOM

- **Source.** McCann & Johnston, UCI ML Repository (2008).
  https://archive.ics.uci.edu/dataset/179
- **License.** UCI / CC BY 4.0.
- **Vendored.** `secom.zip` — the original UCI distribution (`secom.data` +
  `secom_labels.data`, 1567 wafers × 590 sensor features + binary yield label +
  timestamps spanning Jul-Oct 2008).
- **Preprocessing.** None offline. `secom_prep.py::load_secom(chapter=...)`
  reads the ZIP on first call, applies chapter-specific cleaning
  (missingness thresholds, low-variance filtering, period derivation from
  timestamps), and caches the parsed frame as a pickle under `cache/`.
- **Used by.** Labs 1B (associational), 3B (IV), 5B (causal discovery),
  9B (counterfactual diagnostics), 13B (deep front-door).

---

## AI4I 2020 Predictive Maintenance

- **Source.** Matzka (2020), "Explainable AI for Predictive Maintenance
  Applications", AI4I Conference. UCI ML Repository dataset 601.
- **License.** CC BY 4.0.
- **Vendored.** `ai4i2020.csv` (~520 KB) — committed as published.
  10 000 synthetic-but-physically-grounded rows from a milling-machine setup.
- **Preprocessing.** None offline. `ai4i_prep.py::load_ai4i(chapter=...)` just
  renames columns to ergonomic snake_case (`air_temp_K`, `torque_Nm`, etc.) and
  parses dtypes.
- **Used by.** Lab 2B (matching / propensity).

---

## LFP lithium-ion batteries — Severson 2019

- **Source.** Severson, K. A. et al. (2019), "Data-driven prediction of battery
  cycle life before capacity degradation", *Nature Energy* **4**, 383-391.
  Re-published with cleaner Python tooling by Peter Mattia (2021):
  https://github.com/petermattia/revisit-severson-et-al (MIT).
  Original raw data: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
- **License.** Severson data are public-domain (Toyota Research Institute);
  Mattia tooling is MIT.
- **Upstream size.** ~1.5 GB across three batch pickles (`batch1.pkl`,
  `batch2.pkl`, `batch3.pkl`). Not vendored.
- **Vendored.**
  - `lfp_cell_summary.csv` — 124 cells × 7 columns
  - `lfp_cell_cycle.csv` — 12 276 rows (124 cells × 99 cycles) × 6 columns
- **Preprocessing script.** [`lfp_preprocess.py`](./lfp_preprocess.py)

### Pipeline (Severson → vendored CSVs)

1. **Load.** Read the three Mattia batch pickles. Each is a dict keyed by
   cell-id (`b1c0`, …, `b3c44`); each cell carries `cycle_life` plus per-cycle
   capacity/voltage curves under `cycles[<n>]`.
2. **Rename batches.**  `b1 → train`, `b2 → test1`, `b3 → test2` (Severson's
   train/test split, kept as the cohort variable for Lab 4B's DID and Lab 6B's
   CATE-by-batch analyses).
3. **Per-cell summary features.** For each cell:
   - `Q10`, `Q100`: 1000-point `Qdlin` curves (discharge capacity vs
     linearly-spaced voltage) at cycles 10 and 100
   - `log_var_deltaQ = log10(var(Q100 − Q10))` — the dominant Severson feature
     (MAPE < 10 % on `cycle_life` by itself)
   - `max_cap_cyc10`, `max_cap_cyc100`: max of the raw `Qd` discharge samples
   - `fade_cyc10_to_100 = max_cap_cyc100 − max_cap_cyc10` (small negative)
4. **Per-cycle trajectory.** For each (cell, cycle) with `cycle ∈ {2, …, 100}`:
   - `max_cap`, `mean_cap` over that cycle's raw `Qd` samples
   - `cycle_life` repeated for convenience in time-varying labs
5. **Write CSVs.** Sorted, with `cell_id` of the form `"{batch}_cell{idx}"`
   (1-indexed within batch).

### Schema cheat-sheet

```
lfp_cell_summary.csv  (124 rows)
  cell_id              "train_cell1", …, "test2_cell40"
  batch                {train, test1, test2}
  cycle_life           cycles to 80 % capacity (148 → 2237)
  log_var_deltaQ       Severson canonical feature
  max_cap_cyc10        peak discharge capacity at cycle 10
  max_cap_cyc100       peak discharge capacity at cycle 100
  fade_cyc10_to_100    max_cap_cyc100 − max_cap_cyc10

lfp_cell_cycle.csv    (12 276 rows)
  cell_id, batch, cycle, max_cap, mean_cap, cycle_life
```

### Used by

- **Lab 4B (DID).** Batch as cohort, `log_var_deltaQ` as covariate, `cycle_life`
  as outcome. Treats Severson's three batches as natural-experiment cohorts.
- **Lab 6B (CATE).** Heterogeneity of the `log_var_deltaQ → cycle_life` slope
  across batches.
- **Lab 7B (time-varying treatments).** 99-cycle trajectory of `max_cap` per
  cell, with `cycle_life` as terminal outcome.

---

## Tennessee Eastman — Downs & Vogel 1993

- **Source.** Downs & Vogel (1993), "A plant-wide industrial process control
  problem", *Computers & Chemical Engineering* **17(3)**, 245-255. Fortran
  reference simulator wrapped in Python by
  [`tep2py`](https://github.com/camaramm/tep2py).
- **License.** See `te_simulator/LICENSE`. The TE problem itself is public
  reference material; `tep2py` is MIT.
- **Vendored.**
  - `te_simulator/` — Fortran source (`teprob.f`, `temain_mod.f`) +
    `tep2py.py` Python wrapper (patched for NumPy 2.x:
    `np.integer → np.int64`, `np.int(...) → int(...)`)
  - `te_logged.csv` — 500-step trajectory under a Bernoulli(0.5) IDV(1)
    behaviour policy (used by Lab 11B as the offline log)
  - `te_candidate.csv` — 500-step trajectory under IDV(1) = 0 always
    (used by Lab 11B as the ground-truth candidate-policy return)
- **Preprocessing / generation recipe.** Both CSVs are generated by running the
  vendored Fortran simulator. `te_prep.py::simulate_te(idata)` is the canonical
  entry point. To regenerate from scratch:

  ```bash
  # Build the Fortran extension (Colab: apt-get install gfortran first)
  cd labs/data/te_simulator
  python -m numpy.f2py -c temain_mod.pyf temain_mod.f teprob.f -m temain_mod
  ```

  ```python
  # Then in Python:
  import numpy as np
  from te_prep import simulate_te

  N_STEPS = 500
  rng = np.random.default_rng(0)

  # Logged: IDV(1) ~ Bernoulli(0.5), all other IDVs off
  idata_logged = np.zeros((N_STEPS, 20), dtype=int)
  idata_logged[:, 0] = rng.integers(0, 2, size=N_STEPS)
  simulate_te(idata_logged).to_csv("te_logged.csv", index=False)

  # Candidate: IDV(1) = 0 always
  idata_candidate = np.zeros((N_STEPS, 20), dtype=int)
  simulate_te(idata_candidate).to_csv("te_candidate.csv", index=False)
  ```

  The CSVs in the repo also prepend a 3-min `ts_min` timestamp column and an
  `action_idv1` column (a copy of `idata[:, 0]`) for the labs' convenience.

- **Used by.** Lab 11B (off-policy evaluation: IPS / SNIPS / DR with the
  candidate's return as ground truth), Lab 12B (learned linear structural twin
  fitted on `te_logged` and validated against `te_candidate`).

---

## Backblaze Drive Stats

- **Source.** Backblaze (2013-present) — per-day SMART telemetry per drive in
  Backblaze's data centres.
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
- **License.** CC BY 4.0.
- **Upstream size.** ~1 GB per quarterly ZIP (e.g. `data_Q1_2016.zip` →
  ~95 columns × ~13 M rows/quarter). Not vendored.
- **Vendored.** `backblaze_subset.csv` — 42 050 drive-day rows × 10 columns
  (~2 MB).
- **Preprocessing script.** [`backblaze_preprocess.py`](./backblaze_preprocess.py)

### Pipeline (Backblaze raw → vendored subset)

1. **Load.** Read the input — single CSV (e.g. `harddrive.csv`) or a ZIP of
   daily CSVs (the canonical Backblaze quarterly release).
2. **Drop noisy columns.** Keep only `date`, `serial_number`, `model`,
   `failure`, and six SMART raw values:

   | column         | SMART attribute                     | role           |
   |----------------|-------------------------------------|----------------|
   | `smart_5_raw`   | Reallocated Sector Count             | early-warning treatment |
   | `smart_187_raw` | Reported Uncorrectable Errors        | confounder    |
   | `smart_188_raw` | Command Timeout                      | confounder    |
   | `smart_197_raw` | Current Pending Sector Count         | **mediator (Lab 10B)** |
   | `smart_198_raw` | Offline Uncorrectable                | confounder    |
   | `smart_199_raw` | UDMA CRC Error Count                 | confounder    |

   `capacity_bytes` is intentionally dropped — Backblaze stores it as a large
   integer that pandas occasionally re-types as a subnormal float, and the
   labs don't use it.
3. **Filter to one drive model.** The most-prevalent model in the file
   (ST4000DM000 in the 2016 Q1 release we used). A single model keeps the
   population homogeneous so the labs aren't confounded by drive family.
4. **Dedupe.** Drop duplicate `(date, serial_number)` rows; Backblaze
   occasionally double-publishes daily records.
5. **Failure-and-survivor sampling.** Keep ALL drives that fail in the window
   regardless of history length (failures are the rare-event signal); take a
   capped random sample of survivors with at least `MIN_DAYS` of history so
   the per-drive failure rate lands ~3-5 %.
6. **Write CSV** sorted by `(serial_number, date)`.

### Schema cheat-sheet

```
backblaze_subset.csv  (42 050 rows × 10 columns)
  date              YYYY-MM-DD
  serial_number     drive serial
  model             "ST4000DM000"   (constant in the vendored subset)
  failure           1 on the failure day, 0 otherwise
  smart_5_raw       reallocated sectors
  smart_187_raw     reported uncorrectable errors
  smart_188_raw     command timeout
  smart_197_raw     current pending sector count
  smart_198_raw     offline uncorrectable
  smart_199_raw     UDMA CRC errors
```

### Provenance of the vendored slice

- Input file: a multi-day slice of Backblaze 2016 Q1 (`harddrive.csv`,
  1 959 972 rows × 95 columns)
- Parameters: `MIN_DAYS=5`, `SURVIVOR_SAMPLE=1500`, `TARGET_MODEL=ST4000DM000`
- Result: 1576 drives (76 failed + 1500 survivor), 42 050 drive-day rows,
  ~4.8 % drive-level failure rate

### Used by

- **Lab 8B (DTR).** Short-horizon replace-vs-wait policy: state = current
  SMART vector, action = synthesised pre-emptive replacement, outcome =
  failure within the remaining window.
- **Lab 10B (mediation).** `smart_5_raw` → `smart_197_raw` → `failure`,
  with a sensitivity analysis for unmeasured drive-firmware confounding.

---

## OEE synthetic log — curated synthetic for multi-mediator capstones

- **Source.** Generator script. No upstream data; the SCM is documented
  in [`oee_synthetic.py`](./oee_synthetic.py)'s module docstring.
- **License.** MIT (with the rest of the repo).
- **Vendored.** No CSV is shipped — the generator runs in a few
  milliseconds on demand. Deterministic given a fixed seed.
- **Used by.** Capstone Starter E (see `labs/CAPSTONE.md`).

### Structure

Per-shift DataFrame with columns `shift_id, line_id, program, A, P, Q, OEE`
across four production lines. `program` is the maintenance-program
intervention; A/P/Q are the three OEE drivers (Availability ×
Performance × Quality); `OEE = A × P × Q`. The SCM has no direct
`program → OEE` edge — every effect of the intervention flows through
one of the three drivers, which makes the dataset a textbook
multi-mediator NDE/NIE decomposition target.

### Why curated synthetic

A real OEE log with a controllable maintenance intervention requires
internal CMMS access. Public analogs (MetroPT) have time-series
structure but no documented program rollout. The closest public proxy
that includes a documented intervention with clean A/P/Q decomposition
*does not exist*. Curated synthetic with a documented SCM gives the
capstone what no real dataset can: a *verifiable* ground truth for
the mediation decomposition, exposed via
`oee_synthetic.true_oee_decomposition()`.

---

## Multi-site synthetic — curated synthetic for transportability capstones

- **Source.** Generator script. No upstream data; the SCM is documented
  in [`multisite_synthetic.py`](./multisite_synthetic.py)'s module
  docstring.
- **License.** MIT.
- **Vendored.** No CSV — generator runs on demand; deterministic given
  a fixed seed.
- **Used by.** Capstone Starter F.

### Structure

Per-unit DataFrame with columns `unit_id, site, raw_grade, treatment,
outcome`. Two sites A and B share the same SCM in symbolic form; only
the distribution of the `raw_grade` effect modifier differs (Beta(2,5)
at site A vs Beta(5,2) at site B). The treatment effect scales linearly
with `raw_grade`, so the ATE differs between sites by construction
(~0.21 at A, ~0.39 at B). The capstone question: estimate ATE at
source, transport to target via reweighting, validate against the
direct target estimate.

### Why curated synthetic

Cross-site manufacturing data with documented effect modifiers is
almost always internal. Public multi-site datasets either aggregate
sites (losing the modifier structure) or anonymise sites (preventing
domain-grounded transport arguments). Curated synthetic with a known
modifier structure gives the capstone what real data cannot: a
*verifiable* ground truth for the transported estimate, exposed via
`multisite_synthetic.true_ate_per_site()`.

---

## Re-running the preprocessing scripts

The preprocessing scripts run **outside** this container — the upstream raw
files (Severson pickles, Backblaze quarterly ZIPs) are too large to fetch
from a typical sandboxed session. Standard pattern:

```bash
# LFP — after downloading Mattia's three batch pickles to ./raw/
python labs/data/lfp_preprocess.py raw/batch1.pkl raw/batch2.pkl raw/batch3.pkl

# Backblaze — point at your local copy of the Drive Stats CSV / ZIP
python labs/data/backblaze_preprocess.py harddrive.csv labs/data/backblaze_subset.csv
```

Both scripts are deterministic given a fixed input file and seed
(`numpy.random.default_rng(0)`), so re-running on the same upstream snapshot
yields byte-identical CSVs.
