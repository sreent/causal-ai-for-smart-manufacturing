# Tennessee Eastman simulator (vendored)

This directory vendors the [`tep2py`](https://github.com/camaramm/tep2py) Python
wrapper around the Downs & Vogel (1993) Tennessee Eastman Process simulator,
so labs 11B and 12B can be reproduced and extended without depending on a
specific upstream package version.

## Files

- `tep2py.py` — the Python wrapper (patched for NumPy 2.x: `np.integer` → `np.int64`, `np.int(...)` → `int(...)`)
- `temain_mod.f`, `teprob.f` — Fortran source (Downs & Vogel TE simulator, modified by [gmxavier](https://github.com/gmxavier/TEP-meets-LSTM))
- `temain_mod.pyf` — f2py signature file
- `LICENSE` — original tep2py license

## Pre-generated data

For the labs, the canonical trajectories are committed as CSVs alongside this directory:

- `../te_logged.csv` — 500 samples under a Bernoulli(0.5) IDV(1) action (the logged behaviour policy for Lab 11B's OPE setup)
- `../te_candidate.csv` — 500 samples under IDV(1) = 0 always (the evaluation policy ground truth)

Most labs use these CSVs directly via `te_prep.load_te(scenario)`. Re-simulation is only needed for extension exercises.

## Re-simulating in Colab

```bash
apt-get install -y gfortran           # Colab usually has this
cd labs/data/te_simulator
python -m numpy.f2py -c temain_mod.pyf temain_mod.f teprob.f -m temain_mod
```

Then from Python:

```python
import sys, numpy as np
sys.path.insert(0, "labs/data/te_simulator")
from tep2py import tep2py

idata = np.zeros((500, 20))            # 500 steps, 20 IDV columns
idata[:, 0] = np.random.binomial(1, 0.5, 500)
tep = tep2py(idata)
tep.simulate()
df = tep.process_data
```

## Citation

> G. M. Xavier and J. M. de Seixas, "Fault Detection and Diagnosis in a Chemical Process using Long Short-Term Memory Recurrent Neural Network," 2018 International Joint Conference on Neural Networks (IJCNN), 2018, pp. 1-8, doi: 10.1109/IJCNN.2018.8489385.

Original TE problem: Downs, J.J. and Vogel, E.F. (1993) "A plant-wide industrial process control problem," *Computers and Chemical Engineering*, 17(3), 245-255.
