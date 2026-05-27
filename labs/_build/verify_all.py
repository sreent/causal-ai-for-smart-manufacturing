"""Verify every lab notebook runs end-to-end (skipping %pip cells).

Scans `labs/ch01/` through `labs/ch14/` and runs every .ipynb it finds.
Filename-agnostic so renames (Scheme B) don't break the smoke-test.
"""
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
LABS = REPO / "labs"

results = []
for chdir in sorted(LABS.glob("ch[0-9][0-9]")):
    for nb_path in sorted(chdir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        g = {"__name__": "__main__"}
        failed_cell = None
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            non_comment = [l for l in src.split("\n")
                           if l.strip() and not l.strip().startswith("#")]
            if non_comment and non_comment[0].lstrip().startswith(("%", "!")):
                continue
            try:
                exec(src, g)
            except Exception as e:
                failed_cell = (i, type(e).__name__, str(e)[:80])
                break
        rel = nb_path.relative_to(LABS)
        if failed_cell:
            i, exc, msg = failed_cell
            print(f"FAIL {rel}: cell {i}: {exc}: {msg}")
            results.append((rel, False))
        else:
            print(f"OK   {rel}")
            results.append((rel, True))

n_ok = sum(1 for _, ok in results if ok)
print(f"\n{n_ok} / {len(results)} notebooks passed")
