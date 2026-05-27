"""Verify all labs run end-to-end (skipping %pip cells)."""
import json, sys, matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

for n in range(1, 14):
    path = f"labs/lab{n:02d}.ipynb"
    nb = json.load(open(path))
    g = {"__name__": "__main__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # Skip pip-install cells (first non-comment/blank line starts with %)
        non_comment = [l for l in src.split("\n") if l.strip() and not l.strip().startswith("#")]
        if non_comment and non_comment[0].lstrip().startswith("%"):
            continue
        try:
            exec(src, g)
        except Exception as e:
            print(f"lab{n:02d} cell {i} FAIL: {type(e).__name__}: {str(e)[:80]}")
            break
    else:
        print(f"lab{n:02d} OK")
