"""Tiny helper to build Colab-compatible .ipynb files from a list of cells.

Usage:
    from _nb import md, code, write_notebook
    cells = [
        md("# Lab N — Title"),
        md("Introduction text..."),
        code("import numpy as np\nprint('hello')"),
        ...
    ]
    write_notebook("lab01.ipynb", cells)
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}


def code(text, outputs=None):
    src = text.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": src,
    }


def write_notebook(path, cells, title=None):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
            **({"title": title} if title else {}),
        },
        "cells": cells,
    }
    # Trailing newlines normalized: each cell.source line keeps its \n except the last.
    for c in nb["cells"]:
        src = c["source"]
        if not src:
            continue
        c["source"] = [(s + "\n") if i < len(src) - 1 else s for i, s in enumerate(src)]
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
