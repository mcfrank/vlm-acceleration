"""Shared setup for the book's chapters: paths, palette, and small helpers."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THEORY = ROOT / "theory"
RES = THEORY / "results"
sys.path.insert(0, str(THEORY))

# Reference categorical palette, light mode, fixed slot order (dataviz skill).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.figsize": (6.4, 3.8), "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "lines.markersize": 6, "legend.frameon": False, "font.size": 10,
})

def md(df, floatfmt=".2f", fmt=None):
    """Print a DataFrame as a markdown table (use with #| output: asis).
    fmt maps column names to format strings, e.g. {"ρ": "{:g}"}, applied before printing.
    Passing fmt (even {}) also prints string cells verbatim instead of re-parsing them as numbers."""
    df = df.copy()
    for col, f in (fmt or {}).items():
        df[col] = [f.format(v) for v in df[col]]
    print(df.to_markdown(index=False, floatfmt=floatfmt, disable_numparse=fmt is not None))

def direct_label(ax, x, y, text, color=INK2, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", fontsize=9, color=color)
