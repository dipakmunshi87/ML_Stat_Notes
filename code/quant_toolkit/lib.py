"""Shared helpers for figures and computed values."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs"); os.makedirs(FIGD, exist_ok=True)
VALF = os.path.join(HERE, "values.json")

# validated categorical palette (dataviz skill reference instance)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUT, LINE, SOFT = "#1c2530", "#5b6b7a", "#e3e8ee", "#f6f8fb"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.titlecolor": INK, "axes.titleweight": "600",
    "axes.titlesize": 11.5, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 9.5, "figure.dpi": 110, "savefig.dpi": 110, "lines.linewidth": 2.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def fig(w=8.4, h=4.2, ncols=1, nrows=1, **kw):
    f, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return f, ax

def save(f, name):
    f.tight_layout()
    p = os.path.join(FIGD, name + ".png")
    f.savefig(p, bbox_inches="tight")
    plt.close(f)
    print("fig", name)

_vals = {}
if os.path.exists(VALF):
    _vals = json.load(open(VALF))

def V(key, value, fmt=None):
    """Store a computed value for substitution as {{v:key}}."""
    if isinstance(value, (np.floating, np.integer)): value = value.item()
    _vals[key] = (format(value, fmt) if fmt else value)
    return value

def flush():
    json.dump(_vals, open(VALF, "w"), indent=1, sort_keys=True)
    print("values:", len(_vals))
