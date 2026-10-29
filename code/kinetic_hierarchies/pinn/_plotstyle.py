"""Shared matplotlib style for the figures of "Kinetic Hierarchies: from Liouville to Euler"."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUES = ["#7aaee0", "#3D7EBF", "#1d4677"]
INK, MUTED, GRID = "#1c2530", "#5b6b7a", "#e3e8ee"


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, stem, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=150)
    fig.savefig(os.path.join(outdir, stem + ".svg"))
    plt.close(fig)
