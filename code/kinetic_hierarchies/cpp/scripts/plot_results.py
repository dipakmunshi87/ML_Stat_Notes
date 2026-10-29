"""Plot the output of pic_two_stream: field energy vs time with the theoretical growth rate, and the last
phase-space snapshot if one was written.   Usage:  python scripts/plot_results.py output/
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "output"
d = np.genfromtxt(os.path.join(out, "diagnostics.csv"), delimiter=",", names=True)
L, v0 = 4 * np.pi, 1.0
a2 = (2 * np.pi / L * v0) ** 2
gamma = np.sqrt(-0.5 * ((2 * a2 + 1) - np.sqrt(8 * a2 + 1)))

snaps = sorted(glob.glob(os.path.join(out, "phase_*.csv")))
fig, ax = plt.subplots(1, 2 if snaps else 1, figsize=(10 if snaps else 5.5, 3.8), squeeze=False)
ax = ax[0]
ax[0].semilogy(d["time"], d["field_energy"], lw=2, color="#3D7EBF", label="field energy")
t = np.linspace(6, 18, 2); i = np.argmin(abs(d["time"] - 12))
ax[0].semilogy(t, d["field_energy"][i] * np.exp(2 * gamma * (t - 12)), "--", color="#1c2530", label=f"theory, 2γ = {2*gamma:.3f}")
ax[0].set_xlabel("time (1/ω_p)"); ax[0].set_ylabel("field energy"); ax[0].legend(frameon=False)
if snaps:
    p = np.genfromtxt(snaps[-1], delimiter=",", names=True)
    ax[1].plot(p["x"], p["v"], ",", color="#245a95", alpha=0.6)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("v"); ax[1].set_title(os.path.basename(snaps[-1]))
fig.tight_layout(); fig.savefig(os.path.join(out, "pic_results.png"), dpi=150)
print("wrote", os.path.join(out, "pic_results.png"))
