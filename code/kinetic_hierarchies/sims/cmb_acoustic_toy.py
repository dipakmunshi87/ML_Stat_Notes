"""CMB acoustic toy: tight-coupled photon-baryon oscillations frozen at recombination.

Accompanies Section 10 of "Kinetic Hierarchies: from Liouville to Euler".

A Hu & Sugiyama-style matter-era approximation for the effective temperature
on a flat 2D patch, mode by mode in Fourier space:
    [Theta_0 + Psi](k, eta) = Psi_k [ (1+3R)/3 cos(k r_s) - R ] exp(-k^2 / k_D^2),
with sound horizon r_s = (pi / kPeak) eta (first compression peak at k = kPeak when
eta = 1, i.e. recombination) and Silk damping scale k_D = kDstar / sqrt(eta), which
moves to larger scales as photons random-walk further.  The potentials Psi_k are a
Gaussian random field with scale-invariant 2D amplitude 1/k.  Baryon loading R shifts
the oscillation zero point by -R, so compression (odd) peaks grow relative to
rarefaction (even) peaks: the odd/even asymmetry that measures the baryon density.

Run:  python cmb_acoustic_toy.py     (writes ../figures/cmb_toy_*.png)
"""
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")

# blue - grey - red diverging map with a neutral midpoint
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#e6e5e1", "#e34948", "#8a1c1c"])
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "sans-serif",
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
        "axes.edgecolor": "#888888", "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10,
    })


class CMB:
    def __init__(self, n=128, R=0.6, kPeak=7.0, kDstar=30.0, seed=11):
        self.n, self.R, self.kPeak, self.kDstar = n, R, kPeak, kDstar
        rng = np.random.default_rng(seed)
        kint = np.fft.fftfreq(n, d=1.0 / n)                 # integer wavenumbers 0..n/2-1, -n/2..-1
        kx, ky = np.meshgrid(kint, kint)                    # [j, i] = [ky, kx]
        self.kk = np.hypot(kx, ky)
        amp = np.where(self.kk > 0, 1.0 / np.where(self.kk > 0, self.kk, 1.0), 0.0)
        self.psi = amp * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))

    def transfer(self, k, eta, R=None):
        R = self.R if R is None else R
        rs = np.pi / self.kPeak * eta
        kD = self.kDstar / np.sqrt(max(eta, 1e-4))
        return ((1 + 3 * R) / 3 * np.cos(k * rs) - R) * np.exp(-(k * k) / (kD * kD))

    def map(self, eta):
        """Real part of the (unnormalised) inverse 2D FFT of Psi_k T(k, eta)."""
        n = self.n
        return np.real(np.fft.ifft2(self.psi * self.transfer(self.kk, eta))) * n * n


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    cmb = CMB()

    etas = [0.25, 0.5, 1.0]
    maps = [cmb.map(e) for e in etas]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
    for ax, eta, m in zip(axes, etas, maps):
        lim = np.abs(m).max()                               # symmetric limits per panel
        im = ax.imshow(m, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim,
                       extent=(0, 1, 0, 1), interpolation="bilinear")
        ax.set_title(f"$\\eta/\\eta_*$ = {eta:g}")
        ax.set_xlabel("x  [patch size]")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("$\\Theta_0 + \\Psi$  [arb. units]")
    axes[0].set_ylabel("y  [patch size]")
    fig.suptitle(f"CMB acoustic toy: effective temperature on a 2D patch (R = {cmb.R})", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "cmb_toy_maps.png"), dpi=150)
    plt.close(fig)

    k = np.linspace(0, 45, 1200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for R, c in zip((0.0, 0.3, 0.6, 1.0), SERIES):
        T2 = cmb.transfer(k, 1.0, R) ** 2
        ax.plot(k, T2, color=c, lw=1.8, label=f"R = {R:g}")
        # report odd/even peak heights near k = kPeak, 2 kPeak, 3 kPeak
        pk = [T2[np.argmin(abs(k - p * cmb.kPeak))] for p in (1, 2, 3)]
        print(f"R = {R:.1f}: |T|^2 at k = 7, 14, 21: " + ", ".join(f"{v:.3f}" for v in pk)
              + f"   (1st/2nd = {pk[0] / pk[1]:.1f})")
    for p in (1, 2, 3, 4, 5):
        lab = f"{p}: " + ("compr." if p % 2 else "raref.")
        ax.axvline(p * cmb.kPeak, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
        ax.text(p * cmb.kPeak, 1.15e-3, lab, fontsize=7.5, color="#52514e",
                ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10)
    ax.set_xlim(0, 45)
    ax.set_xlabel("wavenumber k  [modes per patch]")
    ax.set_ylabel("$|T(k, \\eta_*)|^2$  [dimensionless]")
    ax.set_title("Acoustic transfer at recombination: baryons boost the odd peaks")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "cmb_toy_spectrum.png"), dpi=150)
    plt.close(fig)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
