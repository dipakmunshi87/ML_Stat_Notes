"""Two-stream instability with a 1D-1V electrostatic particle-in-cell code.

Accompanies Section 9 of "Kinetic Hierarchies: from Liouville to Euler".

Two cold electron beams (velocities +v0 and -v0, thermal spread vth) stream
through a fixed neutralising ion background in a periodic box of length L.
Plasma units: omega_p = 1, electron charge/mass = -1.  Each step deposits the
electron charge on a grid with cloud-in-cell weights, solves Gauss's law
dE/dx = 1 - n_e by a cumulative sum (minus its mean, so <E> = 0), interpolates
E back to the particles and advances them with leapfrog.  A tiny cosine
displacement seeds the fundamental mode k = 2 pi / L, which grows exponentially
(field energy ~ exp(2 gamma t)) until the beams trap and roll up into phase-space
vortices.  Cold-beam theory for k v0 = 0.5 gives gamma = 0.341.

Run:  python two_stream_pic.py      (writes ../figures/two_stream_*.png)
"""
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")

BLUE, ORANGE = "#2a78d6", "#eb6834"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "sans-serif",
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
        "axes.edgecolor": "#888888", "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10,
    })


class TwoStream:
    def __init__(self, N=20000, Ng=64, L=4 * np.pi, v0=1.0, vth=0.05, dt=0.1, seed=7):
        self.N, self.Ng, self.L, self.v0, self.vth, self.dt = N, Ng, L, v0, vth, dt
        self.dx = L / Ng
        rng = np.random.default_rng(seed)
        i = np.arange(N)
        self.beam = i % 2                                   # 1: +v0 beam, 0: -v0 beam
        x = (i // 2 + 0.5) * L / (N / 2)                    # quiet start: pairs on a uniform lattice
        x = x + 1e-3 * L * np.cos(2 * np.pi * x / L)        # seed the fundamental mode
        self.x = np.mod(x, L)
        self.v = np.where(self.beam == 1, 1.0, -1.0) * v0 + vth * rng.standard_normal(N)
        self.t = 0.0
        self.field()
        self.hist_t, self.hist_W = [0.0], [self.WE]
        self.v += 0.5 * dt * self.E_at(self.x)              # half kick back: v lives at t - dt/2

    def _cic(self, x):
        g = x / self.dx - 0.5
        j = np.floor(g).astype(np.int64)
        f = g - j
        return np.mod(j, self.Ng), np.mod(j + 1, self.Ng), f

    def field(self):
        Ng, dx = self.Ng, self.dx
        j0, j1, f = self._cic(self.x)                      # cloud-in-cell deposit
        rho = np.bincount(j0, weights=1 - f, minlength=Ng) + np.bincount(j1, weights=f, minlength=Ng)
        n0 = self.N / Ng
        E = np.cumsum((1 - rho / n0) * dx)                 # Gauss: dE/dx = 1 - n_e
        E -= E.mean()
        self.E, self.rho = E, rho
        self.WE = 0.5 * np.sum(E * E) * dx

    def E_at(self, x):
        j0, j1, f = self._cic(x)
        return (1 - f) * self.E[j0] + f * self.E[j1]

    def step(self):
        self.v -= self.dt * self.E_at(self.x)              # electrons: dv/dt = -E
        self.x = np.mod(self.x + self.dt * self.v, self.L)
        self.field()
        self.t += self.dt
        self.hist_t.append(self.t)
        self.hist_W.append(self.WE)


def cold_growth_rate(k, v0):
    """Cold symmetric two-stream (each beam omega_p^2 = 1/2): solve for Im(omega)."""
    a2 = (k * v0) ** 2
    z = (2 * a2 + 1 - np.sqrt(8 * a2 + 1)) / 2            # omega^2 of the unstable branch
    return np.sqrt(-z) if z < 0 else 0.0


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    sim = TwoStream()
    snap_steps = {0: 0.0, 150: 15.0, 250: 25.0, 400: 40.0}
    snaps = {}
    nsteps = 400
    for n in range(nsteps + 1):
        if n in snap_steps:
            snaps[snap_steps[n]] = (sim.x.copy(), sim.v.copy())
        if n < nsteps:
            sim.step()
    t = np.array(sim.hist_t)
    W = np.array(sim.hist_W)

    # linear growth: fit log W_E on the clean exponential stretch
    fit = (t >= 8) & (t <= 18)
    slope, icpt = np.polyfit(t[fit], np.log(W[fit]), 1)
    gamma = slope / 2
    gth = cold_growth_rate(2 * np.pi / sim.L, sim.v0)
    print(f"field-energy growth rate d ln W_E/dt = {slope:.4f}  ->  gamma = {gamma:.4f}")
    print(f"cold-beam theory for k = 2pi/L: gamma = {gth:.4f}")

    # phase space
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, (ts, (x, v)) in zip(axes, sorted(snaps.items())):
        for b, c in ((1, ORANGE), (0, BLUE)):
            m = sim.beam == b
            ax.scatter(x[m], v[m], s=0.3, c=c, alpha=0.5, linewidths=0, rasterized=True)
        ax.set_title(f"t = {ts:g} $\\omega_p^{{-1}}$")
        ax.set_xlim(0, sim.L)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("x  [$v_0/\\omega_p$]")
    axes[0].set_ylabel("v  [$v_0$]")
    fig.suptitle("Two-stream instability: electron phase space (orange: +v$_0$ beam, blue: -v$_0$ beam)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "two_stream_phase_space.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(t, W, color=BLUE, lw=1.8, label="PIC field energy $W_E$")
    tf = np.linspace(4, 22, 50)
    ax.semilogy(tf, np.exp(icpt + slope * tf), "--", color=ORANGE, lw=1.4,
                label=f"fit t = 8-18:  $\\gamma$ = {gamma:.3f} $\\omega_p$")
    ax.axvspan(8, 18, color="#eeeeee", zorder=0)
    ax.set_xlabel("t  [$\\omega_p^{-1}$]")
    ax.set_ylabel("$W_E = \\frac{1}{2}\\int E^2\\,dx$  [plasma units]")
    ax.set_title(f"Two-stream field energy (cold theory $\\gamma$ = {gth:.3f})")
    ax.set_xlim(0, t[-1])
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "two_stream_field_energy.png"), dpi=150)
    plt.close(fig)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
