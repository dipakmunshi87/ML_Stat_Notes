"""Dynamical friction on a heavy body orbiting in a halo of stars (restricted N-body).

Accompanies Section 7 (case three) of "Kinetic Hierarchies: from Liouville to Euler".

Units G = M_halo = a = 1.  N tracer stars (mass 1/N each, so together they ARE the
halo) move in a fixed 2D Plummer potential, Phi = -1/sqrt(r^2 + 1), and also feel a
heavy body of mass Mb through a softened force.  The body feels only the stars.
The stars are first relaxed for 1500 steps in the fixed potential; the body is then
started on the circular orbit of the stars' smoothed mean field (softening e2 = 0.09,
independent of the softening under test).  The body's wake of deflected stars pulls
back on it (Chandrasekhar friction) and it spirals inward.  Friction scales with the
Coulomb logarithm ln(b_max/b_min), and the softening eps sets b_min: with eps = 0.02
the body sinks much faster than with eps = 0.5, where close encounters are smeared out.

Run:  python dynamical_friction.py   (writes ../figures/friction_*.png)
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
BODY = "#d03b3b"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "sans-serif",
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
        "axes.edgecolor": "#888888", "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10,
    })


class Friction:
    def __init__(self, N=3000, Mb=0.04, eps=0.05, r0=2.5, dt=0.01, seed=3):
        self.N, self.Mb, self.eps, self.r0, self.dt = N, Mb, eps, r0, dt
        self.m = 1.0 / N                                     # stars carry the halo's mass
        rng = np.random.default_rng(seed)
        # 2D Plummer-like disc: radius from the projected Plummer profile, near-isotropic velocities
        u = rng.random(N)
        rr = np.minimum(np.sqrt(u / (1 - u)), 8.0)
        th = 2 * np.pi * rng.random(N)
        self.x, self.y = rr * np.cos(th), rr * np.sin(th)
        vc = np.sqrt(rr ** 2 / (rr ** 2 + 1) ** 1.5)
        s = 0.55 * vc + 0.12                                 # hot: dispersion comparable to rotation
        g = rng.standard_normal((2, N))
        vt, vr = 0.45 * vc + s * g[0], s * g[1]
        self.vx = vr * np.cos(th) - vt * np.sin(th)
        self.vy = vr * np.sin(th) + vt * np.cos(th)

        # let the tracers settle in the fixed potential before the body is added
        for _ in range(1500):
            ih = (self.x ** 2 + self.y ** 2 + 1) ** -1.5
            self.vx -= 0.01 * self.x * ih
            self.vy -= 0.01 * self.y * ih
            self.x += 0.01 * self.vx
            self.y += 0.01 * self.vy

        # the body feels only the stars: start it on the circular orbit of their mean field,
        # estimated with a smooth softening independent of the eps under test
        e2 = 0.09
        ang = np.arange(32) * np.pi / 16
        bx, by = r0 * np.cos(ang)[:, None], r0 * np.sin(ang)[:, None]
        dx, dy = bx - self.x[None, :], by - self.y[None, :]
        d2 = dx ** 2 + dy ** 2 + e2
        gr = np.sum(self.m * (dx * bx + dy * by) / r0 / (d2 * np.sqrt(d2))) / 32
        self.b = np.array([r0, 0.0, 0.0, np.sqrt(max(gr, 1e-6) * r0)])   # x, y, vx, vy
        self.t = 0.0
        self.hist_t, self.hist_r = [0.0], [r0]
        self.trail = [self.b[:2].copy()]

    def step(self):
        dt, e2, Mb, m = self.dt, self.eps ** 2, self.Mb, self.m
        bx, by = self.b[0], self.b[1]
        x, y = self.x, self.y
        ih = (x * x + y * y + 1) ** -1.5                     # Plummer pull
        dx, dy = bx - x, by - y
        d2 = dx * dx + dy * dy + e2
        idd = 1.0 / (d2 * np.sqrt(d2))
        self.vx += dt * (-x * ih + Mb * dx * idd)
        self.vy += dt * (-y * ih + Mb * dy * idd)
        ax, ay = -m * np.sum(dx * idd), -m * np.sum(dy * idd)   # body force from pre-kick star positions
        self.x = x + dt * self.vx
        self.y = y + dt * self.vy
        self.b[2] += dt * ax
        self.b[3] += dt * ay
        self.b[0] += dt * self.b[2]
        self.b[1] += dt * self.b[3]
        self.t += dt
        self.hist_t.append(self.t)
        self.hist_r.append(np.hypot(self.b[0], self.b[1]))
        self.trail.append(self.b[:2].copy())


def run(eps, t_end=60.0, snap_times=(8.0, 30.0)):
    sim = Friction(N=3000, Mb=0.04, eps=eps, r0=2.5, dt=0.01)
    nsteps = int(round(t_end / sim.dt))
    snap_steps = {int(round(ts / sim.dt)): ts for ts in snap_times}
    snaps = {}
    for n in range(1, nsteps + 1):
        sim.step()
        if n in snap_steps:
            snaps[snap_steps[n]] = (sim.x.copy(), sim.y.copy(), np.array(sim.trail))
    return sim, snaps


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    runs = {}
    for eps in (0.02, 0.5):
        t1 = time.time()
        runs[eps] = run(eps)
        r = np.array(runs[eps][0].hist_r)
        tt = np.array(runs[eps][0].hist_t)
        print(f"eps = {eps}: r(t=30) = {r[np.argmin(abs(tt - 30))]:.3f}, r(t=60) = {r[-1]:.3f}"
              f"  ({time.time() - t1:.1f} s)")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for eps, c in ((0.02, BLUE), (0.5, ORANGE)):
        sim = runs[eps][0]
        ax.plot(sim.hist_t, sim.hist_r, color=c, lw=1.6, label=f"softening $\\epsilon$ = {eps}")
    ax.set_xlabel("t  [$(a^3/GM)^{1/2}$]")
    ax.set_ylabel("orbit radius r  [Plummer scale a]")
    ax.set_title("Dynamical friction: sinking body ($M_b$ = 0.04, N = 3000)")
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "friction_radius.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for row, eps in enumerate((0.02, 0.5)):
        snaps = runs[eps][1]
        for col, ts in enumerate((8.0, 30.0)):
            ax = axes[row, col]
            x, y, trail = snaps[ts]
            ax.scatter(x, y, s=0.6, c="#52514e", alpha=0.6, linewidths=0, rasterized=True)
            ax.plot(trail[:, 0], trail[:, 1], color=BODY, lw=0.8, alpha=0.6)
            ax.plot(trail[-1, 0], trail[-1, 1], "o", color=BODY, ms=7, mec="white", mew=1.2)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.set_title(f"$\\epsilon$ = {eps},  t = {ts:g}")
            ax.set_xlabel("x  [a]")
            ax.set_ylabel("y  [a]")
    fig.suptitle("Stars (grey) and heavy body with its trail (red)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "friction_snapshots.png"), dpi=150)
    plt.close(fig)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
