"""Orszag-Tang vortex: 2D compressible ideal MHD on a periodic box [0, 2 pi)^2.

Accompanies Section 9 of "Kinetic Hierarchies: from Liouville to Euler".

The conserved fluid variables (rho, rho vx, rho vy, E) are evolved by a finite-volume
scheme: primitive states are reconstructed to cell faces with minmod-limited MUSCL,
fluxes are Rusanov (local Lax-Friedrichs, fast-magnetosonic speed), and time stepping
is two-stage RK2 with CFL number 0.4, gamma = 5/3.  The in-plane field is carried by
a vector potential A (B = curl A z_hat), advected as dA/dt = -u . grad A with
second-order upwind differences; B is formed from A by central differences, so its
discrete divergence vanishes to round-off.  The smooth initial vortex steepens into
interacting shocks and a thin central current sheet by t ~ 3.

Run:  python orszag_tang.py [--n 128]     (writes ../figures/orszag_tang_density.png)
"""
import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
FLOOR = 1e-6

# arrays are indexed [j, i] = [y, x]; axis 1 is x, axis 0 is y


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "sans-serif",
        "axes.grid": False, "axes.edgecolor": "#888888", "font.size": 10,
    })


def minmod(a, b):
    return np.where(a * b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


class OrszagTang:
    def __init__(self, n=128, gam=5 / 3, cfl=0.4):
        self.n, self.gam, self.cfl = n, gam, cfl
        self.h = h = 2 * np.pi / n
        c = (np.arange(n) + 0.5) * h
        x, y = np.meshgrid(c, c)                             # x varies along axis 1
        g = gam
        rho, p = g * g, g
        vx, vy = -np.sin(y), np.sin(x)
        bx, by = -np.sin(y), np.sin(2 * x)
        self.A = np.cos(y) + 0.5 * np.cos(2 * x)
        self.U = np.array([
            np.full_like(x, rho), rho * vx, rho * vy,
            p / (g - 1) + 0.5 * rho * (vx ** 2 + vy ** 2) + 0.5 * (bx ** 2 + by ** 2),
        ])
        self.t = 0.0

    def B(self, A):
        """B_x = dA/dy, B_y = -dA/dx (central differences, cell-centred)."""
        h = self.h
        bx = (np.roll(A, -1, 0) - np.roll(A, 1, 0)) / (2 * h)
        by = -(np.roll(A, -1, 1) - np.roll(A, 1, 1)) / (2 * h)
        return bx, by

    def rhs(self, U, A):
        h, g = self.h, self.gam
        bx, by = self.B(A)
        r = U[0]
        vx, vy = U[1] / r, U[2] / r
        p = np.maximum((g - 1) * (U[3] - 0.5 * r * (vx ** 2 + vy ** 2) - 0.5 * (bx ** 2 + by ** 2)), FLOOR)
        P = np.array([r, vx, vy, p])
        dU = np.zeros_like(U)
        smax = 0.0
        for d in (0, 1):                                    # 0: x-faces, 1: y-faces
            ax = 1 if d == 0 else 0
            sh = lambda a, k: np.roll(a, -k, axis=ax + 1 if a.ndim == 3 else ax)
            Pp, Pm, Ppp = sh(P, 1), sh(P, -1), sh(P, 2)
            # reconstruct primitive states at face k+1/2
            WL = P + 0.5 * minmod(P - Pm, Pp - P)
            WR = Pp - 0.5 * minmod(Pp - P, Ppp - Pp)
            WL[0] = np.maximum(WL[0], FLOOR); WR[0] = np.maximum(WR[0], FLOOR)
            WL[3] = np.maximum(WL[3], FLOOR); WR[3] = np.maximum(WR[3], FLOOR)
            bnc, btc = (bx, by) if d == 0 else (by, bx)
            bn = 0.5 * (bnc + sh(bnc, 1))                   # normal B continuous: face average
            btL, btR = btc, sh(btc, 1)

            def fast(W, bt):
                b2 = bn ** 2 + bt ** 2
                a2, va2 = g * W[3] / W[0], b2 / W[0]
                return np.sqrt(0.5 * (a2 + va2 + np.sqrt(np.maximum((a2 + va2) ** 2 - 4 * a2 * bn ** 2 / W[0], 0))))

            iv = 1 + d                                      # index of normal velocity in W
            s = np.maximum(np.abs(WL[iv]) + fast(WL, btL), np.abs(WR[iv]) + fast(WR, btR))
            smax = max(smax, s.max())

            def flux_and_state(W, bt):
                rr, ux, uy, pp = W
                vn, vt = (ux, uy) if d == 0 else (uy, ux)
                bxf, byf = (bn, bt) if d == 0 else (bt, bn)
                b2 = bn ** 2 + bt ** 2
                pt = pp + 0.5 * b2
                E = pp / (g - 1) + 0.5 * rr * (ux ** 2 + uy ** 2) + 0.5 * b2
                fn = rr * vn * vn + pt - bn * bn
                ft = rr * vn * vt - bn * bt
                F = np.array([rr * vn, fn if d == 0 else ft, ft if d == 0 else fn,
                              (E + pt) * vn - bn * (ux * bxf + uy * byf)])
                Us = np.array([rr, rr * ux, rr * uy, E])
                return F, Us

            FL, UL = flux_and_state(WL, btL)
            FR, UR = flux_and_state(WR, btR)
            F = 0.5 * (FL + FR) - 0.5 * s * (UR - UL)      # Rusanov flux at face k+1/2
            dU -= (F - np.roll(F, 1, axis=ax + 1)) / h
        # A is advected: dA/dt = -u . grad A, second-order upwind
        Axm = (3 * A - 4 * np.roll(A, 1, 1) + np.roll(A, 2, 1)) / (2 * h)
        Axp = (-3 * A + 4 * np.roll(A, -1, 1) - np.roll(A, -2, 1)) / (2 * h)
        Aym = (3 * A - 4 * np.roll(A, 1, 0) + np.roll(A, 2, 0)) / (2 * h)
        Ayp = (-3 * A + 4 * np.roll(A, -1, 0) - np.roll(A, -2, 0)) / (2 * h)
        dA = -(vx * np.where(vx > 0, Axm, Axp) + vy * np.where(vy > 0, Aym, Ayp))
        return dU, dA, smax

    def step(self, dt_max=np.inf):
        d1, a1, s = self.rhs(self.U, self.A)
        dt = min(self.cfl * self.h / s, dt_max)
        U1, A1 = self.U + dt * d1, self.A + dt * a1
        d2, a2, _ = self.rhs(U1, A1)
        self.U = 0.5 * (self.U + U1 + dt * d2)
        self.A = 0.5 * (self.A + A1 + dt * a2)
        self.t += dt
        return dt

    def advance_to(self, t_target):
        steps = 0
        while self.t < t_target - 1e-12:
            self.step(t_target - self.t)
            steps += 1
        return steps

    def div_b(self):
        """Central-difference divergence of the central-difference B."""
        bx, by = self.B(self.A)
        h = self.h
        return ((np.roll(bx, -1, 1) - np.roll(bx, 1, 1)) + (np.roll(by, -1, 0) - np.roll(by, 1, 0))) / (2 * h)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=128, help="grid cells per side (default 128)")
    args = ap.parse_args()
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    sim = OrszagTang(n=args.n)
    times = [0.5, 1.0, 2.0, 3.0]
    snaps, nsteps = {}, 0
    for tt in times:
        nsteps += sim.advance_to(tt)
        snaps[tt] = sim.U[0].copy()
        print(f"t = {sim.t:.3f}  steps = {nsteps}  rho in [{snaps[tt].min():.3f}, {snaps[tt].max():.3f}]"
              f"  ({time.time() - t0:.1f} s)")
    rho = snaps[3.0]
    B = np.hypot(*sim.B(sim.A))
    print(f"n = {args.n}: density at t = 3: min {rho.min():.3f}, max {rho.max():.3f}")
    print(f"max|div B| = {np.abs(sim.div_b()).max():.2e}  (max|B| = {B.max():.2f})")

    L = 2 * np.pi
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), constrained_layout=True)
    for ax, tt in zip(axes, times):
        im = ax.imshow(snaps[tt], origin="lower", extent=(0, L, 0, L), cmap="magma",
                       vmin=1.0, vmax=6.3, interpolation="nearest")
        ax.set_title(f"t = {tt:g}")
        ax.set_xlabel("x  [code length units]")
        ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "2$\\pi$"])
        ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "2$\\pi$"])
    axes[0].set_ylabel("y  [code length units]")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("density $\\rho$  [code units, $\\rho_0 = \\gamma^2$]")
    fig.suptitle(f"Orszag-Tang vortex, ideal MHD ({args.n}$^2$, Rusanov + MUSCL, RK2)", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "orszag_tang_density.png"), dpi=150)
    plt.close(fig)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
