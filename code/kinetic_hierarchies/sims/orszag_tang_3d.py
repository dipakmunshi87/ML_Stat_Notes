"""3D ideal MHD: a three-dimensional Orszag-Tang vortex (Sections 9 and Appendix A.6 of "Kinetic Hierarchies").

Compressible resistive MHD (small eta; eta = 0 is ideal MHD) on a periodic box [0, 2pi]^3, gamma = 5/3, in units with mu0 = 1:
    d rho/dt + div(rho u) = 0
    d(rho u)/dt + div[rho u u + (p + B^2/2) I - B B] = 0
    dE/dt + div[(E + p + B^2/2) u - B (u.B)] = 0
    B = curl A,     dA/dt + (u.grad) A = -(grad u)^T . A          (advective gauge for the vector potential)
Because B is computed as the curl of A with central differences, div B = 0 to round-off at every step: the 3D
analogue of the vector-potential trick used in the 2D animation, and the idea behind constrained transport.

Initial state: the classic 2D Orszag-Tang flow and field, plus a field component that varies along z, which couples
the planes and makes the problem genuinely three-dimensional:
    rho = gamma^2, p = gamma, u = (-sin y, sin x, 0),
    A = (-beta cos z, beta cos z, cos y + cos(2x)/2)  ->  B = (-sin y + beta sin z, sin 2x + beta sin z, 0).
With beta = 0 the run is exactly the 2D vortex at every z (used as a regression test: --beta 0 --check2d).

Numerics: finite volume, Rusanov (local Lax-Friedrichs) fluxes, minmod-limited linear reconstruction, RK2, CFL 0.35;
A advected with second-order upwinding. Vectorised with numpy.roll.

Run:  python orszag_tang_3d.py                 (64^3 to t = 3: a few minutes on a laptop)
      python orszag_tang_3d.py --n 96          (about 3.5x longer)
Writes ../figures/ot3d_*.png, ot3d_frames.npz (for the page animation) and ot3d.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pinn"))
from _plotstyle import setup, save, BLUES, INK, MUTED

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--tend", type=float, default=3.0)
ap.add_argument("--beta", type=float, default=0.5, help="strength of the z-dependent field (0: pure 2D vortex)")
ap.add_argument("--cfl", type=float, default=0.35)
ap.add_argument("--eta", type=float, default=1e-2, help="resistivity (magnetic diffusivity); keeps current sheets resolved")
ap.add_argument("--frames", type=int, default=40, help="snapshots stored for the animation")
ap.add_argument("--check2d", action="store_true", help="with --beta 0: verify every z-plane is identical")
args = ap.parse_args()

G = 5.0 / 3.0
N = args.n; Lbox = 2 * np.pi; h = Lbox / N
r = lambda f, s, ax: np.roll(f, -s, axis=ax)          # r(f, +1, ax) = f at i+1


def minmod(a, b):
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def curl(Ax, Ay, Az):
    d = lambda f, ax: (r(f, 1, ax) - r(f, -1, ax)) / (2 * h)
    return d(Az, 1) - d(Ay, 2), d(Ax, 2) - d(Az, 0), d(Ay, 0) - d(Ax, 1)


def init():
    x = (np.arange(N) + 0.5) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.full_like(X, G * G); p = np.full_like(X, G)
    ux, uy, uz = -np.sin(Y), np.sin(X), np.zeros_like(X)
    A = [-args.beta * np.cos(Z), args.beta * np.cos(Z), np.cos(Y) + 0.5 * np.cos(2 * X)]
    bx, by, bz = curl(*A)
    E = p / (G - 1) + 0.5 * rho * (ux**2 + uy**2 + uz**2) + 0.5 * (bx**2 + by**2 + bz**2)
    U = np.stack([rho, rho * ux, rho * uy, rho * uz, E])
    return U, np.stack(A)


def primitives(U, B):
    rho = U[0]; u = U[1:4] / rho
    p = (G - 1) * (U[4] - 0.5 * rho * (u**2).sum(0) - 0.5 * (B**2).sum(0))
    return rho, u, np.maximum(p, 1e-8)


def rhs(U, A):
    B = np.stack(curl(*A))
    rho, u, p = primitives(U, B)
    W = np.concatenate([rho[None], u, p[None]])            # 5 primitive variables
    dU = np.zeros_like(U); smax = 0.0
    for d in range(3):
        WL = W + 0.5 * minmod(W - r(W, -1, d + 1), r(W, 1, d + 1) - W)          # left state at face i+1/2
        Wp = r(W, 1, d + 1)
        WR = Wp - 0.5 * minmod(Wp - W, r(W, 2, d + 1) - Wp)                       # right state at face i+1/2
        WL[0] = np.maximum(WL[0], 1e-8); WR[0] = np.maximum(WR[0], 1e-8)
        WL[4] = np.maximum(WL[4], 1e-8); WR[4] = np.maximum(WR[4], 1e-8)
        BL, BR = B, r(B, 1, d)                                                   # cell-centred B either side
        bn = 0.5 * (BL[d] + BR[d])

        def flux_and_state(Wf, Bf):
            rf, uf, pf = Wf[0], Wf[1:4], Wf[4]
            Bc = Bf.copy(); Bc[d] = bn
            b2 = (Bc**2).sum(0); pt = pf + 0.5 * b2
            E = pf / (G - 1) + 0.5 * rf * (uf**2).sum(0) + 0.5 * b2
            un = uf[d]; ub = (uf * Bc).sum(0)
            F = np.empty((5,) + rf.shape)
            F[0] = rf * un
            for j in range(3):
                F[1 + j] = rf * uf[j] * un - bn * Bc[j] + (pt if j == d else 0.0)
            F[4] = (E + pt) * un - bn * ub
            Us = np.concatenate([rf[None], rf * uf, E[None]])
            a2 = G * pf / rf; va2 = b2 / rf
            cf = np.sqrt(0.5 * (a2 + va2 + np.sqrt(np.maximum((a2 + va2) ** 2 - 4 * a2 * bn**2 / rf, 0))))
            return F, Us, np.abs(un) + cf

        FL, UL, sL = flux_and_state(WL, BL)
        FR, UR, sR = flux_and_state(WR, BR)
        s = np.maximum(sL, sR); smax = max(smax, float(s.max()))
        F = 0.5 * (FL + FR) - 0.5 * s * (UR - UL)
        dU -= (F - r(F, -1, d + 1)) / h
    # vector potential, advective gauge: dA_i/dt = -u_j d_j A_i - A_j d_i u_j
    dA = np.zeros_like(A)
    for i in range(3):
        for j in range(3):
            Ai = A[i]
            up = (3 * Ai - 4 * r(Ai, -1, j) + r(Ai, -2, j)) / (2 * h)
            dn = (-3 * Ai + 4 * r(Ai, 1, j) - r(Ai, 2, j)) / (2 * h)
            dA[i] -= u[j] * np.where(u[j] > 0, up, dn)
            dA[i] -= A[j] * (r(u[j], 1, i) - r(u[j], -1, i)) / (2 * h)
        # resistivity: dA/dt += eta lap A. The magnetic energy it removes reappears as heat, because the total
        # energy E is evolved conservatively and p is recovered as E - kinetic - magnetic.
        dA[i] += args.eta * sum(r(A[i], 1, ax) - 2 * A[i] + r(A[i], -1, ax) for ax in range(3)) / h**2
    return dU, dA, smax


def diagnostics(U, A):
    B = np.stack(curl(*A)); rho, u, p = primitives(U, B)
    Jx, Jy, Jz = curl(*B)
    d = lambda f, ax: (r(f, 1, ax) - r(f, -1, ax)) / (2 * h)
    divB = d(B[0], 0) + d(B[1], 1) + d(B[2], 2)
    dv = h**3
    return {"Ek": float(0.5 * (rho * (u**2).sum(0)).sum() * dv), "Em": float(0.5 * (B**2).sum() * dv),
            "Etot": float(U[4].sum() * dv), "Jmax": float(np.sqrt(Jx**2 + Jy**2 + Jz**2).max()),
            "J2": float((Jx**2 + Jy**2 + Jz**2).sum() * dv), "divB": float(np.abs(divB).max()),
            "rho_min": float(rho.min()), "rho_max": float(rho.max())}


def main():
    t0 = time.time()
    U, A = init(); t = 0.0; hist = []; frames = []; nstep = 0
    tframes = np.linspace(0, args.tend, args.frames + 1); fi = 0
    snaps = {}; tsnap = [t_ for t_ in (0.5, 1.0, 1.5, 2.0, 3.0) if t_ <= args.tend]
    while t < args.tend - 1e-12:
        if t >= tframes[fi] - 1e-9:
            B = np.stack(curl(*A)); J = np.stack(curl(*B)); Jm = np.sqrt((J**2).sum(0))
            frames.append((t, U[0][:, :, N // 4].copy(), Jm.max(axis=2), Jm[:, :, N // 4].copy())); fi += 1
        dU, dA, s = rhs(U, A)
        dt = min(args.cfl * h / s, 0.15 * h * h / max(args.eta, 1e-12), args.tend - t)
        for ts in tsnap:                                    # land exactly on snapshot times
            if t < ts < t + dt: dt = ts - t
        U1 = U + dt * dU; A1 = A + dt * dA
        dU2, dA2, _ = rhs(U1, A1)
        U = 0.5 * (U + U1 + dt * dU2); A = 0.5 * (A + A1 + dt * dA2)
        t += dt; nstep += 1
        if nstep % 10 == 0 or t >= args.tend - 1e-12:
            dg = diagnostics(U, A); dg["t"] = t; hist.append(dg)
        for ts in tsnap:
            if abs(t - ts) < 1e-9 and ts not in snaps:
                B = np.stack(curl(*A)); J = np.stack(curl(*B))
                snaps[ts] = (U[0].copy(), np.sqrt((J**2).sum(0)))
                dg = diagnostics(U, A)
                print(f"t = {t:4.2f}  step {nstep:4d}  rho [{dg['rho_min']:.2f}, {dg['rho_max']:.2f}]  max|J| {dg['Jmax']:.2f}"
                      f"  max|divB| {dg['divB']:.1e}  E_tot drift {dg['Etot'] / hist[0]['Etot'] - 1:+.1e}  ({time.time() - t0:.0f}s)")
    B = np.stack(curl(*A)); J = np.stack(curl(*B)); Jm = np.sqrt((J**2).sum(0))
    frames.append((t, U[0][:, :, N // 4].copy(), Jm.max(axis=2), Jm[:, :, N // 4].copy()))

    if args.check2d:
        spread = float(np.abs(U[0] - U[0][:, :, :1]).max())
        print(f"2D check (beta = 0): max variation of rho along z = {spread:.2e}")

    summary = {"n": N, "beta": args.beta, "tend": args.tend, "steps": nstep, "seconds": time.time() - t0,
               "hist": hist, "final": diagnostics(U, A)}
    tag = "" if args.beta != 0 else "_beta0"
    json.dump(summary, open(os.path.join(OUT, f"ot3d{tag}.json"), "w"))
    if args.check2d:
        return

    np.savez_compressed(os.path.join(OUT, "ot3d_frames.npz"), t=np.array([f[0] for f in frames]),
                        rho=np.array([f[1] for f in frames], dtype=np.float32),
                        jproj=np.array([f[2] for f in frames], dtype=np.float32),
                        jslice=np.array([f[3] for f in frames], dtype=np.float32))
    setup()
    import matplotlib.pyplot as plt
    ts_ = sorted(snaps)
    fig, axs = plt.subplots(2, len(ts_), figsize=(3.2 * len(ts_), 6.4), squeeze=False)
    jmax = max(snaps[k][1].max(axis=2).max() for k in ts_)
    for c, ts in enumerate(ts_):
        rho3, J3 = snaps[ts]
        axs[0, c].imshow(rho3[:, :, N // 4].T, origin="lower", cmap="magma", vmin=1, vmax=6.3, extent=[0, 2 * np.pi] * 2)
        axs[0, c].set_title(f"density, plane z = π/2,  t = {ts:g}", fontsize=8.5, color=INK)
        jp = J3.max(axis=2)
        axs[1, c].imshow(jp.T, origin="lower", cmap="inferno", vmin=0, vmax=np.percentile(jp, 99.5), extent=[0, 2 * np.pi] * 2)
        axs[1, c].set_title(f"|J| projected through z,  t = {ts:g}", fontsize=8.5, color=INK)
        for a_ in axs[:, c]: a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    fig.savefig(os.path.join(OUT, "ot3d_snapshots.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    H = {k: np.array([hh[k] for hh in hist]) for k in hist[0]}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    ax[0].plot(H["t"], H["Ek"], color=BLUES[1], lw=2, label="kinetic")
    ax[0].plot(H["t"], H["Em"], color=BLUES[2], lw=2, ls="--", label="magnetic")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("energy"); ax[0].legend(fontsize=8); ax[0].set_title("energy exchange", fontsize=9.5, color=INK)
    ax[1].plot(H["t"], H["J2"], color=BLUES[2], lw=2); ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\int |J|^2\,dV$")
    ax[1].set_title("current sheets form and steepen", fontsize=9.5, color=INK)
    ax[2].semilogy(H["t"], np.maximum(H["divB"], 1e-17), color=BLUES[1], lw=2, label=r"max $|\nabla\cdot B|$")
    ax[2].semilogy(H["t"], np.maximum(np.abs(H["Etot"] / H["Etot"][0] - 1), 1e-17), color=MUTED, lw=2, ls=":", label="|total-energy drift|")
    ax[2].set_xlabel("t"); ax[2].legend(fontsize=8); ax[2].set_title("constraints: div B and energy", fontsize=9.5, color=INK)
    save(fig, "ot3d_diagnostics", OUT)
    print(f"done: {nstep} steps in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
