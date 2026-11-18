"""EFT of large-scale structure in one dimension: Zel'dovich + a k^2 counterterm vs an exact simulation.

Section 7 (cosmology, the EFT subsection) of "Kinetic Hierarchies: from Liouville to Euler".

In one dimension cold dark matter is a set of parallel sheets, and gravity between sheets is exact and simple:
the comoving acceleration of a sheet depends only on how much mass lies to its left. In an Einstein-de Sitter
universe with the scale factor a as time variable (growing mode D = a),
    d2x/da2 + (3/2a) dx/da = (3/2a^2) [x - q_rank(x)],
where q_rank(x) is the Lagrangian position the sheet would have if the sheets were unmixed (its rank times the
sheet spacing). Before any sheets cross, q_rank = q and the Zel'dovich solution x = q + a psi0(q) is EXACT.
After crossing (multi-streaming) it is not: that is where the pressureless single-stream closure fails and
where the EFT counterterm comes in. We evolve the exact sheet dynamics and, from the SAME initial conditions,
the Zel'dovich particles, then fit
    P_sim(k) ~= P_Zel(k) - 2 c_s^2 k^2 P_lin(k)        for k well below the nonlinear scale,
which is the leading EFT correction on top of Zel'dovich (cf. McQuinn & White 2016). Using the same
realisation for all three makes cosmic variance cancel in the comparison.

Run:  python eft_1d.py        (under a minute)
Writes ../figures/eft_1d_power.{png,svg}, ../figures/eft_1d_phase_space.png and ../figures/eft_1d.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pinn"))
from _plotstyle import setup, save, BLUES, INK, MUTED   # shared figure style

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
L, N = 2 * np.pi * 64, 2**16          # box (units with k_fundamental = 1/64) and number of sheets
NREAL = 24                           # realisations averaged to beat down noise
AMP = 30.0                            # amplitude of the linear spectrum at a = 1
KC = 0.5
A_START, A_END = 0.02, 1.0
KFIT = 0.12                           # fit only well below the nonlinear scale


def linear_spectrum(k, kc=KC):
    # blue at small k (little large-scale bulk flow), cut off at kc: structures form and shell-cross at k ~ kc,
    # while k << kc stays perturbative, which is where the EFT expansion in k/k_NL should work
    return AMP * k * np.exp(-(k / kc) ** 2)


def initial_displacement(rng):
    q = (np.arange(N) + 0.5) * L / N
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    amp = np.zeros_like(k); amp[1:] = np.sqrt(linear_spectrum(k[1:]) * L / 2)
    dk = amp * (rng.standard_normal(k.size) + 1j * rng.standard_normal(k.size))
    dk[0] = 0
    # delta = -d psi/dq  ->  psi_k = i delta_k / k
    psik = np.zeros_like(dk); psik[1:] = 1j * dk[1:] / k[1:]
    psi = np.fft.irfft(psik, n=N) * N / L
    delta = np.fft.irfft(dk, n=N) * N / L
    return q, psi, delta


def density_field(x, ng=2**14):
    """Fourier modes of the sheet density (CIC, window deconvolved)."""
    xs = np.mod(x, L) / (L / ng)
    i = np.floor(xs).astype(int); f = xs - i
    rho = np.bincount(i % ng, 1 - f, ng) + np.bincount((i + 1) % ng, f, ng)
    dk = np.fft.rfft(rho / rho.mean() - 1) * (L / ng)
    k = 2 * np.pi * np.fft.rfftfreq(ng, d=L / ng)
    dk[1:] /= np.sinc(k[1:] * (L / ng) / (2 * np.pi)) ** 2
    return k[1:], dk[1:]


def density_power(x, ng=2**14):
    """Power spectrum of the density of sheets at positions x (CIC on ng cells, CIC window deconvolved)."""
    xs = np.mod(x, L) / (L / ng)
    i = np.floor(xs).astype(int); f = xs - i
    rho = np.bincount(i % ng, 1 - f, ng) + np.bincount((i + 1) % ng, f, ng)
    delta = rho / rho.mean() - 1
    dk = np.fft.rfft(delta) * (L / ng)
    k = 2 * np.pi * np.fft.rfftfreq(ng, d=L / ng)
    w = np.sinc(k * (L / ng) / (2 * np.pi)) ** 2
    P = np.abs(dk) ** 2 / L
    P[1:] /= w[1:] ** 2
    return k[1:], P[1:]


def evolve_exact(q, psi, nsteps=1500):
    """Exact 1D sheet dynamics with p = a^{3/2} dx/da, leapfrog in log a."""
    loga = np.linspace(np.log(A_START), np.log(A_END), nsteps + 1)
    x = q + A_START * psi
    p = A_START ** 1.5 * psi                            # growing mode: dx/da = psi
    qgrid = (np.arange(N) + 0.5) * L / N

    def accel_term(x, a):                              # dp/da = (3/2) a^{-1/2} (x - q_rank)
        order = np.argsort(x, kind="stable")
        xr = x[order]
        g = np.empty_like(x)
        # unwrap: mean-subtract so that the periodic box has no net force
        resid = xr - qgrid
        resid -= resid.mean()
        g[order] = resid
        return 1.5 * a ** -0.5 * g

    for s in range(nsteps):
        a0, a1 = np.exp(loga[s]), np.exp(loga[s + 1]); am = np.sqrt(a0 * a1); da = a1 - a0
        p += 0.5 * da * accel_term(x, a0)
        x += da * p * am ** -1.5
        p += 0.5 * da * accel_term(x, a1)
    return x, p


def main():
    rng = np.random.default_rng(42)
    Ps, Pz, Pl, snap = [], [], [], None
    est_np, est_fl, pool_X, pool_y = [], [], [], []
    for r in range(NREAL):
        q, psi, _ = initial_displacement(rng)
        xs, ps = evolve_exact(q, psi)
        xz = q + A_END * psi                                # Zel'dovich from the same ICs
        k, P_sim = density_power(xs); _, P_zel = density_power(xz)
        # linear theory from the same realisation: delta_lin = -a dpsi/dq
        _, P_lin = density_power(q + 1e-3 * psi); P_lin = P_lin * (A_END / 1e-3) ** 2
        Ps.append(P_sim); Pz.append(P_zel); Pl.append(P_lin)
        # --- two estimates of c_s^2 from THIS realisation, using modes k < KFIT only
        kk, ds = density_field(xs); _, dz = density_field(xz); _, dl = density_field(q + 1e-3 * psi); dl *= A_END / 1e-3
        mm = kk < KFIT
        # n-point: P_sim - P_zel = -2 c k^2 P_lin, fitted to the measured (noisy) spectra
        X = -2 * kk[mm] ** 2 * np.abs(dl[mm]) ** 2 / L; y = (np.abs(ds[mm]) ** 2 - np.abs(dz[mm]) ** 2) / L
        est_np.append(float(np.sum(X * y) / np.sum(X * X)))
        # field level: delta_sim(k) - delta_zel(k) = -c k^2 delta_lin(k), mode by mode, phases included
        Xf = -kk[mm] ** 2 * dl[mm]; yf = ds[mm] - dz[mm]
        est_fl.append(float(np.real(np.sum(np.conj(Xf) * yf)) / np.sum(np.abs(Xf) ** 2)))
        pool_X.append(Xf); pool_y.append(yf)
        if r == 0:
            snap = (q, xs, ps * A_END ** -1.5, xz, psi)
        print(f"realisation {r + 1}/{NREAL} done")
    P_sim, P_zel, P_lin = (np.mean(a, 0) for a in (Ps, Pz, Pl))
    # bin in log k
    edges = np.geomspace(k[0], 3.0, 40)
    idx = np.digitize(k, edges)
    kb = np.array([k[idx == i].mean() for i in range(1, len(edges)) if np.any(idx == i)])
    B = lambda P: np.array([P[idx == i].mean() for i in range(1, len(edges)) if np.any(idx == i)])
    Psb, Pzb, Plb = B(P_sim), B(P_zel), B(P_lin)

    # fit the counterterm on large scales only:  P_sim - P_zel = -2 cs2 k^2 P_lin
    kfit = KFIT
    m = kb < kfit
    y = Psb[m] - Pzb[m]; X = -2 * kb[m] ** 2 * Plb[m]
    cs2 = float(np.sum(X * y) / np.sum(X * X))
    P_eft = Pzb - 2 * cs2 * kb ** 2 * Plb
    err = lambda P: float(np.sqrt(np.mean((P[m] / Psb[m] - 1) ** 2)))
    kmid = (kb > 0.1) & (kb < 0.4)
    err_mid = lambda P: float(np.sqrt(np.mean((P[kmid] / Psb[kmid] - 1) ** 2)))
    print(f"c_s^2 = {cs2:.4f};  rms error for k<{kfit}: Zel'dovich {err(Pzb):.3f}, Zel'dovich+EFT {err(P_eft):.4f};"
          f"  for 0.1<k<0.4: Zel {err_mid(Pzb):.3f}, EFT {err_mid(P_eft):.3f}")

    setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax[0].axhline(1, color="#d4dbe2", lw=2, label="linear theory")
    ax[0].plot(kb, Pzb / Plb, color=BLUES[0], lw=2, label="Zel'dovich (single stream)")
    ax[0].plot(kb, P_eft / Plb, color=BLUES[2], lw=2, ls="--", label="Zel'dovich + EFT counterterm")
    ax[0].plot(kb, Psb / Plb, "o", ms=4, mfc="white", mec=INK, label="exact simulation")
    ax[0].axvspan(kb[0] * 0.9, kfit, color="#eef3f9", zorder=0)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.3); ax[0].set_xlabel("wavenumber k"); ax[0].set_ylabel(r"$P(k)\,/\,P_{\rm lin}(k)$")
    ax[0].set_title("power spectrum at a = 1 (shaded: fit range)", fontsize=9.5, color=INK); ax[0].legend(fontsize=7.5, loc="lower left")
    r = (Psb - Pzb) / Plb
    ax[1].plot(kb, r, "o", ms=4, mfc="white", mec=INK, label="exact minus Zel'dovich")
    kk = np.linspace(0, kb.max(), 200)
    ax[1].plot(kk, -2 * cs2 * kk**2, color=BLUES[2], lw=2, ls="--", label=rf"$-2c_s^2k^2$,  $c_s^2$ = {cs2:.2f}")
    ax[1].axvspan(0, kfit, color="#eef3f9", zorder=0); ax[1].axhline(0, color="#d4dbe2", lw=1.5)
    ax[1].set_xlim(0, 0.45); ax[1].set_ylim(-0.1, 0.4)
    ax[1].set_xlabel("wavenumber k"); ax[1].set_ylabel(r"$(P_{\rm sim}-P_{\rm Zel})\,/\,P_{\rm lin}$")
    ax[1].set_title(r"what the closure misses: a $k^2$ term, until $k \sim k_{\rm NL}$", fontsize=9.5, color=INK); ax[1].legend(fontsize=7.5, loc="upper left")
    save(fig, "eft_1d_power", OUT)

    q, xs, vs, xz, psi = snap
    i0 = int(np.argmax(np.abs(xs - xz)))
    sel = np.abs(q - q[i0]) < 0.06 * L
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    ax[0].plot(xz[sel], psi[sel], ".", ms=1, color=BLUES[0]); ax[0].set_title("Zel'dovich: streams pass straight through", fontsize=9, color=INK)
    ax[1].plot(xs[sel], vs[sel], ".", ms=1, color=BLUES[2]); ax[1].set_title("exact: streams fold and wind up into a halo", fontsize=9, color=INK)
    for a_ in ax: a_.set_xlabel("x (comoving)")
    ax[0].set_ylabel("velocity  dx/da")
    fig.savefig(os.path.join(OUT, "eft_1d_phase_space.png"), dpi=150); plt.close(fig)
    en, ef = np.array(est_np), np.array(est_fl)
    PX, Py = np.concatenate(pool_X), np.concatenate(pool_y)
    c_det = float(np.real(np.sum(np.conj(PX) * Py)) / np.sum(np.abs(PX) ** 2))
    corr = float(abs(np.real(np.sum(np.conj(PX) * Py))) / np.sqrt(np.sum(np.abs(PX) ** 2) * np.sum(np.abs(Py) ** 2)))
    print(f"field level, all realisations pooled: deterministic c_s^2 = {c_det:.3f}; |corr(residual, k^2 delta_lin)| = {corr:.2f}"
          f" -> {100 * corr**2:.0f}% of the mode-by-mode residual is the deterministic counterterm, the rest stochastic")
    print(f"per-realisation c_s^2: n-point {en.mean():.3f} +- {en.std(ddof=1):.3f}   field-level {ef.mean():.3f} +- {ef.std(ddof=1):.3f}")
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    rng2 = np.random.default_rng(0)
    for row, (e, lab, col) in enumerate([(en, "n-point fit (power spectrum)", BLUES[0]), (ef, "field-level fit (modes with phases)", BLUES[2])]):
        ax.plot(e, row + 0.12 * rng2.standard_normal(e.size), "o", ms=5, mfc="white", mec=col)
        ax.errorbar(e.mean(), row, xerr=e.std(ddof=1), fmt="s", color=col, ms=6, capsize=4, lw=2)
        ax.text(e.mean(), row + 0.32, f"{lab}:  {e.mean():.2f} \u00b1 {e.std(ddof=1):.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_yticks([]); ax.set_ylim(-0.6, 1.7); ax.set_xlabel(r"$c_s^2$ estimated from one realisation  (24 realisations)")
    ax.axvline(cs2, color=MUTED, ls="--", lw=1)
    ax.set_title("same simulations, same modes: two ways to measure the counterterm", fontsize=9.5, color=INK)
    save(fig, "eft_1d_estimates", OUT)
    json.dump({"cs2_np_mean": float(en.mean()), "cs2_np_std": float(en.std(ddof=1)), "cs2_fl_mean": float(ef.mean()),
               "cs2_fl_std": float(ef.std(ddof=1)), "nreal": NREAL,
               "cs2_det_pooled": c_det, "corr": corr}, open(os.path.join(OUT, "eft_1d_estimates.json"), "w"))
    json.dump({"cs2": cs2, "kfit": kfit, "k": kb.tolist(), "P_sim": Psb.tolist(), "P_zel": Pzb.tolist(),
               "P_lin": Plb.tolist(), "P_eft": P_eft.tolist(), "err_zel": err(Pzb), "err_eft": err(P_eft),
               "err_mid_zel": err_mid(Pzb), "err_mid_eft": err_mid(P_eft)},
              open(os.path.join(OUT, "eft_1d.json"), "w"))


if __name__ == "__main__":
    main()
