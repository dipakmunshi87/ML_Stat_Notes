"""Field-level EFT in two dimensions: particle-mesh simulation vs Zel'dovich / 2LPT forward models + counterterm.

Section 12 ("Effective field theory") of "Kinetic Hierarchies: from Liouville to Euler".

Cold dark matter in a periodic 2D box, Einstein-de Sitter background, scale factor a as the clock (D = a):
    d2x/da2 + (3/2a) dx/da = -(3/2a^2) grad phi,      laplacian(phi) = delta,
solved with a particle-mesh code (CIC, FFT Poisson, kick-drift-kick in log a). From the SAME initial modes we build
two perturbative forward models,
    Zel'dovich:  x = q - D grad phi1,                    laplacian(phi1) = delta_lin(q)
    2LPT:        x = q - D grad phi1 + D2 grad phi2,     D2 = -(3/7) D^2,
                 laplacian(phi2) = phi1_xx phi1_yy - phi1_xy^2,
and fit the leading EFT counterterm at the FIELD level, mode by mode with phases,
    delta_sim(k) = delta_fwd(k) - c k^2 delta_lin(k) + eps(k),       |k| < k_fit,
and, for comparison, at the n-point level from the power spectra of each box. Diagnostics: the error power
P_err = <|delta_sim - delta_model|^2> relative to P_sim (the standard field-level figure of merit), the fraction of
the residual that is deterministic, and the box-to-box scatter of the two estimates of c.

Run:  python eft_2d_field_level.py          (about 1-2 minutes; --nbox, --ng to change)
Writes ../figures/eft2d_maps.png, eft2d_error_power.{png,svg}, eft2d_estimates.{png,svg}, eft2d.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pinn"))
from _plotstyle import setup, save, BLUES, INK, MUTED

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
ap = argparse.ArgumentParser()
ap.add_argument("--ng", type=int, default=256, help="particles per side = mesh cells per side")
ap.add_argument("--nbox", type=int, default=8, help="independent boxes")
ap.add_argument("--amp", type=float, default=10.0, help="amplitude of the linear spectrum at a = 1 (rms delta_lin ~ 1.1)")
ap.add_argument("--kfit", type=float, default=0.12, help="largest |k| used in the fits (k_Nyquist = pi, k_fundamental = 2 pi / ng)")
args = ap.parse_args()

NG = args.ng
L = float(NG)                         # one length unit per mean inter-particle spacing; k_Nyquist = pi
A_START, A_END, NSTEP = 0.02, 1.0, 120
KC = 1.0                              # small-scale cut-off of the initial spectrum


def kgrid():
    k1 = 2 * np.pi * np.fft.fftfreq(NG, d=L / NG)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    return kx, ky, np.hypot(kx, ky)


KX, KY, KK = kgrid()
K2 = KK ** 2; K2s = K2.copy(); K2s[0, 0] = 1.0
LAMBDA = 0.25                                  # EFT cut-off: counterterms act only below it
WL = np.exp(-(KK / LAMBDA) ** 2)               # smooth filter W_Lambda(k)
K2W = K2 * WL                                  # the counterterm operator  c k^2 W_Lambda delta_lin


def p_lin(k):
    # nearly scale-invariant in 2D with a Gaussian cut-off: structure forms and shell-crosses at k ~ 0.3-1
    return args.amp * np.where(k > 0, k, 1.0) ** -1.0 * np.exp(-(k / KC) ** 2) * (k > 0)


def linear_field(rng):
    """Gaussian delta_lin on the grid (Fourier), normalised so <|delta_k|^2> = P(k) L^2 / (cell area)^2 conventions
    below cancel in every ratio we report."""
    w = rng.standard_normal((NG, NG))
    dk = np.fft.fft2(w) * np.sqrt(p_lin(KK) / (L / NG) ** 2)
    dk[0, 0] = 0
    return dk


def grad_from_k(fk):
    return np.real(np.fft.ifft2(1j * KX * fk)), np.real(np.fft.ifft2(1j * KY * fk))


def lpt_positions(dk, D, order):
    q = (np.arange(NG) + 0.5)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    phi1k = -dk / K2s; phi1k[0, 0] = 0                      # lap phi1 = delta_lin
    gx, gy = grad_from_k(phi1k)
    x, y = qx - D * gx, qy - D * gy
    if order >= 2:
        pxx = np.real(np.fft.ifft2(-KX * KX * phi1k)); pyy = np.real(np.fft.ifft2(-KY * KY * phi1k))
        pxy = np.real(np.fft.ifft2(-KX * KY * phi1k))
        src = np.fft.fft2(pxx * pyy - pxy ** 2)
        phi2k = -src / K2s; phi2k[0, 0] = 0
        hx, hy = grad_from_k(phi2k)
        D2 = -3.0 / 7.0 * D ** 2
        x, y = x + D2 * hx, y + D2 * hy
    return x, y


def cic_density(x, y):
    xs, ys = np.mod(x, L), np.mod(y, L)
    i, j = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - i, ys - j
    rho = np.zeros((NG, NG))
    for di, wx in ((0, 1 - fx), (1, fx)):
        for dj, wy in ((0, 1 - fy), (1, fy)):
            np.add.at(rho, ((i + di) % NG, (j + dj) % NG), wx * wy)
    return rho / rho.mean() - 1


def cic_interp(fx_, x, y):
    xs, ys = np.mod(x, L), np.mod(y, L)
    i, j = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - i, ys - j
    out = 0
    for di, wx in ((0, 1 - fx), (1, fx)):
        for dj, wy in ((0, 1 - fy), (1, fy)):
            out = out + wx * wy * fx_[(i + di) % NG, (j + dj) % NG]
    return out


def deconvolve(dk):
    w = (np.sinc(KX / (2 * np.pi)) * np.sinc(KY / (2 * np.pi))) ** 2   # CIC window on unit cells
    return dk / w


def pm_run(dk):
    """Particle-mesh with momentum p = a^{3/2} dx/da, KDK leapfrog in log a; Zel'dovich initial conditions."""
    x, y = lpt_positions(dk, A_START, 1)
    phi1k = -dk / K2s; phi1k[0, 0] = 0
    gx, gy = grad_from_k(phi1k)                             # growing mode: dx/da = -grad phi1
    px, py = -A_START ** 1.5 * gx, -A_START ** 1.5 * gy
    kern = -1.0 / K2s; kern[0, 0] = 0
    sx = 1j * KX * kern; sy = 1j * KY * kern
    wcic = (np.sinc(KX / (2 * np.pi)) * np.sinc(KY / (2 * np.pi))) ** 2

    def accel(x, y, a):
        # grad phi on the mesh, with the CIC window deconvolved once for the deposit and once for the interpolation
        dlt = np.fft.fft2(cic_density(x, y)) / wcic ** 2
        gphx = np.real(np.fft.ifft2(sx * dlt)); gphy = np.real(np.fft.ifft2(sy * dlt))
        # dp/da = a^{3/2} * (-(3/2a^2) grad phi) = -(3/2) a^{-1/2} grad phi
        return -1.5 * a ** -0.5 * cic_interp(gphx, x, y), -1.5 * a ** -0.5 * cic_interp(gphy, x, y)

    la = np.linspace(np.log(A_START), np.log(A_END), NSTEP + 1)
    ax_, ay_ = accel(x, y, A_START)
    for s in range(NSTEP):
        a0, a1 = np.exp(la[s]), np.exp(la[s + 1]); am = np.sqrt(a0 * a1); da = a1 - a0
        px += 0.5 * da * ax_; py += 0.5 * da * ay_
        x += da * px * am ** -1.5; y += da * py * am ** -1.5
        ax_, ay_ = accel(x, y, a1)
        px += 0.5 * da * ax_; py += 0.5 * da * ay_
    return x, y


def fit_c(dsim, dfwd, dlin, mask):
    X = -K2W[mask] * dlin[mask]; y = dsim[mask] - dfwd[mask]
    c = float(np.real(np.vdot(X, y)) / np.vdot(X, X).real)
    return c, X, y


def npoint_c(dsim, dfwd, dlin, mask):
    X = -2 * K2W[mask] * np.abs(dlin[mask]) ** 2; y = np.abs(dsim[mask]) ** 2 - np.abs(dfwd[mask]) ** 2
    return float(np.sum(X * y) / np.sum(X * X))


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    mask = (KK > 0) & (KK < args.kfit)
    kb = np.linspace(0.02, 1.0, 26); kc_ = 0.5 * (kb[1:] + kb[:-1])
    binned = lambda f: np.array([f[(KK >= kb[i]) & (KK < kb[i + 1])].mean() for i in range(len(kb) - 1)])
    res = {m: {"fl": [], "np": [], "perr": [], "perr_ct": [], "Xs": [], "ys": []} for m in ("zel", "2lpt")}
    psim_all, maps = [], None
    store = {"zel": [], "2lpt": [], "sim": [], "lin": []}
    for b in range(args.nbox):
        dk = linear_field(rng)
        xs, ys = pm_run(dk)
        dsim = deconvolve(np.fft.fft2(cic_density(xs, ys)))
        psim_all.append(binned(np.abs(dsim) ** 2)); store["sim"].append(dsim); store["lin"].append(dk * A_END)
        dlin = dk * A_END                                    # linear field today (D = a)
        for m, order in (("zel", 1), ("2lpt", 2)):
            xf, yf = lpt_positions(dk, A_END, order)
            dfwd = deconvolve(np.fft.fft2(cic_density(xf, yf)))
            c, X, y = fit_c(dsim, dfwd, dlin, mask)
            res[m]["fl"].append(c); res[m]["np"].append(npoint_c(dsim, dfwd, dlin, mask))
            res[m]["Xs"].append(X); res[m]["ys"].append(y)
            res[m]["perr"].append(binned(np.abs(dsim - dfwd) ** 2))
            res[m]["dfwd_last"] = dfwd
            store[m].append(dfwd)
        if b == 0:
            maps = (dsim, res["zel"]["dfwd_last"], res["2lpt"]["dfwd_last"], dlin)
        print(f"box {b + 1}/{args.nbox}  ({time.time() - t0:.0f}s)")

    psim = np.mean(psim_all, 0); summary = {"kfit": args.kfit, "nbox": args.nbox, "ng": NG}
    for m in res:
        X = np.concatenate(res[m]["Xs"]); y = np.concatenate(res[m]["ys"])
        c = float(np.real(np.vdot(X, y)) / np.vdot(X, X).real)
        frac = float(np.real(np.vdot(c * X, c * X)) / np.real(np.vdot(y, y)))   # share of residual power explained
        summary[m] = {"c_pooled": c, "deterministic_fraction": frac,
                      "fl_mean": float(np.mean(res[m]["fl"])), "fl_std": float(np.std(res[m]["fl"], ddof=1)),
                      "np_mean": float(np.mean(res[m]["np"])), "np_std": float(np.std(res[m]["np"], ddof=1))}
        print(f"{m:5s}: pooled field-level c = {c:.3f}; deterministic share of residual (k<{args.kfit}) = {100*frac:.0f}%;"
              f"  per box: field {np.mean(res[m]['fl']):.3f}+-{np.std(res[m]['fl'], ddof=1):.3f},"
              f" n-point {np.mean(res[m]['np']):.3f}+-{np.std(res[m]['np'], ddof=1):.3f}")

    # use stored residuals instead of a second simulation pass: |y - cX|^2 per mode, binned
    for m in res:
        c = summary[m]["c_pooled"]
        summary[m]["perr_over_psim"] = (np.mean(res[m]["perr"], 0) / psim).tolist()
        summary[m]["perr_ct_over_psim"] = (np.mean([binned(np.abs(ds - df + c * K2W * dl) ** 2) for ds, df, dl in
                                                    zip(store["sim"], store[m], store["lin"])], 0) / psim).tolist()
    summary["k"] = kc_.tolist(); summary["psim"] = psim.tolist()
    # ratio of residual power with/without counterterm inside the fit range
    for m in res:
        X = np.concatenate(res[m]["Xs"]); y = np.concatenate(res[m]["ys"]); c = summary[m]["c_pooled"]
        summary[m]["ct_power_reduction"] = float(np.sum(np.abs(y - c * X) ** 2) / np.sum(np.abs(y) ** 2))

    setup()
    import matplotlib.pyplot as plt
    # --- maps: smoothed fields of box 0 and the residuals before/after the counterterm
    dsim, dzel, d2l, dlin = maps
    sm = np.exp(-0.5 * (KK / 0.35) ** 2)
    R = lambda f: np.real(np.fft.ifft2(f * sm))
    c2 = summary["2lpt"]["c_pooled"]
    smL = np.exp(-0.5 * (KK / 0.15) ** 2)                 # residuals shown on the large scales the EFT describes
    RL = lambda f: np.real(np.fft.ifft2(f * smL))
    panels = [(R(dsim), "simulation (particle-mesh)"), (R(dzel), "Zel'dovich forward model"), (R(d2l), "2LPT forward model"),
              (RL(dsim - d2l), "large-scale residual: sim \u2212 2LPT"), (RL(dsim - d2l + c2 * K2W * dlin), "\u2026 after adding the counterterm")]
    vmax = np.percentile(np.abs(panels[0][0]), 99)
    rmax = np.percentile(np.abs(panels[3][0]), 99.5)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
    for a_, (f, title), lim in zip(axs, panels, [vmax, vmax, vmax, rmax, rmax]):
        a_.imshow(f.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a_.set_title(title, fontsize=9, color=INK); a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    fig.savefig(os.path.join(OUT, "eft2d_maps.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- error power
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m, col, lab in (("zel", BLUES[0], "Zel'dovich"), ("2lpt", BLUES[2], "2LPT")):
        ax.plot(kc_, summary[m]["perr_over_psim"], color=col, lw=2, label=f"{lab}")
        ax.plot(kc_, summary[m]["perr_ct_over_psim"], color=col, lw=2, ls="--", label=f"{lab} + counterterm")
    ax.axvspan(0, args.kfit, color="#eef3f9", zorder=0)
    ax.set_yscale("log"); ax.set_xlabel("wavenumber |k|  (k_Nyquist = π)"); ax.set_ylabel(r"$P_{\rm err}(k)\,/\,P_{\rm sim}(k)$")
    ax.set_title("field-level error power: what the forward model gets wrong", fontsize=9.5, color=INK)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1.0)
    save(fig, "eft2d_error_power", OUT)

    # --- estimates per box
    fig, ax = plt.subplots(figsize=(7.0, 3.0)); rr = np.random.default_rng(0); row = 0; ylab = []
    for m, lab in (("zel", "Zel'dovich"), ("2lpt", "2LPT")):
        for kind, nm, col in (("np", "n-point", BLUES[0]), ("fl", "field level", BLUES[2])):
            e = np.array(res[m][kind])
            ax.plot(e, row + 0.1 * rr.standard_normal(e.size), "o", ms=5, mfc="white", mec=col)
            ax.errorbar(e.mean(), row, xerr=e.std(ddof=1), fmt="s", color=col, ms=6, capsize=4, lw=2)
            ylab.append(f"{lab}, {nm}:  {e.mean():.2f} ± {e.std(ddof=1):.2f}"); row += 1
    ax.set_yticks(range(row)); ax.set_yticklabels(ylab, fontsize=8); ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"counterterm $c$ estimated from one box"); ax.set_title(f"{args.nbox} boxes, same modes |k| < {args.kfit}", fontsize=9.5, color=INK)
    save(fig, "eft2d_estimates", OUT)
    json.dump(summary, open(os.path.join(OUT, "eft2d.json"), "w"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
