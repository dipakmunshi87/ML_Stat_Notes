from lib import *
from scipy import stats, optimize
rng = np.random.default_rng(1)

# ---------------- Ch1 discounting & bonds ----------------
r, T = 0.05, 10
V("c1_annual", 100*(1+r)**T, ".2f"); V("c1_monthly", 100*(1+r/12)**(12*T), ".2f"); V("c1_cont", 100*np.exp(r*T), ".2f")
V("c1_daily", 100*(1+r/365)**(365*T), ".2f")
# bond
c, F, n, y = 0.04, 100, 5, 0.05
t = np.arange(1, n+1); cf = np.full(n, c*F); cf[-1] += F
df = (1+y)**-t
P = np.sum(cf*df); V("c1_P", P, ".3f")
D_mac = np.sum(t*cf*df)/P; D_mod = D_mac/(1+y); C = np.sum(t*(t+1)*cf*df)/(P*(1+y)**2)
V("c1_Dmac", D_mac, ".3f"); V("c1_Dmod", D_mod, ".3f"); V("c1_C", C, ".2f")
dy = 0.01
P2 = np.sum(cf*(1+y+dy)**-t); V("c1_P2", P2, ".3f"); V("c1_dP_exact", P2-P, ".3f")
V("c1_dP_dur", -D_mod*P*dy, ".3f"); V("c1_dP_durconv", -D_mod*P*dy + 0.5*C*P*dy**2, ".3f")
V("c1_pv_cf", " + ".join(f"{x:.2f}" for x in cf*df))
# zero-coupon table and par
# figure: price-yield with tangent + convexity approx
ys = np.linspace(0.0, 0.12, 200)
Py = np.array([np.sum(cf*(1+yy)**-t) for yy in ys])
f_, ax = fig(8.4, 4.0)
ax.plot(ys*100, Py, color=BLUE, label="exact price")
ax.plot(ys*100, P - D_mod*P*(ys-y), color=ORANGE, lw=1.6, ls="--", label="duration (tangent)")
ax.plot(ys*100, P - D_mod*P*(ys-y) + 0.5*C*P*(ys-y)**2, color=AQUA, lw=1.6, ls=":", label="duration + convexity")
ax.plot([y*100], [P], "o", color=INK, ms=7); ax.annotate(f"y = 5%, P = {P:.2f}", (y*100, P), xytext=(6.2, P+6), color=INK)
ax.set_xlabel("yield (%)"); ax.set_ylabel("price"); ax.set_title("A 5-year 4% bond: price vs yield, and its Taylor approximations"); ax.legend()
save(f_, "c1_priceyield")
# Nelson-Siegel curve + forwards
b0, b1, b2, tau = 0.045, -0.02, 0.01, 2.0
tt = np.linspace(0.05, 30, 400)
z = b0 + b1*(1-np.exp(-tt/tau))/(tt/tau) + b2*((1-np.exp(-tt/tau))/(tt/tau) - np.exp(-tt/tau))
fwd = np.gradient(z*tt, tt)   # instantaneous forward f = d(zT)/dT
f_, ax = fig(8.4, 3.8)
ax.plot(tt, z*100, color=BLUE, label="zero (spot) rate z(T)")
ax.plot(tt, fwd*100, color=ORANGE, label="instantaneous forward f(T)")
ax.set_xlabel("maturity T (years)"); ax.set_ylabel("rate (%)"); ax.set_title("A Nelson–Siegel yield curve and the forward curve it implies"); ax.legend()
save(f_, "c1_curve")
V("c1_z2", z[np.argmin(abs(tt-2))]*100, ".2f"); V("c1_z10", z[np.argmin(abs(tt-10))]*100, ".2f"); V("c1_f10", fwd[np.argmin(abs(tt-10))]*100, ".2f")
# forward rate from two zeros
z1, z2_ = 0.03, 0.035
V("c1_f12", ((1+z2_)**2/(1+z1) - 1)*100, ".3f")

# ---------------- Ch2 probability & fat tails ----------------
N = 250*40  # 40 years of daily
sig = 0.01
g = rng.normal(0, sig, N)
nu = 4
tdist = rng.standard_t(nu, N) * sig*np.sqrt((nu-2)/nu)
V("c2_N", N); V("c2_kurt_g", stats.kurtosis(g), ".2f"); V("c2_kurt_t", stats.kurtosis(tdist), ".2f")
V("c2_kurt_t_theory", 6/(nu-4) if nu > 4 else "infinite")
V("c2_max_g", g.min()/sig, ".1f"); V("c2_max_t", tdist.min()/sig, ".1f")
V("c2_p5_g", 2*stats.norm.sf(5), ".2e"); V("c2_yrs5_g", 1/(250*2*stats.norm.sf(5)), ".0f")
scale = sig*np.sqrt((nu-2)/nu)
V("c2_p5_t", 2*stats.t.sf(5*sig/scale, nu), ".2e"); V("c2_yrs5_t", 1/(250*2*stats.t.sf(5*sig/scale, nu)), ".1f")
V("c2_n4_g", int((abs(g) > 4*sig).sum())); V("c2_n4_t", int((abs(tdist) > 4*sig).sum()))
f_, ax = fig(8.4, 4.0, ncols=2)
bins = np.linspace(-8, 8, 81)
for a, d, name, col in [(ax[0], g, "Gaussian", BLUE), (ax[0], tdist, "Student-t, ν=4", ORANGE)]:
    a.hist(d/sig, bins=bins, density=True, histtype="step", color=col, lw=1.8, label=name)
x = np.linspace(-8, 8, 400); ax[0].plot(x, stats.norm.pdf(x), color=INK, lw=1, alpha=.5)
ax[0].set_yscale("log"); ax[0].set_ylim(1e-5, 1); ax[0].set_xlabel("return / σ"); ax[0].set_title("Same variance, different tails (log scale)"); ax[0].legend()
stats.probplot(tdist/sig, dist="norm", plot=ax[1]); ax[1].get_lines()[0].set(color=ORANGE, marker=".", ms=3, alpha=.6); ax[1].get_lines()[1].set(color=INK, lw=1)
ax[1].set_title("QQ plot: t-returns vs normal"); ax[1].set_xlabel("normal quantile"); ax[1].set_ylabel("sample quantile (σ units)")
save(f_, "c2_tails")
# CLT: aggregate 20 days
agg = tdist[:N//20*20].reshape(-1, 20).sum(1); agg /= agg.std()
V("c2_kurt_20d", stats.kurtosis(agg), ".2f")
# log vs simple
V("c2_simple_ex", (np.exp(0.10)-1)*100, ".2f"); V("c2_log_ex", np.log(1-0.5), ".3f")
# volatility scaling
V("c2_ann_vol", sig*np.sqrt(250)*100, ".1f")
# VaR gaussian vs t at 99%
V("c2_var_g", -stats.norm.ppf(0.01)*sig*100, ".2f"); V("c2_var_t", -stats.t.ppf(0.01, nu)*scale*100, ".2f")
V("c2_var_t_999", -stats.t.ppf(0.001, nu)*scale*100, ".2f"); V("c2_var_g_999", -stats.norm.ppf(0.001)*sig*100, ".2f")
# lognormal moments
mu_, s_ = 0.08, 0.2
V("c2_logn_mean", np.exp(mu_ + s_**2/2), ".4f"); V("c2_logn_med", np.exp(mu_), ".4f")

# ---------------- Ch3 statistics ----------------
mu_a, sig_a = 0.08, 0.20
V("c3_T_needed", (2*sig_a/mu_a)**2, ".0f")
V("c3_se_1y", sig_a/np.sqrt(1)*100, ".0f"); V("c3_se_10y", sig_a/np.sqrt(10)*100, ".1f")
# variance estimation precision: daily data, 1 year
nd = 250
V("c3_se_vol_rel", 1/np.sqrt(2*nd)*100, ".1f")
# sharpe se
SR = 0.4
for Ty in [1, 3, 10]:
    V(f"c3_se_sr_{Ty}", np.sqrt((1+SR**2/2)/Ty), ".2f")
# simulate 3-year sample Sharpe distribution
M = 20000
daily_mu, daily_sig = mu_a/250, sig_a/np.sqrt(250)
rets = rng.normal(daily_mu, daily_sig, (M, 750))
sr_hat = rets.mean(1)/rets.std(1)*np.sqrt(250)
V("c3_sr_true", mu_a/sig_a, ".2f"); V("c3_sr_sd_sim", sr_hat.std(), ".2f"); V("c3_sr_frac_neg", (sr_hat < 0).mean()*100, ".1f")
V("c3_sr_frac_gt1", (sr_hat > 1).mean()*100, ".1f")
f_, ax = fig(8.4, 3.8)
ax.hist(sr_hat, bins=80, color=BLUE, alpha=.85, density=True)
ax.axvline(mu_a/sig_a, color=INK, lw=1.5); ax.text(mu_a/sig_a+0.03, ax.get_ylim()[1]*0.9, "true SR = 0.40", color=INK)
ax.axvline(0, color=ORANGE, lw=1.2, ls="--")
ax.set_xlabel("sample Sharpe ratio over 3 years"); ax.set_ylabel("density"); ax.set_title("Sampling distribution of a 3-year Sharpe ratio when the true value is 0.40")
save(f_, "c3_sharpe")
# MLE of t-dist nu on the simulated returns
nu_hat, loc_hat, sc_hat = stats.t.fit(tdist)
V("c3_nu_hat", nu_hat, ".2f"); V("c3_sc_hat", sc_hat/sig, ".3f")
# bootstrap CI for 99% VaR of t returns (first 3 years)
samp = tdist[:750]
B = 2000
boot = np.array([np.quantile(rng.choice(samp, 750), 0.01) for _ in range(B)])
V("c3_var_hat", -np.quantile(samp, 0.01)*100, ".2f"); V("c3_var_lo", -np.quantile(boot, 0.975)*100, ".2f"); V("c3_var_hi", -np.quantile(boot, 0.025)*100, ".2f")
# hypothesis test: strategy with mean 0.03%/day, sd 1%, 500 days
tstat = 0.0003/(0.01/np.sqrt(500)); V("c3_tstat", tstat, ".2f"); V("c3_pval", 2*stats.norm.sf(tstat), ".3f")
# Fisher information for Gaussian mean/var
V("c3_crlb_mu", sig_a**2/10, ".4f")

# ---------------- Ch4 linear algebra ----------------
vols = np.array([0.20, 0.25, 0.15]); corr = np.array([[1, .6, .2], [.6, 1, .3], [.2, .3, 1]])
Sigma = np.outer(vols, vols)*corr
w = np.array([0.5, 0.3, 0.2])
V("c4_Sigma", "; ".join(", ".join(f"{x:.4f}" for x in row) for row in Sigma))
V("c4_port_var", w@Sigma@w, ".5f"); V("c4_port_vol", np.sqrt(w@Sigma@w)*100, ".2f")
lam, Q = np.linalg.eigh(Sigma); lam, Q = lam[::-1], Q[:, ::-1]
V("c4_eig", ", ".join(f"{x:.4f}" for x in lam)); V("c4_eig_frac1", lam[0]/lam.sum()*100, ".1f")
V("c4_ev1", ", ".join(f"{x:.3f}" for x in Q[:, 0]*np.sign(Q[0, 0])))
L = np.linalg.cholesky(Sigma); V("c4_chol", "; ".join(", ".join(f"{x:.4f}" for x in row) for row in L))
Z = rng.normal(size=(100000, 3)); X = Z@L.T
V("c4_chol_check", "; ".join(", ".join(f"{x:.4f}" for x in row) for row in np.cov(X.T)))
# PCA on 30 assets, 1 market factor + sector
na, nt = 30, 1000
beta = rng.uniform(0.6, 1.4, na); fm = rng.normal(0, 0.01, nt)
sector = np.repeat(np.arange(3), 10); fs = rng.normal(0, 0.005, (nt, 3))
R = np.outer(fm, beta) + fs[:, sector] + rng.normal(0, 0.008, (nt, na))
C_ = np.corrcoef(R.T); ev, EV = np.linalg.eigh(C_); ev = ev[::-1]
V("c4_pca_ev1", ev[0], ".2f"); V("c4_pca_ev2", ev[1], ".2f"); V("c4_pca_ev3", ev[2], ".2f"); V("c4_pca_ev4", ev[3], ".2f"); V("c4_pca_frac3", ev[:3].sum()/na*100, ".0f")
f_, ax = fig(8.4, 3.8, ncols=2)
ax[0].bar(np.arange(1, na+1), ev, color=BLUE, width=0.75); ax[0].set_xlabel("component"); ax[0].set_ylabel("eigenvalue of correlation matrix"); ax[0].set_title("Eigenvalue spectrum (30 assets)")
ax[0].axhline(1, color=MUT, lw=1, ls="--"); ax[0].text(20, 1.2, "noise level ≈ 1", color=MUT)
v1 = EV[:, -1]*np.sign(EV[:, -1].sum()); v2 = EV[:, -2]
ax[1].bar(np.arange(na)-0.2, v1, width=0.4, color=BLUE, label="eigenvector 1 (market)"); ax[1].bar(np.arange(na)+0.2, v2, width=0.4, color=ORANGE, label="eigenvector 2 (sector)")
ax[1].set_xlabel("asset"); ax[1].set_title("Loadings of the first two eigenvectors"); ax[1].legend(loc="lower right")
save(f_, "c4_pca")
# least squares & condition number
A = np.column_stack([np.ones(nt), fm, fs[:, 0]]); yv = R[:, 0]
bhat = np.linalg.lstsq(A, yv, rcond=None)[0]; V("c4_ls", ", ".join(f"{x:.4f}" for x in bhat)); V("c4_ls_true", f"0, {beta[0]:.4f}, 1")
V("c4_cond", np.linalg.cond(Sigma), ".1f")

# ---------------- Ch5 calculus / PDE ----------------
# heat equation u_t = D u_xx, spike -> gaussian; explicit FD vs analytic
D = 1.0; L_ = 10; nx = 201; x = np.linspace(-L_/2, L_/2, nx); dx = x[1]-x[0]
dt = 0.4*dx**2/D; V("c5_dt", dt, ".5f"); V("c5_dx", dx, ".3f"); V("c5_alpha", D*dt/dx**2, ".2f")
u = np.exp(-x**2/(4*D*0.05))/np.sqrt(4*np.pi*D*0.05)
f_, ax = fig(8.4, 3.8)
times = [0.05, 0.3, 1.0, 2.5]; tcur = 0.05; cols = [BLUE, ORANGE, AQUA, YELLOW]
ax.plot(x, u, color=cols[0], label=f"t = {tcur}")
err = []
for k, tgt in enumerate(times[1:]):
    while tcur < tgt - 1e-12:
        u[1:-1] = u[1:-1] + D*dt/dx**2*(u[2:]-2*u[1:-1]+u[:-2]); tcur += dt
    ua = np.exp(-x**2/(4*D*tcur))/np.sqrt(4*np.pi*D*tcur)
    err.append(abs(u-ua).max())
    ax.plot(x, u, color=cols[k+1], label=f"t = {tgt}"); ax.plot(x, ua, color=INK, lw=0.8, ls=":")
ax.set_xlabel("x"); ax.set_ylabel("u(x,t)"); ax.set_title("Heat equation: explicit finite differences (colour) vs exact Gaussian kernel (dotted)"); ax.legend()
save(f_, "c5_heat")
V("c5_err", max(err), ".1e"); V("c5_sd_t1", np.sqrt(2*D*1.0), ".3f"); V("c5_sd_t25", np.sqrt(2*D*2.5), ".3f")
# Taylor example: exp(0.1)
V("c5_taylor1", 1+0.1, ".4f"); V("c5_taylor2", 1+0.1+0.005, ".4f"); V("c5_exp01", np.exp(0.1), ".5f")
# ODE: dP/dt = rP -> P
V("c5_ode", 100*np.exp(0.05*3), ".3f")
# unstable explicit alpha=0.6 demonstration
u2 = np.exp(-x**2/(4*D*0.05))/np.sqrt(4*np.pi*D*0.05); dt2 = 0.6*dx**2/D
for _ in range(200): u2[1:-1] = u2[1:-1] + D*dt2/dx**2*(u2[2:]-2*u2[1:-1]+u2[:-2])
V("c5_unstable_max", abs(u2).max(), ".1e")

# ---------------- Ch6 regression ----------------
nt = 500
rm = rng.normal(0.0004, 0.01, nt); eps = rng.normal(0, 0.012, nt)
alpha_t, beta_t = 0.0002, 1.2
ri = alpha_t + beta_t*rm + eps
Xd = np.column_stack([np.ones(nt), rm]); b = np.linalg.solve(Xd.T@Xd, Xd.T@ri)
res = ri - Xd@b; s2 = res@res/(nt-2); cov_b = s2*np.linalg.inv(Xd.T@Xd); se = np.sqrt(np.diag(cov_b))
V("c6_alpha", b[0]*1e4, ".2f"); V("c6_beta", b[1], ".3f"); V("c6_se_alpha", se[0]*1e4, ".2f"); V("c6_se_beta", se[1], ".3f")
V("c6_t_alpha", b[0]/se[0], ".2f"); V("c6_t_beta", b[1]/se[1], ".1f")
R2 = 1 - res@res/((ri-ri.mean())@(ri-ri.mean())); V("c6_R2", R2, ".3f")
V("c6_alpha_ann", b[0]*250*100, ".1f"); V("c6_alpha_true_ann", alpha_t*250*100, ".1f")
f_, ax = fig(8.4, 3.9)
ax.scatter(rm*100, ri*100, s=8, color=BLUE, alpha=.5, label="daily returns")
xx = np.linspace(rm.min(), rm.max(), 2)*100; ax.plot(xx, (b[0]+b[1]*xx/100)*100, color=ORANGE, label=f"OLS fit: β = {b[1]:.2f}")
ax.set_xlabel("market return (%)"); ax.set_ylabel("asset return (%)"); ax.set_title("Estimating a CAPM beta by least squares"); ax.legend()
save(f_, "c6_capm")
# multicollinearity: two nearly identical factors; OLS vs ridge
f1 = rng.normal(0, 1, nt); f2 = f1 + rng.normal(0, 0.05, nt); yv = 1.0*f1 + 1.0*f2 + rng.normal(0, 1, nt)
Xm = np.column_stack([f1, f2]); bo = np.linalg.solve(Xm.T@Xm, Xm.T@yv)
V("c6_ols_coll", ", ".join(f"{x:.2f}" for x in bo)); V("c6_cond_coll", np.linalg.cond(Xm.T@Xm), ".0f")
lam_r = 10.0; br = np.linalg.solve(Xm.T@Xm + lam_r*np.eye(2), Xm.T@yv); V("c6_ridge_coll", ", ".join(f"{x:.2f}" for x in br))
seo = np.sqrt(np.diag((yv-Xm@bo)@(yv-Xm@bo)/(nt-2)*np.linalg.inv(Xm.T@Xm))); V("c6_se_coll", ", ".join(f"{x:.2f}" for x in seo))
# 3-factor regression
fsize = rng.normal(0, 0.006, nt); fval = rng.normal(0, 0.005, nt)
r3 = 0.0001 + 1.1*rm + 0.5*fsize - 0.3*fval + rng.normal(0, 0.01, nt)
X3 = np.column_stack([np.ones(nt), rm, fsize, fval]); b3 = np.linalg.lstsq(X3, r3, rcond=None)[0]
res3 = r3 - X3@b3; se3 = np.sqrt(np.diag(res3@res3/(nt-4)*np.linalg.inv(X3.T@X3)))
V("c6_b3", ", ".join(f"{x:.3f}" for x in b3[1:])); V("c6_t3", ", ".join(f"{x:.1f}" for x in (b3/se3)[1:]))
# HAC note value: autocorrelated residuals inflate t by sqrt((1+rho)/(1-rho))
V("c6_hac_factor", np.sqrt((1+0.3)/(1-0.3)), ".2f")

flush()
