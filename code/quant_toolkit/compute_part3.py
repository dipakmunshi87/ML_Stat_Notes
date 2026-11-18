from lib import *
from scipy import stats, optimize
rng = np.random.default_rng(3)

# ---------------- Ch14 time series ----------------
T = 1000
phi = 0.7; e = rng.normal(size=T); ar = np.zeros(T)
for t in range(1, T): ar[t] = phi*ar[t-1] + e[t]
rw = np.cumsum(rng.normal(size=T))
def acf(x, k): x = x - x.mean(); return np.array([1] + [np.sum(x[:-j]*x[j:])/np.sum(x*x) for j in range(1, k+1)])
a_ar, a_rw = acf(ar, 20), acf(rw, 20)
V("c14_acf_ar", ", ".join(f"{x:.2f}" for x in a_ar[1:5])); V("c14_acf_theory", ", ".join(f"{phi**k:.2f}" for k in range(1, 5)))
V("c14_acf_rw", ", ".join(f"{x:.2f}" for x in a_rw[1:5])); V("c14_acf_rw20", a_rw[20], ".2f")
V("c14_ar_var", 1/(1-phi**2), ".2f"); V("c14_ar_var_sim", ar.var(), ".2f")
# AR(1) estimate and DF t-stat
def ar1_fit(x):
    X = x[:-1]; Y = x[1:]; b = np.sum(X*Y)/np.sum(X*X); res = Y - b*X; se = np.sqrt(res.var()/np.sum(X*X)); return b, se
b1, se1 = ar1_fit(ar); b2, se2 = ar1_fit(rw)
V("c14_phi_hat", b1, ".3f"); V("c14_phi_se", se1, ".3f"); V("c14_df_ar", (b1-1)/se1, ".1f"); V("c14_rw_phi", b2, ".3f"); V("c14_df_rw", (b2-1)/se2, ".2f")
# spurious regression: two independent random walks
y1 = np.cumsum(rng.normal(size=T)); y2 = np.cumsum(rng.normal(size=T))
X = np.column_stack([np.ones(T), y2]); bb = np.linalg.lstsq(X, y1, rcond=None)[0]; res = y1 - X@bb
se_b = np.sqrt(res.var()*np.linalg.inv(X.T@X)[1, 1]); V("c14_spur_t", bb[1]/se_b, ".1f"); V("c14_spur_R2", 1-res.var()/y1.var(), ".2f")
# cointegration & pairs
n = 1500; common = np.cumsum(rng.normal(0, 0.015, n))
kappa_s, sig_s = 0.05, 0.01   # daily mean reversion
spread = np.zeros(n)
for t in range(1, n): spread[t] = spread[t-1]*(1-kappa_s) + sig_s*rng.normal()
pA = 100*np.exp(common); pB = 100*np.exp(common - spread)   # ln pA - ln pB = spread (hedge ratio 1)
la, lb = np.log(pA), np.log(pB)
X = np.column_stack([np.ones(n), lb]); bh = np.linalg.lstsq(X, la, rcond=None)[0]; V("c14_hedge", bh[1], ".3f")
sp_hat = la - X@bh
b_s, se_s = ar1_fit(sp_hat); V("c14_sp_phi", b_s, ".4f"); V("c14_sp_halflife", -np.log(2)/np.log(b_s), ".1f"); V("c14_sp_halflife_true", np.log(2)/kappa_s, ".1f")
V("c14_eg_t", (b_s-1)/se_s, ".1f")
# simple z-score pairs backtest on second half, params from first half
half = n//2; mu_s, sd_s = sp_hat[:half].mean(), sp_hat[:half].std()
z = (sp_hat[half:] - mu_s)/sd_s; pos = np.zeros(len(z))
for t in range(1, len(z)):
    if z[t-1] > 1.5: pos[t] = -1
    elif z[t-1] < -1.5: pos[t] = 1
    elif abs(z[t-1]) < 0.25: pos[t] = 0
    else: pos[t] = pos[t-1]
ret_sp = np.diff(sp_hat[half-1:]); pnl = pos*ret_sp
V("c14_pairs_sharpe", pnl.mean()/pnl.std()*np.sqrt(250), ".2f"); V("c14_pairs_trades", int((np.diff(pos) != 0).sum()))
V("c14_pairs_sharpe_cost", (pnl.mean() - 0.0005*np.abs(np.diff(np.concatenate([[0], pos]))).mean())/pnl.std()*np.sqrt(250), ".2f")
f_, ax = fig(8.4, 3.8, ncols=2)
ax[0].plot(pA, color=BLUE, lw=1, label="A"); ax[0].plot(pB, color=ORANGE, lw=1, label="B"); ax[0].set_title("Two cointegrated prices"); ax[0].set_xlabel("day"); ax[0].legend()
ax[1].plot(sp_hat*100, color=AQUA, lw=0.9); ax[1].axhline(0, color=INK, lw=1)
for s_, ls in [(1.5, "--"), (-1.5, "--")]: ax[1].axhline((mu_s + s_*sd_s)*100, color=MUT, lw=1, ls=ls)
ax[1].set_title("The spread ln A − β ln B (%), with ±1.5σ bands"); ax[1].set_xlabel("day")
save(f_, "c14_pairs")
# Kalman filter: time-varying beta
nk = 1000; beta_true = 1.0 + 0.5*np.sin(np.linspace(0, 4*np.pi, nk)); xm = rng.normal(0, 0.01, nk); yk = beta_true*xm + rng.normal(0, 0.008, nk)
Q, R = 1e-4, 0.008**2; bk = np.zeros(nk); Pk = np.zeros(nk); b, P_ = 1.0, 1.0
for t in range(nk):
    P_ += Q; S_ = xm[t]**2*P_ + R; Kg = P_*xm[t]/S_; b = b + Kg*(yk[t] - b*xm[t]); P_ = (1 - Kg*xm[t])*P_; bk[t] = b; Pk[t] = P_
V("c14_kf_rmse", np.sqrt(np.mean((bk[100:]-beta_true[100:])**2)), ".3f")
# rolling OLS 60 days for comparison
roll = np.array([np.sum(xm[t-60:t]*yk[t-60:t])/np.sum(xm[t-60:t]**2) for t in range(60, nk)])
V("c14_roll_rmse", np.sqrt(np.mean((roll[40:]-beta_true[100:])**2)), ".3f")
f_, ax = fig(8.4, 3.4)
ax.plot(beta_true, color=INK, lw=1.2, ls="--", label="true β")
ax.plot(np.arange(60, nk), roll, color=ORANGE, lw=1, label="60-day rolling OLS")
ax.plot(bk, color=BLUE, lw=1.2, label="Kalman filter")
ax.set_xlabel("day"); ax.set_ylabel("β"); ax.set_title("Tracking a time-varying hedge ratio"); ax.legend(); ax.set_ylim(0.2, 1.8)
save(f_, "c14_kalman")

# ---------------- Ch15 GARCH ----------------
om, al, be = 2e-6, 0.08, 0.90
n = 3000; eps = np.zeros(n); h = np.zeros(n); h[0] = om/(1-al-be)
for t in range(1, n):
    h[t] = om + al*eps[t-1]**2 + be*h[t-1]; eps[t] = np.sqrt(h[t])*rng.normal()
V("c15_uncond", np.sqrt(om/(1-al-be))*np.sqrt(250)*100, ".1f"); V("c15_persist", al+be, ".2f"); V("c15_halflife", np.log(0.5)/np.log(al+be), ".1f")
V("c15_kurt", stats.kurtosis(eps), ".2f"); V("c15_kurt_th", 6*al**2/(1-be**2-2*al*be-3*al**2), ".2f")
V("c15_acf_r", acf(eps, 5)[1], ".3f"); V("c15_acf_r2", acf(eps**2, 5)[1], ".3f"); V("c15_acf_r2_20", acf(eps**2, 20)[20], ".3f")
def garch_nll(p, x):
    o, a, b = p
    if o <= 0 or a < 0 or b < 0 or a + b >= 1: return 1e10
    hh = np.empty(len(x)); hh[0] = x.var()
    for t in range(1, len(x)): hh[t] = o + a*x[t-1]**2 + b*hh[t-1]
    return 0.5*np.sum(np.log(hh) + x**2/hh)
res = optimize.minimize(garch_nll, [1e-6, 0.05, 0.9], args=(eps,), method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-9, maxiter=4000))
oh, ah, bh_ = res.x; V("c15_om_hat", oh, ".2e"); V("c15_al_hat", ah, ".3f"); V("c15_be_hat", bh_, ".3f"); V("c15_persist_hat", ah+bh_, ".3f")
# EWMA
lam = 0.94; ew = np.zeros(n); ew[0] = h[0]
for t in range(1, n): ew[t] = lam*ew[t-1] + (1-lam)*eps[t-1]**2
# realised (21-day rolling) vs garch
rv = np.array([np.mean(eps[t-21:t]**2) for t in range(21, n)])
f_, ax = fig(8.4, 3.8)
tt = np.arange(1000, 1600)
ax.plot(tt, np.sqrt(h[tt]*250)*100, color=BLUE, lw=1.3, label="true GARCH σ")
ax.plot(tt, np.sqrt(ew[tt]*250)*100, color=ORANGE, lw=1, label="EWMA (λ = 0.94)")
ax.plot(tt, np.sqrt(rv[tt-21]*250)*100, color=AQUA, lw=1, label="21-day realised")
ax.set_xlabel("day"); ax.set_ylabel("annualised volatility (%)"); ax.set_title("Volatility clustering and three estimates of the conditional volatility"); ax.legend()
save(f_, "c15_garch")
V("c15_vol_max", np.sqrt(h.max()*250)*100, ".0f"); V("c15_vol_min", np.sqrt(h.min()*250)*100, ".0f")
# forecast n-day variance
hT = h[-1]; V("c15_h_now", np.sqrt(hT*250)*100, ".1f")
fc = [om/(1-al-be) + (al+be)**k*(hT - om/(1-al-be)) for k in range(1, 251)]
V("c15_fc_10", np.sqrt(np.mean(fc[:10])*250)*100, ".1f"); V("c15_fc_250", np.sqrt(np.mean(fc)*250)*100, ".1f")

# ---------------- Ch16 RMT ----------------
Nn, Tt = 200, 500; q = Nn/Tt
lam_min, lam_max = (1-np.sqrt(q))**2, (1+np.sqrt(q))**2; V("c16_q", q, ".2f"); V("c16_lmin", lam_min, ".3f"); V("c16_lmax", lam_max, ".3f")
Xn = rng.normal(size=(Tt, Nn)); Cn = np.corrcoef(Xn.T); evn = np.linalg.eigvalsh(Cn)
V("c16_ev_noise_max", evn.max(), ".3f"); V("c16_ev_noise_min", evn.min(), ".3f")
def mp(x, q): return np.where((x > lam_min) & (x < lam_max), np.sqrt(np.maximum((lam_max-x)*(x-lam_min), 0))/(2*np.pi*q*x), 0)
# factor model: market + 3 sectors + noise
beta = rng.uniform(0.5, 1.5, Nn); fm = rng.normal(0, 0.012, Tt); sec = rng.integers(0, 4, Nn); fs = rng.normal(0, 0.006, (Tt, 4))
Rr = np.outer(fm, beta) + fs[:, sec] + rng.normal(0, 0.012, (Tt, Nn))
Cr = np.corrcoef(Rr.T); evr, EVr = np.linalg.eigvalsh(Cr), None
V("c16_ev_top", ", ".join(f"{x:.1f}" for x in np.sort(evr)[::-1][:5])); V("c16_n_above", int((evr > lam_max).sum()))
f_, ax = fig(8.4, 3.8, ncols=2)
xs = np.linspace(0.01, 4, 400)
ax[0].hist(evn, bins=40, density=True, color=BLUE, alpha=.7, label="pure-noise eigenvalues"); ax[0].plot(xs, mp(xs, q), color=INK, lw=1.5, label="Marchenko–Pastur, q = 0.4"); ax[0].set_xlabel("eigenvalue"); ax[0].set_title("Noise has a predictable spectrum"); ax[0].legend()
ax[1].hist(evr[evr < 4], bins=40, density=True, color=ORANGE, alpha=.7, label="factor-model eigenvalues (<4)"); ax[1].plot(xs, mp(xs, q), color=INK, lw=1.5, label="MP (noise)"); ax[1].axvline(lam_max, color=MUT, ls="--", lw=1)
ax[1].set_xlabel("eigenvalue"); ax[1].set_title(f"{int((evr > lam_max).sum())} eigenvalues above λ₊ (largest {evr.max():.0f})"); ax[1].legend()
save(f_, "c16_mp")
# out-of-sample min variance: sample vs clipped vs Ledoit-Wolf (identity target) using true cov
Sig_true = np.cov(Rr.T)  # use long-sample "true" from a separate big sample
Rbig = np.outer(rng.normal(0, 0.012, 20000), beta) + rng.normal(0, 0.006, (20000, 4))[:, sec] + rng.normal(0, 0.012, (20000, Nn)); Sig_true = np.cov(Rbig.T)
def minvar(S):
    w = np.linalg.solve(S, np.ones(Nn)); return w/w.sum()
def oos_vol(w): return np.sqrt(w@Sig_true@w*250)*100
Rs = Rbig[:Tt]  # in-sample window
S_s = np.cov(Rs.T)
sd = np.sqrt(np.diag(S_s)); C_s = S_s/np.outer(sd, sd); ev, EV = np.linalg.eigh(C_s)
ev_clip = ev.copy(); mask = ev < lam_max; ev_clip[mask] = ev[mask].mean(); C_clip = EV@np.diag(ev_clip)@EV.T; C_clip /= np.sqrt(np.outer(np.diag(C_clip), np.diag(C_clip))); S_clip = C_clip*np.outer(sd, sd)
# Ledoit-Wolf shrinkage to scaled identity (simple version)
mu_lw = np.trace(S_s)/Nn; F = mu_lw*np.eye(Nn); d2 = np.linalg.norm(S_s-F, "fro")**2
b2 = sum(np.linalg.norm(np.outer(x, x) - S_s, "fro")**2 for x in (Rs - Rs.mean(0)))/Tt**2; b2 = min(b2, d2); alpha_lw = b2/d2
S_lw = (1-alpha_lw)*S_s + alpha_lw*F
V("c16_lw_alpha", alpha_lw, ".3f")
V("c16_oos_sample", oos_vol(minvar(S_s)), ".2f"); V("c16_oos_clip", oos_vol(minvar(S_clip)), ".2f"); V("c16_oos_lw", oos_vol(minvar(S_lw)), ".2f"); V("c16_oos_true", oos_vol(minvar(Sig_true)), ".2f")
V("c16_is_sample", np.sqrt(minvar(S_s)@S_s@minvar(S_s)*250)*100, ".2f")
V("c16_gross_sample", np.abs(minvar(S_s)).sum(), ".1f"); V("c16_gross_clip", np.abs(minvar(S_clip)).sum(), ".1f"); V("c16_gross_lw", np.abs(minvar(S_lw)).sum(), ".1f")

# ---------------- Ch17 portfolio ----------------
vols = np.array([0.20, 0.25, 0.15]); corr = np.array([[1, .6, .2], [.6, 1, .3], [.2, .3, 1]]); Sig = np.outer(vols, vols)*corr
mu = np.array([0.07, 0.09, 0.04]); rf = 0.02
iS = np.linalg.inv(Sig); one = np.ones(3)
w_tan = iS@(mu-rf); w_tan /= w_tan.sum(); V("c17_wtan", ", ".join(f"{x:.3f}" for x in w_tan))
mu_t = w_tan@mu; s_t = np.sqrt(w_tan@Sig@w_tan); V("c17_mu_tan", mu_t*100, ".2f"); V("c17_sig_tan", s_t*100, ".2f"); V("c17_sr_tan", (mu_t-rf)/s_t, ".3f")
w_mv = iS@one/(one@iS@one); V("c17_wmv", ", ".join(f"{x:.3f}" for x in w_mv)); V("c17_sig_mv", np.sqrt(w_mv@Sig@w_mv)*100, ".2f"); V("c17_mu_mv", w_mv@mu*100, ".2f")
# frontier
A_ = one@iS@one; B_ = one@iS@mu; C_ = mu@iS@mu; D_ = A_*C_ - B_**2
mus = np.linspace(0.03, 0.12, 100); sig_f = np.sqrt((A_*mus**2 - 2*B_*mus + C_)/D_)
f_, ax = fig(8.4, 4.0)
ax.plot(sig_f*100, mus*100, color=BLUE, label="efficient frontier (long/short)")
ax.plot([0, s_t*100*1.6], [rf*100, (rf + (mu_t-rf)*1.6)*100], color=ORANGE, ls="--", label="capital market line")
ax.plot(s_t*100, mu_t*100, "o", color=ORANGE, ms=8); ax.annotate("tangency", (s_t*100, mu_t*100), xytext=(s_t*100-6.5, mu_t*100+0.9))
ax.plot(np.sqrt(w_mv@Sig@w_mv)*100, w_mv@mu*100, "s", color=AQUA, ms=8); ax.annotate("min variance", (np.sqrt(w_mv@Sig@w_mv)*100, w_mv@mu*100), xytext=(np.sqrt(w_mv@Sig@w_mv)*100-7.5, w_mv@mu*100-0.2))
for i, nm in enumerate(["A", "B", "C"]): ax.plot(vols[i]*100, mu[i]*100, "^", color=MAGENTA, ms=8); ax.annotate(nm, (vols[i]*100, mu[i]*100), xytext=(vols[i]*100+0.6, mu[i]*100))
ax.set_xlabel("volatility (%)"); ax.set_ylabel("expected return (%)"); ax.set_title("Mean–variance frontier for three assets"); ax.legend(loc="lower right"); ax.set_xlim(0, 32); ax.set_ylim(2, 12)
save(f_, "c17_frontier")
# estimation error: perturb mu by 1-year noise (se = vol/sqrt(1))... use 10-year se
wt_list = []
scale_w = (iS@(mu-rf)).sum()
for _ in range(2000):
    mu_n = mu + rng.normal(0, vols/np.sqrt(10)); wt_list.append(iS@(mu_n-rf)/scale_w)
wt_list = np.array(wt_list)
V("c17_w_iqr", "; ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(np.quantile(wt_list, 0.25, 0), np.quantile(wt_list, 0.75, 0))))
V("c17_w_frac_short", (wt_list.min(1) < 0).mean()*100, ".0f")
# Kelly
mu_k, sig_k = 0.08, 0.20; fstar = mu_k/sig_k**2; V("c17_kelly", fstar, ".2f"); V("c17_kelly_g", (mu_k**2/(2*sig_k**2))*100, ".2f")
fs = np.linspace(0, 5, 200); g = fs*mu_k - 0.5*fs**2*sig_k**2
V("c17_g_half", (0.5*fstar*mu_k - 0.5*(0.5*fstar)**2*sig_k**2)*100, ".2f"); V("c17_g_2x", (2*fstar*mu_k - 0.5*(2*fstar)**2*sig_k**2)*100, ".2f")
# discrete bet Kelly: p=0.55 even odds
V("c17_kelly_bet", 2*0.55-1, ".2f"); V("c17_kelly_bet_g", (0.55*np.log(1.1) + 0.45*np.log(0.9))*100, ".3f")
f_, ax = fig(8.4, 3.6)
ax.plot(fs, g*100, color=BLUE); ax.axvline(fstar, color=ORANGE, ls="--"); ax.text(fstar+0.05, 1.8, f"f* = μ/σ² = {fstar:.0f}", color=ORANGE)
ax.axhline(0, color=INK, lw=1); ax.set_xlabel("fraction of wealth invested f (leverage)"); ax.set_ylabel("long-run growth rate (%/yr)"); ax.set_title("Kelly: growth rate g(f) = fμ − ½f²σ² for μ = 8%, σ = 20%")
save(f_, "c17_kelly")
# risk parity on 3 assets
def rp(S):
    n_ = len(S); w = np.ones(n_)/n_
    for _ in range(2000):
        rc = w*(S@w); w = w*(rc.mean()/rc); w /= w.sum()
    return w
w_rp = rp(Sig); V("c17_wrp", ", ".join(f"{x:.3f}" for x in w_rp)); rc = w_rp*(Sig@w_rp)/(w_rp@Sig@w_rp); V("c17_rp_rc", ", ".join(f"{x:.3f}" for x in rc))
V("c17_sig_rp", np.sqrt(w_rp@Sig@w_rp)*100, ".2f"); V("c17_mu_rp", w_rp@mu*100, ".2f")
# transaction cost / turnover note
V("c17_mv_mu_pct", "n/a")

# ---------------- Ch18 risk ----------------
sig_d = 0.01; nu = 4; scale = sig_d*np.sqrt((nu-2)/nu)
for a in [0.95, 0.99, 0.999]:
    V(f"c18_varg_{int(a*1000)}", -stats.norm.ppf(1-a)*sig_d*100, ".2f"); V(f"c18_esg_{int(a*1000)}", stats.norm.pdf(stats.norm.ppf(1-a))/(1-a)*sig_d*100, ".2f")
    tq = stats.t.ppf(1-a, nu); V(f"c18_vart_{int(a*1000)}", -tq*scale*100, ".2f")
    es_t = (stats.t.pdf(tq, nu)*(nu + tq**2)/((nu-1)*(1-a)))*scale*100; V(f"c18_est_{int(a*1000)}", es_t, ".2f")
# VaR not subadditive: two independent bonds, 4% default prob each, loss 100
p = 0.04; V("c18_bond_var_single", 0.0); V("c18_bond_var_joint_p", (1-(1-p)**2)*100, ".2f")
# EVT: GPD fit to tail of 10 years daily t returns
x = rng.standard_t(nu, 2500)*scale; losses = -x; u = np.quantile(losses, 0.95); exc = losses[losses > u] - u
xi_hat, _, beta_hat = stats.genpareto.fit(exc, floc=0); V("c18_gpd_xi", xi_hat, ".2f"); V("c18_gpd_beta", beta_hat*100, ".3f"); V("c18_u", u*100, ".2f"); V("c18_nexc", len(exc))
def gpd_var(a): return u + beta_hat/xi_hat*(((1-a)/(len(exc)/len(losses)))**(-xi_hat) - 1)
V("c18_evt_var999", gpd_var(0.999)*100, ".2f"); V("c18_emp_var999", np.quantile(losses, 0.999)*100, ".2f"); V("c18_true_var999", -stats.t.ppf(0.001, nu)*scale*100, ".2f")
V("c18_evt_var9999", gpd_var(0.9999)*100, ".2f"); V("c18_true_var9999", -stats.t.ppf(0.0001, nu)*scale*100, ".2f"); V("c18_gauss_var9999", -stats.norm.ppf(0.0001)*sig_d*100, ".2f")
f_, ax = fig(8.4, 3.8, ncols=2)
srt = np.sort(losses)[::-1]; ranks = np.arange(1, len(srt)+1)/len(srt)
ax[0].loglog(srt*100, ranks, ".", color=BLUE, ms=3, label="empirical tail P(L > x)")
xx = np.linspace(u, srt[0]*1.5, 200); ax[0].loglog(xx*100, (len(exc)/len(losses))*(1 + xi_hat*(xx-u)/beta_hat)**(-1/xi_hat), color=ORANGE, label=f"GPD fit, ξ = {xi_hat:.2f}")
ax[0].loglog(xx*100, stats.norm.sf(xx/sig_d), color=MUT, ls="--", label="Gaussian"); ax[0].set_xlabel("loss (%)"); ax[0].set_ylabel("exceedance probability"); ax[0].set_title("Tail of daily losses (10 years)"); ax[0].legend(fontsize=8.5); ax[0].set_ylim(1e-4, 1); ax[0].set_xlim(0.2, 15)
# drawdown distribution
Mdd = 5000; rets = rng.normal(0.08/250, 0.20/np.sqrt(250), (Mdd, 1250)); eq = np.cumsum(rets, 1); dd = (np.maximum.accumulate(eq, 1) - eq).max(1)
V("c18_dd_med", np.median(dd)*100, ".0f"); V("c18_dd_90", np.quantile(dd, 0.9)*100, ".0f"); V("c18_dd_99", np.quantile(dd, 0.99)*100, ".0f")
ax[1].hist(dd*100, bins=60, color=AQUA, alpha=.8, density=True); ax[1].set_xlabel("maximum drawdown over 5 years (%)"); ax[1].set_title("Max drawdown, SR = 0.4 strategy (5,000 runs)")
save(f_, "c18_tails")

# ---------------- Ch19 ML signals ----------------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
nT, nf = 3000, 20
F = rng.normal(size=(nT, nf))
# AR-ish features (persistent) to create leakage with random CV
for j in range(nf):
    for t in range(1, nT): F[t, j] = 0.9*F[t-1, j] + np.sqrt(1-0.81)*F[t, j]
signal = 3.2*(0.03*F[:, 0] + 0.02*np.tanh(F[:, 1]) + 0.015*F[:, 2]*F[:, 3])
y = signal + rng.normal(0, 1.0, nT)*1.0
V("c19_true_R2", signal.var()/y.var()*100, ".1f")
# walk-forward
def walk_forward(model_fn, X, y, start=1000, step=250):
    preds = np.full(len(y), np.nan)
    for s in range(start, len(y), step):
        m = model_fn(); m.fit(X[:s], y[:s]); preds[s:s+step] = m.predict(X[s:s+step])
    return preds
def oos_stats(preds, y):
    m = ~np.isnan(preds); r2 = 1 - np.sum((y[m]-preds[m])**2)/np.sum((y[m]-y[m].mean())**2)
    pnl = np.sign(preds[m])*y[m]; return r2*100, pnl.mean()/pnl.std()*np.sqrt(250), np.corrcoef(preds[m], y[m])[0, 1]
p_lin = walk_forward(lambda: Ridge(alpha=10.0), F, y); p_gb = walk_forward(lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.7, random_state=0), F, y)
r2l, srl, icl = oos_stats(p_lin, y); r2g, srg, icg = oos_stats(p_gb, y)
V("c19_lin_R2", r2l, ".2f"); V("c19_lin_SR", srl, ".2f"); V("c19_lin_IC", icl, ".3f"); V("c19_gb_R2", r2g, ".2f"); V("c19_gb_SR", srg, ".2f"); V("c19_gb_IC", icg, ".3f")
# random k-fold with leakage-prone target: use overlapping 5-day forward return target
y5 = np.array([y[t:t+5].sum() for t in range(nT-5)]); F5 = F[:nT-5]
kf = KFold(5, shuffle=True, random_state=0); pk = np.zeros(len(y5))
for tr, te in kf.split(F5):
    m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.7, random_state=0); m.fit(F5[tr], y5[tr]); pk[te] = m.predict(F5[te])
V("c19_kfold_R2", (1 - np.sum((y5-pk)**2)/np.sum((y5-y5.mean())**2))*100, ".1f")
pw = walk_forward(lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.7, random_state=0), F5, y5)
m_ = ~np.isnan(pw); V("c19_wf5_R2", (1 - np.sum((y5[m_]-pw[m_])**2)/np.sum((y5[m_]-y5[m_].mean())**2))*100, ".1f")
# overfitting: in-sample R2 of deep trees
m = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=0).fit(F[:1000], y[:1000]); V("c19_is_R2_deep", (1 - np.sum((y[:1000]-m.predict(F[:1000]))**2)/np.sum((y[:1000]-y[:1000].mean())**2))*100, ".0f")
V("c19_oos_R2_deep", (1 - np.sum((y[1000:]-m.predict(F[1000:]))**2)/np.sum((y[1000:]-y[1000:].mean())**2))*100, ".1f")
f_, ax = fig(8.4, 3.6)
mm = ~np.isnan(p_lin); t_ = np.arange(nT)[mm]
ax.plot(t_, np.cumsum(np.sign(p_lin[mm])*y[mm]), color=BLUE, label=f"ridge (OOS SR {srl:.2f})"); ax.plot(t_, np.cumsum(np.sign(p_gb[mm])*y[mm]), color=ORANGE, label=f"gradient boosting (OOS SR {srg:.2f})")
ax.plot(t_, np.cumsum(np.sign(signal[mm])*y[mm]), color=INK, ls="--", lw=1, label="oracle (true signal)")
ax.set_xlabel("day"); ax.set_ylabel("cumulative P&L (units of daily σ)"); ax.set_title("Walk-forward out-of-sample performance of two learners on a 1%-R² signal"); ax.legend()
save(f_, "c19_ml")
V("c19_oracle_SR", (np.sign(signal[mm])*y[mm]).mean()/(np.sign(signal[mm])*y[mm]).std()*np.sqrt(250), ".2f")

# ---------------- Ch20 backtesting / multiple testing ----------------
Ty = 5; nd = Ty*250
for Nn_ in [1, 10, 100, 1000, 10000]:
    reps = 30 if Nn_ < 10000 else 5
    V(f"c20_maxsr_{Nn_}", np.mean([rng.normal(0, 1, (Nn_, nd)).mean(1).max()*np.sqrt(250) for _ in range(reps)]), ".2f")
    V(f"c20_exp_maxsr_{Nn_}", (np.sqrt(2*np.log(Nn_)) if Nn_ > 1 else 0)/np.sqrt(Ty), ".2f")
# distribution of max over 1000 trials, repeated
mx = [rng.normal(0, 1, (1000, nd)).mean(1).max()*np.sqrt(250)/1 for _ in range(200)]
V("c20_maxsr_1000_mean", np.mean(mx), ".2f"); V("c20_maxsr_1000_sd", np.std(mx), ".2f")
# in-sample vs out-of-sample for the best of 1000 (split 5y/5y)
R = rng.normal(0, 1, (1000, 2*nd)); sr_is = R[:, :nd].mean(1)*np.sqrt(250); best = sr_is.argmax(); V("c20_best_is", sr_is[best], ".2f"); V("c20_best_oos", R[best, nd:].mean()*np.sqrt(250), ".2f")
# with 10% true skill
Rs = rng.normal(0, 1, (1000, 2*nd)); true_sr = np.where(rng.random(1000) < 0.1, 0.5, 0.0); Rs += (true_sr/np.sqrt(250))[:, None]
sr_is = Rs[:, :nd].mean(1)*np.sqrt(250); top = np.argsort(sr_is)[-10:]; V("c20_top10_is", sr_is[top].mean(), ".2f"); V("c20_top10_oos", (Rs[top, nd:].mean(1)*np.sqrt(250)).mean(), ".2f"); V("c20_top10_skilled", int((true_sr[top] > 0).sum()))
V("c20_bonf_t", stats.norm.isf(0.025/1000), ".2f"); V("c20_t_needed_sr", stats.norm.isf(0.025/1000)/np.sqrt(Ty), ".2f")
# deflated sharpe: probability that observed SR 1.0 over 5y with N=1000 trials exceeds expected max
V("c20_sr_se_5y", np.sqrt(1/Ty), ".2f")
f_, ax = fig(8.4, 3.6)
Ns = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000]); mxs = []
for Nn_ in Ns: mxs.append(np.mean([rng.normal(0, 1, (Nn_, nd)).mean(1).max()*np.sqrt(250) for _ in range(30)]))
ax.semilogx(Ns, mxs, "o-", color=BLUE, label="simulated mean of best Sharpe")
ax.semilogx(Ns[1:], np.sqrt(2*np.log(Ns[1:]))/np.sqrt(Ty), color=ORANGE, ls="--", label="√(2 ln N) / √T")
ax.set_xlabel("number of strategies tried, N"); ax.set_ylabel("best 5-year Sharpe (all strategies have zero skill)"); ax.set_title("The best of N random strategies looks better and better"); ax.legend()
save(f_, "c20_maxsharpe")
flush()
