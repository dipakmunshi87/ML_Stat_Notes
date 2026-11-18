from lib import *
from scipy import stats, optimize, integrate
rng = np.random.default_rng(2)
N_ = stats.norm.cdf; n_ = stats.norm.pdf

def bs(S, K, r, sig, T, kind="call", q=0.0):
    d1 = (np.log(S/K) + (r - q + 0.5*sig**2)*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
    if kind == "call": return S*np.exp(-q*T)*N_(d1) - K*np.exp(-r*T)*N_(d2)
    return K*np.exp(-r*T)*N_(-d2) - S*np.exp(-q*T)*N_(-d1)
def greeks(S, K, r, sig, T):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
    return dict(delta=N_(d1), gamma=n_(d1)/(S*sig*np.sqrt(T)), vega=S*n_(d1)*np.sqrt(T),
                theta=-S*n_(d1)*sig/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*N_(d2), rho=K*T*np.exp(-r*T)*N_(d2))
def impvol(price, S, K, r, T, kind="call"):
    try: return optimize.brentq(lambda s: bs(S, K, r, s, T, kind) - price, 1e-4, 5.0)
    except ValueError: return np.nan

# ---------------- Ch7 random walk / Brownian motion ----------------
for n in [10, 100, 1000, 10000]:
    walks = rng.choice([-1, 1], size=(2000, n)).sum(1); V(f"c7_rw_sd_{n}", walks.std(), ".1f")
# quadratic variation of a BM path on [0,1] vs a smooth function
for n in [100, 1000, 10000]:
    dW = rng.normal(0, np.sqrt(1/n), n); W = np.cumsum(dW)
    V(f"c7_qv_{n}", (dW**2).sum(), ".3f")
    tgrid = np.linspace(0, 1, n+1); f = np.sin(2*np.pi*tgrid); V(f"c7_qv_smooth_{n}", (np.diff(f)**2).sum(), ".2e")
V("c7_qv_theory_sd", np.sqrt(2/1000), ".3f")
# self-similarity figure: same path viewed at three zooms
n = 2**16; dW = rng.normal(0, np.sqrt(1/n), n); W = np.concatenate([[0], np.cumsum(dW)]); tg = np.linspace(0, 1, n+1)
f_, ax = fig(8.4, 3.4, ncols=3)
for a, frac in zip(ax, [1, 1/16, 1/256]):
    m = int(n*frac); a.plot(tg[:m+1], W[:m+1], color=BLUE, lw=0.8); a.set_title(f"t ∈ [0, {frac:g}]"); a.set_xlabel("t")
    a.set_ylim(W[:m+1].min()-0.02*np.sqrt(frac), W[:m+1].max()+0.02*np.sqrt(frac))
ax[0].set_ylabel("W(t)")
save(f_, "c7_selfsimilar")
# hitting / reflection: P(max W > a)
a_ = 1.0; V("c7_pmax", 2*(1-N_(a_)), ".3f")
Wmax = np.array([np.cumsum(rng.normal(0, np.sqrt(1/1000), 1000)).max() for _ in range(5000)]); V("c7_pmax_sim", (Wmax > a_).mean(), ".3f")

# ---------------- Ch8 Ito ----------------
S0, mu, sig, T = 100.0, 0.10, 0.20, 1.0
V("c8_ES", S0*np.exp(mu*T), ".3f"); V("c8_median", S0*np.exp((mu-0.5*sig**2)*T), ".3f"); V("c8_drag", 0.5*sig**2*100, ".1f")
M, n = 100000, 250; dt = T/n
Z = rng.normal(size=(M, n))
S_exact = S0*np.exp(np.cumsum((mu-0.5*sig**2)*dt + sig*np.sqrt(dt)*Z, axis=1))[:, -1]
V("c8_ES_sim", S_exact.mean(), ".2f"); V("c8_median_sim", np.median(S_exact), ".2f")
V("c8_ElogS", np.log(S_exact).mean(), ".4f"); V("c8_ElogS_th", np.log(S0) + (mu-0.5*sig**2)*T, ".4f")
# Euler weak error vs n
for nn in [4, 16, 64, 250]:
    dtn = T/nn; Zn = rng.normal(size=(50000, nn)); Se = S0*np.prod(1 + mu*dtn + sig*np.sqrt(dtn)*Zn, axis=1)
    V(f"c8_euler_{nn}", Se.mean(), ".2f")
# OU process
kappa, theta, s_ou = 5.0, 0.03, 0.02; x0 = 0.08
nt = 2000; tt = np.linspace(0, 2, nt+1); dtt = tt[1]-tt[0]
x = np.empty(nt+1); x[0] = x0
for k in range(nt): x[k+1] = x[k] + kappa*(theta - x[k])*dtt + s_ou*np.sqrt(dtt)*rng.normal()
V("c8_ou_halflife", np.log(2)/kappa, ".3f"); V("c8_ou_statsd", s_ou/np.sqrt(2*kappa)*100, ".3f")
f_, ax = fig(8.4, 3.6, ncols=2)
pth = S0*np.exp(np.cumsum((mu-0.5*sig**2)*dt + sig*np.sqrt(dt)*rng.normal(size=(5, n)), axis=1))
for i in range(5): ax[0].plot(np.linspace(0, T, n), pth[i], lw=1.1, color=SERIES[i])
ax[0].plot(np.linspace(0, T, n), S0*np.exp(mu*np.linspace(0, T, n)), color=INK, ls="--", lw=1.2, label="E[S_t]"); ax[0].legend()
ax[0].set_title("Five GBM paths (μ = 10%, σ = 20%)"); ax[0].set_xlabel("t (years)"); ax[0].set_ylabel("S")
ax[1].plot(tt, x*100, color=ORANGE, lw=1); ax[1].axhline(theta*100, color=INK, ls="--", lw=1.2); ax[1].text(1.5, theta*100+0.4, "θ = 3%", color=INK)
ax[1].set_title("Ornstein–Uhlenbeck: mean reversion (κ = 5)"); ax[1].set_xlabel("t (years)"); ax[1].set_ylabel("x (%)")
save(f_, "c8_paths")

# ---------------- Ch9 Black-Scholes ----------------
S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
C = bs(S0, K, r, sig, T); P = bs(S0, K, r, sig, T, "put")
d1 = (np.log(S0/K) + (r + 0.5*sig**2)*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
V("c9_d1", d1, ".4f"); V("c9_d2", d2, ".4f"); V("c9_Nd1", N_(d1), ".4f"); V("c9_Nd2", N_(d2), ".4f")
V("c9_call", C, ".4f"); V("c9_put", P, ".4f"); V("c9_parity", S0 - K*np.exp(-r*T), ".4f"); V("c9_df", np.exp(-r*T), ".5f")
# MC estimate
for M in [1000, 100000, 1000000]:
    ST = S0*np.exp((r-0.5*sig**2)*T + sig*np.sqrt(T)*rng.normal(size=M)); pay = np.exp(-r*T)*np.maximum(ST-K, 0)
    V(f"c9_mc_{M}", pay.mean(), ".4f"); V(f"c9_mcse_{M}", pay.std()/np.sqrt(M), ".4f")
# price with mu instead (wrong): shows mu-independence
V("c9_call_mu_wrong", np.exp(-r*T)*np.mean(np.maximum(S0*np.exp((0.10-0.5*sig**2)*T + sig*np.sqrt(T)*rng.normal(size=1000000))-K, 0)), ".3f")
# figure: price vs S at several T (smoothing of hockey stick)
Ss = np.linspace(60, 140, 300)
f_, ax = fig(8.4, 3.9)
for Tt, col in zip([0.02, 0.25, 1.0, 2.0], [BLUE, AQUA, ORANGE, MAGENTA]): ax.plot(Ss, bs(Ss, K, r, sig, Tt), color=col, label=f"T = {Tt}")
ax.plot(Ss, np.maximum(Ss-K, 0), color=INK, lw=1, ls="--", label="payoff (T = 0)")
ax.set_xlabel("spot S"); ax.set_ylabel("call price"); ax.set_title("Black–Scholes call price: the payoff kink diffuses away"); ax.legend()
save(f_, "c9_smoothing")

# ---------------- Ch10 Greeks & hedging ----------------
G = greeks(S0, K, r, sig, T)
for k, v in G.items(): V(f"c10_{k}", v, ".4f")
V("c10_vega_1pct", G["vega"]/100, ".4f"); V("c10_theta_day", G["theta"]/250, ".4f")
V("c10_gamma_pnl_1", 0.5*G["gamma"]*1**2, ".4f"); V("c10_gamma_pnl_5", 0.5*G["gamma"]*5**2, ".4f")
V("c10_theta_gamma_check", -(G["theta"] + r*K*np.exp(-r*T)*N_(d2)) / (0.5*G["gamma"]*S0**2*sig**2), ".4f")
# discrete delta hedging simulation (real-world drift 10%)
def hedge_sim(nsteps, M=20000, sig_real=0.20, sig_hedge=0.20, mu_real=0.10):
    dt = T/nsteps; S = np.full(M, S0); cash = np.zeros(M)
    C0 = bs(S0, K, r, sig_hedge, T); delta = greeks(S0, K, r, sig_hedge, T)["delta"]
    cash = C0 - delta*S
    for k in range(1, nsteps+1):
        S = S*np.exp((mu_real-0.5*sig_real**2)*dt + sig_real*np.sqrt(dt)*rng.normal(size=M)); cash *= np.exp(r*dt)
        tau = T - k*dt
        nd = greeks(S, K, r, sig_hedge, max(tau, 1e-9))["delta"] if tau > 1e-9 else (S > K).astype(float)
        cash -= (nd - delta)*S; delta = nd
    pnl = cash + delta*S - np.maximum(S-K, 0)   # writer's hedged P&L
    return pnl
for ns in [10, 50, 250]:
    pnl = hedge_sim(ns); V(f"c10_hedge_mean_{ns}", pnl.mean(), ".3f"); V(f"c10_hedge_sd_{ns}", pnl.std(), ".3f")
    if ns == 250: pnl250 = pnl
    if ns == 10: pnl10 = pnl
V("c10_hedge_sd_ratio", np.sqrt(250/10), ".1f")
pnl_mis = hedge_sim(250, sig_real=0.25, sig_hedge=0.20); V("c10_mis_mean", pnl_mis.mean(), ".3f"); V("c10_mis_sd", pnl_mis.std(), ".3f")
V("c10_mis_vega_est", -G["vega"]*0.05, ".3f")
V("c10_price_25", bs(S0, K, r, 0.25, T), ".4f")
f_, ax = fig(8.4, 3.8)
ax.hist(pnl10, bins=80, color=ORANGE, alpha=.6, density=True, label="hedge 10× (every 25 days)")
ax.hist(pnl250, bins=80, color=BLUE, alpha=.7, density=True, label="hedge 250× (daily)")
ax.set_xlim(-8, 8); ax.set_xlabel("P&L of a delta-hedged short call at expiry"); ax.set_ylabel("density"); ax.set_title("Discrete delta hedging: the error shrinks as 1/√(number of rebalances)"); ax.legend()
save(f_, "c10_hedging")

# ---------------- Ch11 smiles: Merton jumps and Heston ----------------
def merton(S, K, r, sig, T, lam, m, delta, kind="call", nmax=40):
    k_ = np.exp(m + 0.5*delta**2) - 1; lamp = lam*(1+k_); tot = 0.0
    for nj in range(nmax):
        sig_n = np.sqrt(sig**2 + nj*delta**2/T); r_n = r - lam*k_ + nj*np.log(1+k_)/T
        tot += np.exp(-lamp*T)*(lamp*T)**nj/np.math.factorial(nj) * bs(S, K, r_n, sig_n, T, kind)
    return tot
import math
np.math = math
lam, mJ, dJ = 0.5, -0.10, 0.15
Ks = np.linspace(70, 130, 25)
iv_m = [impvol(merton(S0, k, r, 0.15, T, lam, mJ, dJ), S0, k, r, T) for k in Ks]
V("c11_merton_atm", merton(S0, 100, r, 0.15, T, lam, mJ, dJ), ".3f"); V("c11_merton_iv90", impvol(merton(S0, 90, r, 0.15, T, lam, mJ, dJ), S0, 90, r, T)*100, ".1f")
V("c11_merton_iv110", impvol(merton(S0, 110, r, 0.15, T, lam, mJ, dJ), S0, 110, r, T)*100, ".1f"); V("c11_merton_ivatm", impvol(merton(S0, 100, r, 0.15, T, lam, mJ, dJ), S0, 100, r, T)*100, ".1f")
V("c11_merton_totvar", np.sqrt(0.15**2 + lam*(mJ**2 + dJ**2))*100, ".1f")

def heston_call(S, K, r, T, v0, kappa, theta, xi, rho):
    x = np.log(S)
    def P(j):
        u = 0.5 if j == 1 else -0.5; b = kappa - rho*xi if j == 1 else kappa; a = kappa*theta
        def integrand(phi):
            d = np.sqrt((rho*xi*1j*phi - b)**2 - xi**2*(2*u*1j*phi - phi**2))
            c = (b - rho*xi*1j*phi - d)/(b - rho*xi*1j*phi + d)
            Cc = r*1j*phi*T + a/xi**2*((b - rho*xi*1j*phi - d)*T - 2*np.log((1 - c*np.exp(-d*T))/(1 - c)))
            Dd = (b - rho*xi*1j*phi - d)/xi**2*(1 - np.exp(-d*T))/(1 - c*np.exp(-d*T))
            f = np.exp(Cc + Dd*v0 + 1j*phi*x)
            return np.real(np.exp(-1j*phi*np.log(K))*f/(1j*phi))
        val, _ = integrate.quad(integrand, 1e-8, 200, limit=400)
        return 0.5 + val/np.pi
    return S*P(1) - K*np.exp(-r*T)*P(2)
hp = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.39, rho=-0.7)
V("c11_feller", 2*hp["kappa"]*hp["theta"] >= hp["xi"]**2)
hc = heston_call(S0, K, r, T, **hp); V("c11_heston_atm", hc, ".4f")
# MC check of Heston (full truncation Euler)
Mh, nh = 100000, 500; dth = T/nh
Sh = np.full(Mh, S0); vh = np.full(Mh, hp["v0"])
for _ in range(nh):
    z1 = rng.normal(size=Mh); z2 = hp["rho"]*z1 + np.sqrt(1-hp["rho"]**2)*rng.normal(size=Mh)
    vp = np.maximum(vh, 0)
    Sh = Sh*np.exp((r - 0.5*vp)*dth + np.sqrt(vp*dth)*z1)
    vh = vh + hp["kappa"]*(hp["theta"] - vp)*dth + hp["xi"]*np.sqrt(vp*dth)*z2
hmc = np.exp(-r*T)*np.maximum(Sh-K, 0); V("c11_heston_mc", hmc.mean(), ".4f"); V("c11_heston_mcse", hmc.std()/np.sqrt(Mh), ".4f")
iv_h = [impvol(heston_call(S0, k, r, T, **hp), S0, k, r, T) for k in Ks]
V("c11_heston_iv90", iv_h[np.argmin(abs(Ks-90))]*100, ".1f"); V("c11_heston_iv110", iv_h[np.argmin(abs(Ks-110))]*100, ".1f"); V("c11_heston_ivatm", iv_h[np.argmin(abs(Ks-100))]*100, ".1f")
hp2 = dict(hp, rho=0.0); iv_h0 = [impvol(heston_call(S0, k, r, T, **hp2), S0, k, r, T) for k in Ks]
f_, ax = fig(8.4, 3.9)
ax.plot(Ks, np.array(iv_m)*100, color=ORANGE, marker="o", ms=4, label="Merton jump-diffusion (σ=15%, λ=0.5, m=−10%, δ=15%)")
ax.plot(Ks, np.array(iv_h)*100, color=BLUE, marker="s", ms=4, label="Heston (ρ = −0.7)")
ax.plot(Ks, np.array(iv_h0)*100, color=AQUA, marker="^", ms=4, label="Heston (ρ = 0)")
ax.axhline(20, color=INK, ls="--", lw=1); ax.text(126, 20.3, "flat BS 20%", color=INK, ha="right")
ax.set_xlabel("strike K (spot = 100, T = 1y)"); ax.set_ylabel("implied volatility (%)"); ax.set_title("Implied-volatility smiles produced by jumps and by stochastic volatility"); ax.legend(fontsize=8.5)
save(f_, "c11_smiles")
# variance swap fair strike from Heston: E[1/T int v dt]
V("c11_varswap", np.sqrt(hp["theta"] + (hp["v0"]-hp["theta"])*(1-np.exp(-hp["kappa"]*T))/(hp["kappa"]*T))*100, ".2f")

# ---------------- Ch12 interest rates: Vasicek ----------------
def vasicek_P(r0, t, kappa, theta, s):
    B = (1 - np.exp(-kappa*t))/kappa
    A = np.exp((theta - s**2/(2*kappa**2))*(B - t) - s**2*B**2/(4*kappa))
    return A*np.exp(-B*r0)
kv, thv, sv = 0.3, 0.05, 0.01
V("c12_yinf", (thv - sv**2/(2*kv**2))*100, ".3f")
f_, ax = fig(8.4, 3.8)
ts = np.linspace(0.05, 30, 300)
for r0, col in zip([0.02, 0.05, 0.08], [BLUE, AQUA, ORANGE]):
    yv = -np.log(vasicek_P(r0, ts, kv, thv, sv))/ts; ax.plot(ts, yv*100, color=col, label=f"r₀ = {r0*100:.0f}%")
    V(f"c12_y10_{int(r0*100)}", (-np.log(vasicek_P(r0, 10, kv, thv, sv))/10)*100, ".3f")
ax.set_xlabel("maturity (years)"); ax.set_ylabel("zero yield (%)"); ax.set_title("Vasicek yield curves (κ = 0.3, θ = 5%, σ = 1%) for three short rates"); ax.legend()
save(f_, "c12_vasicek")
V("c12_P5_5", vasicek_P(0.05, 5, kv, thv, sv), ".5f"); V("c12_B5", (1-np.exp(-kv*5))/kv, ".4f")
V("c12_P5_5_naive", np.exp(-0.05*5), ".5f")
# MC check of P(0,5) with r0=5%
Mr, nr = 50000, 500; dtr = 5/nr; rr = np.full(Mr, 0.05); integ = np.zeros(Mr)
for _ in range(nr):
    integ += rr*dtr; rr = rr + kv*(thv - rr)*dtr + sv*np.sqrt(dtr)*rng.normal(size=Mr)
V("c12_P5_5_mc", np.exp(-integ).mean(), ".5f")
V("c12_rneg", (rr < 0).mean()*100, ".2f")
V("c12_cir_feller", 2*kv*thv >= sv**2)
# duration of 10y zero in Vasicek vs naive
V("c12_B10", (1-np.exp(-kv*10))/kv, ".3f")

# ---------------- Ch13 numerical methods ----------------
def crr(S, K, r, sig, T, n, kind="call", american=False):
    dt = T/n; u = np.exp(sig*np.sqrt(dt)); d = 1/u; p = (np.exp(r*dt) - d)/(u - d); disc = np.exp(-r*dt)
    ST = S*u**np.arange(n, -1, -1)*d**np.arange(0, n+1)
    V_ = np.maximum(ST-K, 0) if kind == "call" else np.maximum(K-ST, 0)
    for k in range(n-1, -1, -1):
        ST = ST[:-1]/u; V_ = disc*(p*V_[:-1] + (1-p)*V_[1:])
        if american: V_ = np.maximum(V_, (ST-K if kind == "call" else K-ST))
    return V_[0]
ns = np.arange(5, 400)
tree = np.array([crr(S0, K, r, sig, T, int(n)) for n in ns])
for n in [10, 50, 100, 500, 2000]: V(f"c13_tree_{n}", crr(S0, K, r, sig, T, n), ".4f")
V("c13_am_put_2000", crr(S0, K, r, sig, T, 2000, "put", True), ".4f"); V("c13_eu_put", P, ".4f")
# Crank-Nicolson European call (log-space not needed; S-space with uniform grid)
def cn_call(S0, K, r, sig, T, nS=400, nT=400, Smax=300.0):
    S = np.linspace(0, Smax, nS+1); dS = S[1]-S[0]; dt = T/nT
    Vv = np.maximum(S-K, 0)
    i = np.arange(1, nS); a = 0.25*dt*(sig**2*i**2 - r*i); b = -0.5*dt*(sig**2*i**2 + r); c = 0.25*dt*(sig**2*i**2 + r*i)
    from scipy.linalg import solve_banded
    ab = np.zeros((3, nS-1)); ab[0, 1:] = -c[:-1]; ab[1] = 1 - b; ab[2, :-1] = -a[1:]
    for k in range(nT):
        tau = (k+1)*dt
        rhs = a*Vv[:-2] + (1 + b)*Vv[1:-1] + c*Vv[2:]
        rhs[-1] += c[-1]*(Smax - K*np.exp(-r*tau))   # upper boundary contribution (new level)
        Vn = np.zeros_like(Vv); Vn[-1] = Smax - K*np.exp(-r*tau); Vn[0] = 0
        Vn[1:-1] = solve_banded((1, 1), ab, rhs); Vv = Vn
    return np.interp(S0, S, Vv)
V("c13_cn", cn_call(S0, K, r, sig, T), ".4f"); V("c13_cn_coarse", cn_call(S0, K, r, sig, T, 50, 50), ".4f")
# MC variance reduction
M = 100000; Z = rng.normal(size=M)
ST = S0*np.exp((r-0.5*sig**2)*T + sig*np.sqrt(T)*Z); pay = np.exp(-r*T)*np.maximum(ST-K, 0)
V("c13_mc_plain", pay.mean(), ".4f"); V("c13_mc_plain_se", pay.std()/np.sqrt(M), ".4f")
STa = S0*np.exp((r-0.5*sig**2)*T - sig*np.sqrt(T)*Z); paya = 0.5*(pay + np.exp(-r*T)*np.maximum(STa-K, 0))
V("c13_mc_anti", paya.mean(), ".4f"); V("c13_mc_anti_se", paya.std()/np.sqrt(M), ".4f")
cv = np.exp(-r*T)*ST; beta_cv = np.cov(pay, cv)[0, 1]/np.var(cv); paycv = pay - beta_cv*(cv - S0)
V("c13_mc_cv", paycv.mean(), ".4f"); V("c13_mc_cv_se", paycv.std()/np.sqrt(M), ".4f"); V("c13_cv_beta", beta_cv, ".3f")
V("c13_mc_varred", (pay.std()/paycv.std())**2, ".1f")
# quasi MC (Sobol)
from scipy.stats import qmc
sob = qmc.Sobol(d=1, scramble=True, seed=3).random(2**17).ravel(); Zq = stats.norm.ppf(sob)
STq = S0*np.exp((r-0.5*sig**2)*T + sig*np.sqrt(T)*Zq); V("c13_qmc", np.exp(-r*T)*np.maximum(STq-K, 0).mean(), ".4f")
# LSMC American put
Mp, npth = 100000, 50; dtp = T/npth
Sp = np.empty((Mp, npth+1)); Sp[:, 0] = S0
for k in range(npth): Sp[:, k+1] = Sp[:, k]*np.exp((r-0.5*sig**2)*dtp + sig*np.sqrt(dtp)*rng.normal(size=Mp))
cash = np.maximum(K - Sp[:, -1], 0)
for k in range(npth-1, 0, -1):
    cash *= np.exp(-r*dtp); itm = K - Sp[:, k] > 0
    X = Sp[itm, k]; Y = cash[itm]
    A = np.column_stack([np.ones_like(X), X, X**2, X**3]); coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    cont = A@coef; ex = K - X
    exercise = ex > cont; idx = np.where(itm)[0][exercise]; cash[idx] = ex[exercise]
lsmc = np.exp(-r*dtp)*cash
V("c13_lsmc", lsmc.mean(), ".4f"); V("c13_lsmc_se", lsmc.std()/np.sqrt(Mp), ".4f")
V("c13_am_premium", crr(S0, K, r, sig, T, 2000, "put", True) - P, ".4f")
# convergence figure
f_, ax = fig(8.4, 3.8, ncols=2)
ax[0].plot(ns, tree, color=BLUE, lw=1); ax[0].axhline(C, color=INK, ls="--", lw=1); ax[0].set_xlabel("tree steps n"); ax[0].set_ylabel("call price"); ax[0].set_title("Binomial tree → Black–Scholes (odd/even sawtooth)"); ax[0].set_ylim(C-0.25, C+0.25)
Ms = np.logspace(2, 6, 9).astype(int); se_p, se_a, se_c = [], [], []
for Mm in Ms:
    Zm = rng.normal(size=Mm); STm = S0*np.exp((r-0.5*sig**2)*T + sig*np.sqrt(T)*Zm); pm = np.exp(-r*T)*np.maximum(STm-K, 0)
    STam = S0*np.exp((r-0.5*sig**2)*T - sig*np.sqrt(T)*Zm); pam = 0.5*(pm + np.exp(-r*T)*np.maximum(STam-K, 0))
    cvm = np.exp(-r*T)*STm; pcm = pm - beta_cv*(cvm - S0)
    se_p.append(pm.std()/np.sqrt(Mm)); se_a.append(pam.std()/np.sqrt(Mm)); se_c.append(pcm.std()/np.sqrt(Mm))
ax[1].loglog(Ms, se_p, "o-", color=ORANGE, label="plain"); ax[1].loglog(Ms, se_a, "^-", color=MAGENTA, label="antithetic"); ax[1].loglog(Ms, se_c, "s-", color=AQUA, label="control variate")
ax[1].set_xlabel("paths M"); ax[1].set_ylabel("standard error of price"); ax[1].set_title("Monte Carlo error ∝ 1/√M"); ax[1].legend(fontsize=8.5)
ax[0].set_title("Binomial tree → Black–Scholes")
save(f_, "c13_convergence")
flush()
