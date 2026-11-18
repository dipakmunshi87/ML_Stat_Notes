"""Inverse PINN for the adhesion model (viscous Burgers), Section 11 of "Kinetic Hierarchies".

   u_t + u u_x = nu u_xx,  x in [-1,1], t in [0,1],  u(x,0) = -sin(pi x), u(+-1,t) = 0.
The network learns u(t,x) AND the unknown viscosity nu from 200 noisy samples.
Exact solution by the Cole-Hopf transform, used only to make data and to score the result."""
import numpy as np, torch, json, time, os
from _plotstyle import setup, save, BLUES, INK, MUTED
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
torch.manual_seed(0); np.random.seed(0)
NU_TRUE = 0.03

# ---------- exact solution (Cole-Hopf) ----------
def exact(x, t, nu=NU_TRUE, n=4001):
    eta = np.linspace(-6, 6, n)[None, :]          # integration variable, in units of sqrt(4 nu t)
    x = np.atleast_1d(x)[:, None]; t = np.atleast_1d(t)[:, None]
    if np.all(t == 0): return -np.sin(np.pi * x[:, 0])
    s = np.sqrt(4 * nu * t)
    y = x - s * eta
    logw = -np.cos(np.pi * y) / (2 * np.pi * nu) - eta**2
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return (-np.sin(np.pi * y) * w).sum(1) / w.sum(1)

# ---------- network ----------
class MLP(torch.nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

u_net = MLP()
log_nu = torch.nn.Parameter(torch.tensor(np.log(0.1)))   # start 3x too large

def residual(t, x):
    t.requires_grad_(True); x.requires_grad_(True)
    u = u_net(t, x)
    g = lambda out, inp: torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]
    u_t, u_x = g(u, t), g(u, x)
    u_xx = g(u_x, x)
    return u_t + u * u_x - torch.exp(log_nu) * u_xx

# ---------- training points ----------
Nf = 4000
tf = torch.rand(Nf, 1); xf = 2 * torch.rand(Nf, 1) - 1               # collocation
x0 = torch.linspace(-1, 1, 200)[:, None]; t0 = torch.zeros_like(x0)    # initial condition
tb = torch.rand(200, 1); xb = torch.sign(torch.rand(200, 1) - .5)      # boundaries x=+-1
td_np = np.random.rand(200); xd_np = 2 * np.random.rand(200) - 1       # sparse "observations"
ud_np = np.array([exact(a, b)[0] for a, b in zip(xd_np, td_np)]) + 0.01 * np.random.randn(200)
td = torch.tensor(td_np, dtype=torch.float32)[:, None]
xd = torch.tensor(xd_np, dtype=torch.float32)[:, None]
ud = torch.tensor(ud_np, dtype=torch.float32)[:, None]

def loss_fn():
    l_pde = residual(tf, xf).pow(2).mean()
    l_ic = (u_net(t0, x0) + torch.sin(np.pi * x0)).pow(2).mean()
    l_bc = u_net(tb, xb).pow(2).mean()
    l_dat = (u_net(td, xd) - ud).pow(2).mean()
    return l_pde + l_ic + l_bc + l_dat, (l_pde, l_ic, l_bc, l_dat)

hist = []
opt = torch.optim.Adam(list(u_net.parameters()) + [log_nu], lr=2e-3)
t_start = time.time()
for it in range(6000):
    opt.zero_grad(); L, parts = loss_fn(); L.backward(); opt.step()
    if it % 100 == 0:
        hist.append((it, L.item(), torch.exp(log_nu).item()))
        if it % 1000 == 0: print(it, L.item(), torch.exp(log_nu).item(), round(time.time() - t_start))
opt2 = torch.optim.LBFGS(list(u_net.parameters()) + [log_nu], max_iter=2000, line_search_fn="strong_wolfe",
                         tolerance_grad=1e-9, tolerance_change=1e-12)
def closure():
    opt2.zero_grad(); L, _ = loss_fn(); L.backward(); return L
for k in range(3):
    opt2.step(closure)
    L, _ = loss_fn(); hist.append((6000 + 2000 * (k + 1), L.item(), torch.exp(log_nu).item()))
    print('lbfgs', k, L.item(), torch.exp(log_nu).item(), round(time.time() - t_start))

# ---------- score ----------
xs = np.linspace(-1, 1, 201)
out = {"nu_true": NU_TRUE, "nu_learned": torch.exp(log_nu).item(), "hist": hist, "slices": {}}
errs = []
for tt in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ue = exact(xs, np.full_like(xs, tt)) if tt > 0 else -np.sin(np.pi * xs)
    with torch.no_grad():
        up = u_net(torch.full((201, 1), tt), torch.tensor(xs, dtype=torch.float32)[:, None]).numpy()[:, 0]
    out["slices"][str(tt)] = {"x": xs.tolist(), "exact": ue.tolist(), "pinn": up.tolist()}
    errs.append(np.linalg.norm(up - ue) / np.linalg.norm(ue))
out["rel_l2"] = float(np.mean(errs)); out["rel_l2_each"] = [float(e) for e in errs]
out["data"] = {"t": td_np.tolist(), "x": xd_np.tolist()}
print("nu learned", out["nu_learned"], "rel L2", out["rel_l2"], errs)
json.dump(out, open(os.path.join(OUT, "pinn_burgers.json"), "w"))
setup()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
for c, tt in zip(BLUES, ["0.25", "0.5", "1.0"]):
    sl = out["slices"][tt]
    ax[0].plot(sl["x"], sl["exact"], color=c, lw=2, label=f"exact, t = {tt}")
    ax[0].plot(sl["x"][::8], sl["pinn"][::8], "o", ms=4, mfc="white", mec=c)
ax[0].set_xlabel("x"); ax[0].set_ylabel("u(t, x)"); ax[0].legend(fontsize=8)
ax[0].set_title("exact (lines) and PINN (circles)", fontsize=9.5, color=INK)
h = np.array(hist)
ax[1].plot(h[:, 0], h[:, 2], color=BLUES[1], lw=2); ax[1].axhline(NU_TRUE, color=MUTED, ls="--", lw=1)
ax[1].set_xlabel("training iteration"); ax[1].set_ylabel("learned viscosity")
ax[1].set_title(f"learned nu = {out['nu_learned']:.4f} (true {NU_TRUE})", fontsize=9.5, color=INK)
save(fig, "pinn_burgers", OUT)
