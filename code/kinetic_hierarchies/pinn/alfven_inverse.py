"""A damped Alfven wave learned by a PINN: infer the Alfven speed and the resistivity (Section 11).

Linearise incompressible MHD about a uniform field B0 x^ for transverse perturbations u_y(x,t), B_y(x,t),
and write the field in velocity units, b = B_y / sqrt(mu0 rho). With equal viscosity and resistivity eta
(magnetic Prandtl number 1),
    u_t = v_A b_x + eta u_xx,        b_t = v_A u_x + eta b_xx,        v_A = B0 / sqrt(mu0 rho).
For u(x,0) = sin x, b(x,0) = 0 on a periodic domain the exact solution is a standing, decaying Alfven wave
    u = exp(-eta t) sin x cos(v_A t),     b = exp(-eta t) cos x sin(v_A t).

Inverse problem: v_A and eta are UNKNOWN. From 300 noisy samples of u and b the network learns both fields
and both parameters. Periodicity is built in exactly by feeding the network (t, sin x, cos x).

Run:  python alfven_inverse.py        (about 3-5 minutes on a laptop CPU)
Writes ../figures/pinn_alfven.{png,svg} and ../figures/pinn_alfven.json
"""
import json, os, time
import numpy as np
import torch
from _plotstyle import setup, save, BLUES, INK, MUTED

torch.manual_seed(2); np.random.seed(2)
VA_TRUE, ETA_TRUE, T_END = 1.0, 0.1, 6.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def exact(x, t):
    a = np.exp(-ETA_TRUE * t)
    return a * np.sin(x) * np.cos(VA_TRUE * t), a * np.cos(x) * np.sin(VA_TRUE * t)


class Net(torch.nn.Module):
    def __init__(self, w=48, d=4):
        super().__init__()
        layers, n = [], 3
        for _ in range(d):
            layers += [torch.nn.Linear(n, w), torch.nn.Tanh()]; n = w
        self.f = torch.nn.Sequential(*layers, torch.nn.Linear(n, 2))

    def forward(self, t, x):                    # periodic in x by construction
        o = self.f(torch.cat([t / T_END * 2 - 1, torch.sin(x), torch.cos(x)], 1))
        return o[:, :1], o[:, 1:]


net = Net()
log_va = torch.nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))
log_eta = torch.nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32))
d = lambda y, z: torch.autograd.grad(y, z, torch.ones_like(y), create_graph=True)[0]


def residual(t, x):
    t = t.clone().requires_grad_(True); x = x.clone().requires_grad_(True)
    u, b = net(t, x)
    va, eta = torch.exp(log_va), torch.exp(log_eta)
    ux, bx = d(u, x), d(b, x)
    return d(u, t) - va * bx - eta * d(ux, x), d(b, t) - va * ux - eta * d(bx, x)


Nc = 4000
tc = T_END * torch.rand(Nc, 1); xc = 2 * np.pi * torch.rand(Nc, 1)
x0 = torch.linspace(0, 2 * np.pi, 128)[:, None]; t0_ = torch.zeros_like(x0)
td_np = T_END * np.random.rand(300); xd_np = 2 * np.pi * np.random.rand(300)
ud_np, bd_np = exact(xd_np, td_np)
ud_np = ud_np + 0.01 * np.random.randn(300); bd_np = bd_np + 0.01 * np.random.randn(300)
T = lambda a: torch.tensor(a, dtype=torch.float32)[:, None]
td, xd, ud, bd = T(td_np), T(xd_np), T(ud_np), T(bd_np)


def loss():
    ru, rb = residual(tc, xc)
    u0, b0 = net(t0_, x0); um, bm = net(td, xd)
    return (ru.pow(2).mean() + rb.pow(2).mean()
            + (u0 - torch.sin(x0)).pow(2).mean() + b0.pow(2).mean()
            + (um - ud).pow(2).mean() + (bm - bd).pow(2).mean())


hist = []
params = list(net.parameters()) + [log_va, log_eta]
opt = torch.optim.Adam(params, lr=2e-3)
t0 = time.time()
for it in range(5000):
    opt.zero_grad(); L = loss(); L.backward(); opt.step()
    if it % 50 == 0:
        hist.append((it, L.item(), torch.exp(log_va).item(), torch.exp(log_eta).item()))
    if it % 1000 == 0:
        print(f"adam {it:5d}  loss {L.item():.3e}  vA {torch.exp(log_va).item():.4f}  eta {torch.exp(log_eta).item():.4f}  {time.time()-t0:.0f}s")
lb = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)
def closure():
    lb.zero_grad(); L = loss(); L.backward(); return L
for k in range(5):
    lb.step(closure)
    hist.append((5000 + 200 * (k + 1), loss().item(), torch.exp(log_va).item(), torch.exp(log_eta).item()))
va, eta = torch.exp(log_va).item(), torch.exp(log_eta).item()

xs = np.linspace(0, 2 * np.pi, 201); times = [0.0, 1.5, 3.0, 4.5]
errs = []
setup()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
for j, tt in enumerate(times):
    ue, be = exact(xs, np.full_like(xs, tt))
    with torch.no_grad():
        up, bp = net(torch.full((201, 1), tt), T(xs))
    up, bp = up.numpy()[:, 0], bp.numpy()[:, 0]
    errs.append(float(np.sqrt(np.mean((up - ue) ** 2 + (bp - be) ** 2))))
    col = plt.cm.Blues(0.45 + 0.17 * j)
    ax[0].plot(xs, ue, color="#b8c2cc", lw=2.4); ax[0].plot(xs, up, "--", color=col, lw=1.4, label=f"t = {tt}")
    ax[1].plot(xs, be, color="#b8c2cc", lw=2.4); ax[1].plot(xs, bp, "--", color=col, lw=1.4)
ax[0].set_title("velocity u: exact (grey), PINN (dashed)", fontsize=9.5, color=INK); ax[0].set_xlabel("x"); ax[0].legend(fontsize=7.5, loc="upper right")
ax[1].set_title(r"field $b = B_y/\sqrt{\mu_0\rho}$", fontsize=9.5, color=INK); ax[1].set_xlabel("x")
h = np.array(hist)
ax[2].plot(h[:, 0], h[:, 2], color=BLUES[1], lw=2, label=r"Alfvén speed $v_A$")
ax[2].plot(h[:, 0], h[:, 3] * 10, color=BLUES[2], lw=2, ls="-.", label=r"magnetic diffusivity $\eta\times10$")
ax[2].axhline(VA_TRUE, color=MUTED, ls="--", lw=1)
ax[2].set_xlabel("training iteration"); ax[2].set_title(rf"learned $v_A$ = {va:.4f}, $\eta$ = {eta:.4f}", fontsize=9.5, color=INK)
ax[2].legend(fontsize=7.5, loc="lower right")
save(fig, "pinn_alfven", OUT)
print(f"vA learned {va:.4f} (true {VA_TRUE}), eta learned {eta:.4f} (true {ETA_TRUE}), rms error {np.mean(errs):.2e}, {time.time()-t0:.0f}s")
json.dump({"va_true": VA_TRUE, "eta_true": ETA_TRUE, "va_learned": va, "eta_learned": eta, "rms_err": float(np.mean(errs)), "hist": hist},
          open(os.path.join(OUT, "pinn_alfven.json"), "w"))
