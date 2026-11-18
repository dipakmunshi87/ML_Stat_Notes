"""Equilibrium reconstruction with a PINN: the Grad-Shafranov equation (Section 11).

The axisymmetric MHD equilibrium of a tokamak obeys
    Delta* psi = R d/dR (1/R dpsi/dR) + d2psi/dZ2 = -mu0 R^2 p'(psi) - F F'(psi).
With p' constant and FF' = 0 (a Solov'ev equilibrium) the right-hand side is c R^2, and
    psi(R, Z) = (R^2 - 1)^2 / 8 + (E/2) R^2 Z^2,   c = 1 + E
is an exact solution with nested, elongated flux surfaces around the magnetic axis (R, Z) = (1, 0).

Inverse problem (what EFIT-style reconstruction does): the pressure-gradient constant c is UNKNOWN.
We know psi on the boundary of the box (as from flux loops) and 40 noisy interior samples, and train a
network psi_theta(R, Z) together with c so that the Grad-Shafranov residual vanishes everywhere.

Run:  python grad_shafranov_inverse.py      (about 2-4 minutes on a laptop CPU)
Writes ../figures/pinn_grad_shafranov.{png,svg} and ../figures/pinn_grad_shafranov.json
"""
import json, os, time
import numpy as np
import torch
from _plotstyle import setup, save, BLUES, INK, MUTED

torch.manual_seed(1); np.random.seed(1)
E_TRUE = 0.8
C_TRUE = 1.0 + E_TRUE
R0, R1, Z0, Z1 = 0.6, 1.4, -0.6, 0.6
SCALE = 0.05                                   # psi is O(0.05); the network predicts psi / SCALE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def psi_exact(R, Z):
    return (R**2 - 1) ** 2 / 8 + 0.5 * E_TRUE * R**2 * Z**2


class Net(torch.nn.Module):
    def __init__(self, w=40, d=4):
        super().__init__()
        layers, n = [], 2
        for _ in range(d):
            layers += [torch.nn.Linear(n, w), torch.nn.Tanh()]; n = w
        self.f = torch.nn.Sequential(*layers, torch.nn.Linear(n, 1))

    def forward(self, R, Z):                    # inputs rescaled to [-1, 1]
        r = (R - 1.0) / 0.4; z = Z / 0.6
        return SCALE * self.f(torch.cat([r, z], 1))


net = Net()
log_c = torch.nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))   # start far from 1.8
d = lambda y, x: torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residual(R, Z):
    R = R.clone().requires_grad_(True); Z = Z.clone().requires_grad_(True)
    psi = net(R, Z)
    pR, pZ = d(psi, R), d(psi, Z)
    return d(pR, R) - pR / R + d(pZ, Z) - torch.exp(log_c) * R**2       # Delta* psi - c R^2


# collocation points, boundary data, interior measurements
Nc = 3000
Rc = R0 + (R1 - R0) * torch.rand(Nc, 1); Zc = Z0 + (Z1 - Z0) * torch.rand(Nc, 1)
s = torch.linspace(0, 1, 60)[:, None]
Rb = torch.cat([R0 + (R1 - R0) * s, R0 + (R1 - R0) * s, torch.full_like(s, R0), torch.full_like(s, R1)])
Zb = torch.cat([torch.full_like(s, Z0), torch.full_like(s, Z1), Z0 + (Z1 - Z0) * s, Z0 + (Z1 - Z0) * s])
psib = psi_exact(Rb, Zb)
Rd_np = R0 + (R1 - R0) * np.random.rand(40); Zd_np = Z0 + (Z1 - Z0) * np.random.rand(40)
psid_np = psi_exact(Rd_np, Zd_np) + 0.01 * 0.05 * np.random.randn(40)        # 1% of the psi range
Rd = torch.tensor(Rd_np, dtype=torch.float32)[:, None]; Zd = torch.tensor(Zd_np, dtype=torch.float32)[:, None]
psid = torch.tensor(psid_np, dtype=torch.float32)[:, None]


def loss():
    w = 1.0 / SCALE**2
    return (residual(Rc, Zc).pow(2).mean()
            + w * (net(Rb, Zb) - psib).pow(2).mean()
            + w * (net(Rd, Zd) - psid).pow(2).mean())


hist = []
params = list(net.parameters()) + [log_c]
opt = torch.optim.Adam(params, lr=2e-3)
t0 = time.time()
for it in range(4000):
    opt.zero_grad(); L = loss(); L.backward(); opt.step()
    if it % 50 == 0:
        hist.append((it, L.item(), torch.exp(log_c).item()))
    if it % 1000 == 0:
        print(f"adam {it:5d}  loss {L.item():.3e}  c {torch.exp(log_c).item():.4f}  {time.time()-t0:.0f}s")
lb = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)
def closure():
    lb.zero_grad(); L = loss(); L.backward(); return L
for k in range(6):
    lb.step(closure)
    hist.append((4000 + 200 * (k + 1), loss().item(), torch.exp(log_c).item()))
c_fit = torch.exp(log_c).item()

# score on a grid
Rg, Zg = np.meshgrid(np.linspace(R0, R1, 121), np.linspace(Z0, Z1, 181))
with torch.no_grad():
    P = net(torch.tensor(Rg.ravel(), dtype=torch.float32)[:, None],
            torch.tensor(Zg.ravel(), dtype=torch.float32)[:, None]).numpy().reshape(Rg.shape)
Pe = psi_exact(Rg, Zg)
rel = float(np.linalg.norm(P - Pe) / np.linalg.norm(Pe))
print(f"c learned {c_fit:.4f} (true {C_TRUE}),  relative L2 error in psi {rel:.2e},  {time.time()-t0:.0f}s")

# figure
setup()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})
lev = np.linspace(0.001, Pe.max(), 9)
ax[0].contour(Rg, Zg, Pe, levels=lev, colors="#b8c2cc", linewidths=2.2)
ax[0].contour(Rg, Zg, P, levels=lev, colors=BLUES[2], linewidths=1.1, linestyles="--")
ax[0].plot(Rd_np, Zd_np, "o", ms=3.5, mfc="white", mec=BLUES[1], mew=1.1, label="flux measurements")
ax[0].plot([1], [0], "+", color=INK, ms=9); ax[0].set_aspect("equal")
ax[0].set_xlabel("R"); ax[0].set_ylabel("Z")
ax[0].set_title("flux surfaces: exact (grey) vs PINN (dashed)", fontsize=9.5, color=INK)
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8)
h = np.array(hist)
ax[1].plot(h[:, 0], h[:, 2], color=BLUES[1], lw=2)
ax[1].axhline(C_TRUE, color=MUTED, ls="--", lw=1.2)
ax[1].text(h[-1, 0], C_TRUE - 0.13, f"true c = {C_TRUE:.2f}   learned c = {c_fit:.4f}", ha="right", color=INK, fontsize=8.5)
ax[1].set_xlabel("training iteration"); ax[1].set_ylabel("pressure-gradient constant c")
ax[1].set_title("reconstructing p′ from 40 noisy flux samples", fontsize=9.5, color=INK)
save(fig, "pinn_grad_shafranov", OUT)
json.dump({"c_true": C_TRUE, "c_learned": c_fit, "rel_l2_psi": rel, "hist": hist},
          open(os.path.join(OUT, "pinn_grad_shafranov.json"), "w"))
