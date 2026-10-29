"""Hartmann flow learned by a PINN: infer the Hartmann number from a velocity profile (Section 11).

A conducting fluid driven along a channel -1 < y < 1 by a pressure gradient, with a uniform magnetic field
across the channel, obeys (in units of the Poiseuille scale)
    u''(y) - Ha^2 u(y) + 1 = 0,     u(+-1) = 0,     Ha = B0 L sqrt(sigma / (rho nu)),
whose exact solution  u = (1 - cosh(Ha y) / cosh Ha) / Ha^2  is flat in the core with thin Hartmann layers
of thickness ~ 1/Ha at the walls: the Lorentz force flattens the parabolic Poiseuille profile.

Inverse problem: Ha is UNKNOWN. From 25 noisy velocity samples the PINN learns the profile and Ha, i.e. it
measures the field strength from the flow. The boundary layers are the classic hard case for PINNs.

Run:  python hartmann_inverse.py      (under a minute)
Writes ../figures/pinn_hartmann.{png,svg} and ../figures/pinn_hartmann.json
"""
import json, os, time
import numpy as np
import torch
from _plotstyle import setup, save, BLUES, INK, MUTED

torch.manual_seed(3); np.random.seed(3)
HA_TRUE = 6.0
SCALE = 0.03                                        # the network predicts u / SCALE (u_max ~ 0.028)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
exact = lambda y, Ha=HA_TRUE: (1 - np.cosh(Ha * y) / np.cosh(Ha)) / Ha**2

net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                          torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
u = lambda y: SCALE * (1 - y**2) * net(y)            # u(+-1) = 0 imposed exactly
log_ha = torch.nn.Parameter(torch.tensor(np.log(2.0), dtype=torch.float32))
d = lambda a, b: torch.autograd.grad(a, b, torch.ones_like(a), create_graph=True)[0]


def residual(y):
    y = y.clone().requires_grad_(True); uu = u(y)
    return d(d(uu, y), y) - torch.exp(log_ha) ** 2 * uu + 1


yc = torch.linspace(-1, 1, 400)[:, None]
yd_np = np.sort(2 * np.random.rand(25) - 1)
ud_np = exact(yd_np) + 0.01 * exact(0.0) * np.random.randn(25)
yd = torch.tensor(yd_np, dtype=torch.float32)[:, None]; ud = torch.tensor(ud_np, dtype=torch.float32)[:, None]
loss = lambda: residual(yc).pow(2).mean() + (u(yd) - ud).pow(2).mean() / SCALE**2

hist = []; params = list(net.parameters()) + [log_ha]
opt = torch.optim.Adam(params, lr=3e-3); t0 = time.time()
for it in range(3000):
    opt.zero_grad(); L = loss(); L.backward(); opt.step()
    if it % 25 == 0:
        hist.append((it, L.item(), torch.exp(log_ha).item()))
lb = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)
def closure():
    lb.zero_grad(); L = loss(); L.backward(); return L
for k in range(4):
    lb.step(closure); hist.append((3000 + 200 * (k + 1), loss().item(), torch.exp(log_ha).item()))
ha = torch.exp(log_ha).item()

ys = np.linspace(-1, 1, 401)
with torch.no_grad():
    up = u(torch.tensor(ys, dtype=torch.float32)[:, None]).numpy()[:, 0]
rel = float(np.linalg.norm(up - exact(ys)) / np.linalg.norm(exact(ys)))
print(f"Ha learned {ha:.4f} (true {HA_TRUE}), relative L2 error {rel:.2e}, {time.time()-t0:.0f}s")

setup()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
pois = (1 - ys**2) / 2
ax[0].plot(ys, pois / pois.max(), color="#d4dbe2", lw=2, label="Poiseuille (Ha = 0)")
ax[0].plot(ys, exact(ys) / exact(0.0), color="#b8c2cc", lw=2.6, label=f"exact, Ha = {HA_TRUE:g}")
ax[0].plot(ys, up / exact(0.0), "--", color=BLUES[2], lw=1.5, label="PINN")
ax[0].plot(yd_np, ud_np / exact(0.0), "o", ms=3.5, mfc="white", mec=BLUES[1], label="noisy samples")
ax[0].set_xlabel("y (across the channel)"); ax[0].set_ylabel("u / u_centre")
ax[0].set_title("flattened profile with Hartmann layers", fontsize=9.5, color=INK); ax[0].legend(fontsize=7.5, loc="lower center")
h = np.array(hist)
ax[1].plot(h[:, 0], h[:, 2], color=BLUES[1], lw=2); ax[1].axhline(HA_TRUE, color=MUTED, ls="--", lw=1)
ax[1].text(h[-1, 0], HA_TRUE - 0.35, f"true Ha = {HA_TRUE:g}   learned Ha = {ha:.3f}", ha="right", fontsize=8.5, color=INK)
ax[1].set_xlabel("training iteration"); ax[1].set_ylabel("Hartmann number")
ax[1].set_title("measuring the field from the flow", fontsize=9.5, color=INK)
save(fig, "pinn_hartmann", OUT)
json.dump({"ha_true": HA_TRUE, "ha_learned": ha, "rel_l2": rel, "hist": hist}, open(os.path.join(OUT, "pinn_hartmann.json"), "w"))
