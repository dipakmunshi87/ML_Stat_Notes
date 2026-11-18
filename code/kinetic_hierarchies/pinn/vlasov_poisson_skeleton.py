"""Skeleton of a Vlasov-Poisson PINN (Section 11): 1D-1V electrons on a fixed ion background.

    f_t + v f_x - E f_v = 0,        E_x = 1 - int f dv          (plasma units)

Two networks, f_theta(t,x,v) and E_theta(t,x); the density in Gauss's law is a Gauss-Legendre quadrature of
f_theta over fixed v-nodes. This is the code shown on the page: it runs and trains, but for brevity it omits
the initial-condition and periodic-boundary losses a real Landau-damping or two-stream run needs. Add them
(e.g. f(0,x,v) = (1 + 0.01 cos 0.5x) exp(-v^2/2)/sqrt(2 pi), periodic features sin/cos of x) to experiment.

Run:  python vlasov_poisson_skeleton.py --steps 200
"""
import argparse
import torch, numpy as np
# 1D-1V electrostatic Vlasov-Poisson for electrons on a fixed ion background (plasma units):
#   f_t + v f_x - E f_v = 0,     E_x = 1 - \int f dv
mlp = lambda n_in: torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
f_net, E_net = mlp(3), mlp(2)                       # f(t,x,v) and E(t,x): two networks
vq, wq = np.polynomial.legendre.leggauss(64)        # quadrature in v on [-6, 6]
vq = torch.tensor(6 * vq, dtype=torch.float32); wq = torch.tensor(6 * wq, dtype=torch.float32)
d = lambda y, z: torch.autograd.grad(y, z, torch.ones_like(y), create_graph=True)[0]

def residuals(t, x, v):
    t.requires_grad_(True); x.requires_grad_(True); v.requires_grad_(True)
    f = f_net(torch.cat([t, x, v], 1)); E = E_net(torch.cat([t, x], 1))
    r_vlasov = d(f, t) + v * d(f, x) - E * d(f, v)
    # density n(t,x) = \int f dv by Gauss-Legendre: evaluate f on the v-nodes at each (t,x)
    tq, xq = t.repeat_interleave(64, 0), x.repeat_interleave(64, 0)
    fq = f_net(torch.cat([tq, xq, vq.repeat(len(t))[:, None]], 1)).view(-1, 64)
    n = (fq * wq).sum(1, keepdim=True)
    r_poisson = d(E, x) - (1 - n)
    return r_vlasov, r_poisson

# one training step (initial and periodic-boundary losses omitted for brevity)
opt = torch.optim.Adam(list(f_net.parameters()) + list(E_net.parameters()), lr=1e-3)
ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=50); args = ap.parse_args()
for step in range(args.steps):
    t = torch.rand(512, 1) * 20; x = torch.rand(512, 1) * 4 * np.pi; v = (torch.rand(512, 1) - .5) * 12
    rv, rp = residuals(t, x, v)
    loss = rv.pow(2).mean() + rp.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0: print(step, loss.item())
