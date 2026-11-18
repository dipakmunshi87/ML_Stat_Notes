import torch
from torch import nn

torch.manual_seed(0)
# learn the 1D pendulum period T(theta0)/T0 from samples (a smooth physics function)
theta0 = torch.rand(2000, 1) * 2.5                       # amplitude in radians
k = torch.sin(theta0 / 2)
# exact ratio via the arithmetic-geometric mean: T/T0 = 1 / AGM(1, sqrt(1-k^2))
a, b = torch.ones_like(k), torch.sqrt(1 - k**2)
for _ in range(8):
    a, b = (a + b) / 2, torch.sqrt(a * b)
ratio = 1 / a

net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
train, test = slice(0, 1600), slice(1600, 2000)
for epoch in range(4000):
    opt.zero_grad()
    loss = nn.functional.mse_loss(net(theta0[train]), ratio[train])
    loss.backward()                                      # autograd computes all gradients
    opt.step()
with torch.no_grad():
    err = (net(theta0[test]) - ratio[test]).abs().max().item()
print(f"train MSE {loss.item():.2e}, worst test error {err:.1e} (ratio ranges 1 to {ratio.max().item():.2f})")
