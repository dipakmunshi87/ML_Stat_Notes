import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# 1) a stiff ODE (Robertson chemical kinetics): implicit solver needed
def robertson(t, y):
    a, b, c = y
    return [-0.04 * a + 1e4 * b * c, 0.04 * a - 1e4 * b * c - 3e7 * b * b, 3e7 * b * b]

for method in ["RK45", "Radau"]:
    sol = solve_ivp(robertson, (0, 40), [1, 0, 0], method=method, rtol=1e-6, atol=1e-10)
    print(f"{method:5s}: {sol.nfev:7d} function evaluations, mass conserved to {abs(sol.y[:, -1].sum() - 1):.1e}")

# 2) non-linear least squares with uncertainties
rng = np.random.default_rng(3)
t = np.linspace(0, 6, 200)
y = 1.0 * np.exp(-0.25 * t) * np.cos(3.0 * t) + rng.normal(0, 0.05, t.size)
f = lambda t, A, g, w: A * np.exp(-g * t) * np.cos(w * t)
p, C = curve_fit(f, t, y, p0=[0.9, 0.3, 3.05], sigma=np.full(t.size, 0.05), absolute_sigma=True)
print("A, gamma, omega =", np.round(p, 3), "+/-", np.round(np.sqrt(np.diag(C)), 3))
