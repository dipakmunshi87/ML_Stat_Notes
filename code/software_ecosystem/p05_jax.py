import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# data: damped oscillator y = A exp(-g t) cos(w t) + noise
rng = np.random.default_rng(0)
t = jnp.linspace(0, 6, 400)
y = 1.0 * jnp.exp(-0.25 * t) * jnp.cos(3.0 * t) + 0.05 * rng.normal(size=t.size)

def model(theta, t):
    A, g, w = theta
    return A * jnp.exp(-g * t) * jnp.cos(w * t)

def chi2(theta):
    return jnp.sum((y - model(theta, t)) ** 2) / 0.05**2

def residuals(theta):
    return (y - model(theta, t)) / 0.05

jac = jax.jit(jax.jacfwd(residuals))     # exact Jacobian by automatic differentiation, compiled by XLA
hess = jax.jit(jax.hessian(chi2))        # exact Hessian of chi^2 -> Fisher matrix = Hessian / 2

theta = jnp.array([0.9, 0.3, 3.05])
for _ in range(10):                      # Gauss-Newton: solve (J^T J) step = -J^T r
    J, r = jac(theta), residuals(theta)
    theta = theta - jnp.linalg.solve(J.T @ J, J.T @ r)
cov = jnp.linalg.inv(hess(theta) / 2)
print("best fit  A, gamma, omega =", np.round(np.asarray(theta), 4))
print("1-sigma errors            =", np.round(np.sqrt(np.diag(np.asarray(cov))), 4))

# vmap: evaluate chi2 for 10,000 parameter vectors at once, no Python loop
thetas = jnp.stack([jnp.linspace(0.5, 1.5, 10_000), jnp.full(10_000, 0.25), jnp.full(10_000, 3.0)], axis=1)
scan = jax.vmap(chi2)(thetas)
print("best A on the grid        =", float(thetas[jnp.argmin(scan), 0]).__round__(4), "| device:", jax.devices()[0].platform)
