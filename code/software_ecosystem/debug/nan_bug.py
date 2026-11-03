import numpy as np

def log_likelihood(sigma, residuals):
    return -0.5 * np.sum(residuals**2) / sigma**2 - residuals.size * np.log(sigma)

rng = np.random.default_rng(0)
r = rng.normal(size=100)
sigmas = np.linspace(-0.5, 2.0, 6)        # bug: includes a negative "standard deviation"

np.seterr(all="raise")                    # turn silent NaN/inf into exceptions
for s in sigmas:
    print(f"sigma={s:+.2f}  logL={log_likelihood(s, r):.2f}")
