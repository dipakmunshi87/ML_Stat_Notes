"""Worked examples of chapter 21: Scores and diffusion models

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 21_diffusion.py
"""


def example_1():
    """one denoising step for a Gaussian"""
    m, v = 1.2 * 4 / 4.25, 1 / (1 / 4 + 1 / 0.25)    # the posterior of the ABC chapter: 1.129, 0.235
    noise, theta_t = 1.0, 3.0
    score = -(theta_t - m) / (v + noise**2)
    print("score %.3f, denoised estimate %.3f" % (score, theta_t + noise**2 * score))
    print("Bayes' rule:              %.3f" % (m + v / (v + noise**2) * (theta_t - m)))


def example_2():
    """shift and stretch, by diffusion"""
    import numpy as np
    rng = np.random.default_rng(0)
    mu, sigma, s_max, n = 1.0, 0.5, 20.0, 20_000
    u = rng.standard_normal(n)
    theta = mu + np.sqrt(sigma**2 + s_max**2) * u    # start from the fully blurred distribution
    levels = np.geomspace(s_max, 1e-4, 20000)
    for s, s_next in zip(levels[:-1], levels[1:]):   # lower the noise, moving each sample along the rule
        mid = 0.5 * (s + s_next)
        theta += (s_next - s) * mid * (theta - mu) / (sigma**2 + mid**2)
    print("after the diffusion: mean %.3f, sd %.3f (target %.1f, %.1f)" % (theta.mean(), theta.std(), mu, sigma))
    print("largest difference from the flow map mu + sigma*u: %.4f" % np.abs(theta - (mu + sigma * u)).max())


def example_3():
    """what the network is trained on"""
    import numpy as np
    rng = np.random.default_rng(0)
    mu, sigma, s, n = 1.0, 0.5, 0.8, 2_000_000
    clean = rng.normal(mu, sigma, n)
    noisy = clean + s * rng.standard_normal(n)
    target = -(noisy - clean) / s**2                 # what the network is asked to predict
    k = -np.cov(target, noisy)[0, 1] / noisy.var()   # least-squares slope of the target against the noisy value
    print("fitted k = %.4f, true 1/(sigma^2 + s^2) = %.4f" % (k, 1 / (sigma**2 + s**2)))


def example_4():
    """the lognormal"""
    import numpy as np
    mu, sigma = 0.0, 0.5
    score0 = lambda t: -1 / t - (np.log(t) - mu) / (sigma**2 * t)        # score at vanishing noise
    print("score at the peak 0.779: %.4f;  at 0.3: %+.1f;  at 3: %+.2f" % (score0(np.exp(mu - sigma**2)), score0(0.3), score0(3.0)))
    rng = np.random.default_rng(0)
    clean = np.exp(mu + sigma * rng.standard_normal(2_000_000))
    s = 0.3
    for theta in (-0.2, 0.8, 2.5):                   # score at noise level s: (mean clean value given theta - theta) / s^2
        w = np.exp(-(theta - clean)**2 / (2 * s**2))
        print("noise 0.3, theta = %+.1f: mean clean value %.2f, score %+.2f" % (theta, (w * clean).sum() / w.sum(), ((w * clean).sum() / w.sum() - theta) / s**2))


def example_5():
    """the heat equation sets the noise level"""
    import numpy as np
    D, v, m = 0.5, 0.2, 1.0
    x = np.linspace(-10, 12, 2201); dx = x[1] - x[0]; dt = 0.2 * dx**2 / D
    p = np.exp(-(x - m)**2 / (2 * v)) / np.sqrt(2 * np.pi * v)   # start: a Gaussian of variance v
    t = 0.0
    for target in (0.5, 2.0):
        while t < target - 1e-12:                    # explicit finite-difference solution of dp/dt = D d2p/dx2
            p[1:-1] += D * dt / dx**2 * (p[2:] - 2 * p[1:-1] + p[:-2]); t += dt
        mean = np.trapezoid(x * p, x); var = np.trapezoid((x - mean)**2 * p, x)
        print("t = %.1f: mean %.3f, variance %.3f (v + 2Dt = %.3f)" % (t, mean, var, v + 2 * D * t))


if __name__ == "__main__":
    print("\n--- one denoising step for a Gaussian ---")
    example_1()
    print("\n--- shift and stretch, by diffusion ---")
    example_2()
    print("\n--- what the network is trained on ---")
    example_3()
    print("\n--- the lognormal ---")
    example_4()
    print("\n--- the heat equation sets the noise level ---")
    example_5()
