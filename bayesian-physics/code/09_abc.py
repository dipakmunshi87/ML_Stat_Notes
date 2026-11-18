"""Worked examples of chapter 9: ABC: approximate Bayesian computation

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 09_abc.py
"""


def example_1():
    """ABC for a Gaussian mean"""
    import numpy as np
    tau, s2, xbar = 2.0, 0.25, 1.2                   # prior width, variance of the mean (sigma^2/n), observed mean
    for eps in (0.0, 0.1, 0.5, 1.0):
        v = s2 + eps**2                              # ABC = exact posterior of a noisier experiment
        mean, sd = xbar * tau**2 / (tau**2 + v), (1 / tau**2 + 1 / v)**-0.5
        V = eps**2 + tau**2 + s2
        kept = np.sqrt(eps**2 / V) * np.exp(-xbar**2 / (2 * V))
        print("eps = %.1f: mean %.3f, sd %.3f, kept %.1f%%" % (eps, mean, sd, 100 * kept))
    rng = np.random.default_rng(0)                   # check eps = 0.5 by doing it
    theta = rng.normal(0, tau, 2_000_000)
    sim = theta + rng.normal(0, np.sqrt(s2), theta.size)
    keep = rng.random(theta.size) < np.exp(-(sim - xbar)**2 / (2 * 0.5**2))
    print("simulated, eps = 0.5: mean %.3f, sd %.3f, kept %.1f%%" % (theta[keep].mean(), theta[keep].std(), 100 * keep.mean()))


def example_2():
    """ABC with zero tolerance, done exactly"""
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(1)
    p = rng.random(1_000_000)                        # flat prior on the probability of heads
    heads = rng.binomial(10, p)                      # simulate 10 tosses for each draw
    kept = p[heads == 7]                             # exact match
    exact = stats.beta(8, 4)
    print("kept %.1f%% (1/11 = %.1f%%)" % (100 * kept.size / p.size, 100 / 11))
    print("ABC: %.3f +/- %.3f    exact: %.3f +/- %.3f" % (kept.mean(), kept.std(), exact.mean(), exact.std()))


def example_3():
    """the price of a tolerance"""
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(1)
    p = rng.random(1_000_000)
    heads = rng.binomial(10, p)
    kept = p[np.abs(heads - 7) <= 1]                 # accept 6, 7 or 8 heads
    exact_sd = stats.beta(8, 4).std()
    print("kept %.1f%%, mean %.3f, sd %.3f, sd / exact = %.2f" % (100 * kept.size / p.size, kept.mean(), kept.std(), kept.std() / exact_sd))


def example_4():
    """removing the regulator by extrapolation"""
    tau, s2, xbar = 2.0, 0.25, 1.2
    mean = lambda eps: xbar * tau**2 / (tau**2 + s2 + eps**2)
    m1, m2 = mean(0.5), mean(0.25)
    extrap = m2 + (m2 - m1) * 0.25**2 / (0.5**2 - 0.25**2)       # straight line in eps^2, read off at eps = 0
    print("eps = 0.5: %.4f   eps = 0.25: %.4f   extrapolated: %.4f   exact: %.4f" % (m1, m2, extrap, mean(0)))
    print("errors: tighter run alone %.3f, extrapolation %.3f" % (mean(0) - m2, mean(0) - extrap))


if __name__ == "__main__":
    print("\n--- ABC for a Gaussian mean ---")
    example_1()
    print("\n--- ABC with zero tolerance, done exactly ---")
    example_2()
    print("\n--- the price of a tolerance ---")
    example_3()
    print("\n--- removing the regulator by extrapolation ---")
    example_4()
