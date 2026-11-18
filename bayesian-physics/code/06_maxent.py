"""Worked examples of chapter 6: Maximum entropy: where distributions and priors come from

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 06_maxent.py
"""


def example_1():
    """the loaded die"""
    import numpy as np
    from scipy.optimize import brentq
    faces = np.arange(1, 7)
    mean_of = lambda lam: (faces * np.exp(lam * faces)).sum() / np.exp(lam * faces).sum()
    lam = brentq(lambda l: mean_of(l) - 4.5, 0, 2)   # fix the multiplier by the constraint on the mean
    p = np.exp(lam * faces); p /= p.sum()
    print("lambda = %.3f, ratio between faces %.2f" % (lam, np.exp(lam)))
    print("probabilities", p.round(3), " mean %.2f" % (p @ faces))


def example_2():
    """the prior for a noise level"""
    import numpy as np
    rng = np.random.default_rng(0)
    for sigma in (0.5, 2.0):
        x = rng.normal(0, sigma, 2_000_000)
        score = -1 / sigma + x**2 / sigma**3         # derivative of ln N(x; 0, sigma) with respect to sigma
        print("sigma = %.1f: Fisher information %.3f, 2/sigma^2 = %.3f" % (sigma, (score**2).mean(), 2 / sigma**2))
    # Jeffreys prior = sqrt(Fisher) = sqrt(2)/sigma. In terms of v = sigma^2 it is (1/sigma) dsigma/dv = 1/(2v): the same form.


def example_3():
    """a lifetime with a known mean"""
    import numpy as np
    from scipy import stats
    print("entropy of the exponential with mean 2: %.3f" % stats.expon(scale=2).entropy())
    print("entropy of the uniform on [0, 4]:       %.3f" % stats.uniform(0, 4).entropy())


def example_4():
    """temperature from a mean energy"""
    import numpy as np
    p_upper = 0.25                                   # mean energy / level spacing
    beta_eps = np.log((1 - p_upper) / p_upper)
    print("beta * eps = %.3f,  kT = %.2f eps" % (beta_eps, 1 / beta_eps))


if __name__ == "__main__":
    print("\n--- the loaded die ---")
    example_1()
    print("\n--- the prior for a noise level ---")
    example_2()
    print("\n--- a lifetime with a known mean ---")
    example_3()
    print("\n--- temperature from a mean energy ---")
    example_4()
