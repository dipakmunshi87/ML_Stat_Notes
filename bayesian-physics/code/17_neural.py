"""Worked examples of chapter 17: Neural methods: NPE, NLE and NRE

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 17_neural.py
"""


def example_1():
    """NPE with the simplest possible network"""
    import numpy as np
    rng = np.random.default_rng(0)
    tau, s2, N = 2.0, 0.25, 1_000_000
    theta = rng.normal(0, tau, N)                    # parameters from the prior
    x = theta + rng.normal(0, np.sqrt(s2), N)        # simulated data
    a = (theta * x).mean() / (x * x).mean()          # least-squares slope: the loss minimised over a
    v = ((theta - a * x)**2).mean()                  # leftover scatter: the loss minimised over v
    print("learned from simulations: a = %.3f, sqrt(v) = %.3f" % (a, np.sqrt(v)))
    print("exact posterior:          a = %.3f, sqrt(v) = %.3f" % (tau**2 / (tau**2 + s2), np.sqrt(tau**2 * s2 / (tau**2 + s2))))


def example_2():
    """what the best classifier knows"""
    import numpy as np
    from scipy import stats
    tau2, s2 = 4.0, 0.25
    theta, x = 0.7, 1.2
    direct = stats.norm(theta, np.sqrt(s2)).logpdf(x) - stats.norm(0, np.sqrt(tau2 + s2)).logpdf(x)
    formula = -(x - theta)**2 / (2 * s2) + x**2 / (2 * (tau2 + s2)) + 0.5 * np.log((tau2 + s2) / s2)
    d = 1 / (1 + np.exp(-direct))                    # what a perfect classifier would output
    print("ln r: from the densities %.4f, from the quadratic formula %.4f" % (direct, formula))
    print("classifier output d = %.3f, odds d/(1-d) = %.3f = r" % (d, d / (1 - d)))


def example_3():
    """from a classifier's output to a posterior"""
    prior_density = 0.1                              # flat between 0 and 10
    for d in (0.8, 0.5, 0.1):
        ratio = d / (1 - d)
        print("classifier output %.1f: ratio %.3f, posterior density %.3f" % (d, ratio, ratio * prior_density))


def example_4():
    """why few training points give narrow error bars"""
    import numpy as np
    rng = np.random.default_rng(0)
    for N in (10, 50, 200):
        samples = rng.standard_normal((200_000, N))
        fitted_var = samples.var(axis=1).mean()      # maximum-likelihood variance, averaged over repeats
        print("N = %3d: fitted width %.3f (formula %.3f)" % (N, np.sqrt(fitted_var), np.sqrt((N - 1) / N)))


if __name__ == "__main__":
    print("\n--- NPE with the simplest possible network ---")
    example_1()
    print("\n--- what the best classifier knows ---")
    example_2()
    print("\n--- from a classifier's output to a posterior ---")
    example_3()
    print("\n--- why few training points give narrow error bars ---")
    example_4()
