"""Worked examples of chapter 15: Importance sampling and the work of changing a distribution

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 15_importance.py
"""


def example_1():
    """evidence from prior samples"""
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)
    tau2, s2, x = 4.0, 0.25, 1.2
    Z = stats.norm(0, np.sqrt(tau2 + s2)).pdf(x)
    theta = rng.normal(0, np.sqrt(tau2), 2_000_000)  # draw from the prior
    L = stats.norm(theta, np.sqrt(s2)).pdf(x)        # weight by the likelihood
    print("exact Z = %.4f, ln Z = %.3f;  from prior samples Z = %.4f" % (Z, np.log(Z), L.mean()))
    print("effective fraction of the samples: %.2f" % (L.mean()**2 / (L**2).mean()))


def example_2():
    """how many rungs?"""
    import numpy as np
    from scipy import stats
    tau2, s2, x = 4.0, 0.25, 1.2
    def mean_lnL(beta):                              # average log-likelihood at inverse temperature beta
        prec = beta / s2 + 1 / tau2
        mu, var = (beta * x / s2) / prec, 1 / prec
        return -0.5 * np.log(2 * np.pi * s2) - 0.5 * ((x - mu)**2 + var) / s2
    lnZ = stats.norm(0, np.sqrt(tau2 + s2)).logpdf(x)
    for K in (1, 2, 10, 100, 1000):
        betas = np.arange(K) / K                     # each rung is sampled, then its weight to the next is added
        work = -sum(mean_lnL(b) / K for b in betas)
        print("%4d rungs: average work %.2f, excess over -ln Z (%.3f) = %.3f" % (K, work, -lnZ, work + lnZ))


def example_3():
    """a proposal that is too narrow is fatal"""
    import numpy as np
    rng = np.random.default_rng(0)
    for s in (2.0, 1.0, 0.8, 0.72):
        x = s * rng.standard_normal(2_000_000)               # proposal of width s
        w = np.exp(-x**2 / 2 + x**2 / (2 * s**2))            # target / proposal, up to a constant
        print("s = %.2f: effective fraction %.2f (formula %.2f)" % (s, w.sum()**2 / (w**2).sum() / w.size, np.sqrt(2*s*s - 1) / s**2))
    # At s = 0.72 the sampled value disagrees with the formula and changes from run to run:
    # the variance of the weights is almost infinite, which is the failure the example describes.


def example_4():
    """Jarzynski's equality for a shifted spring"""
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2_000_000)               # equilibrium positions, stiffness 1, temperature 1
    W = 0.5 * (x - 2)**2 - 0.5 * x**2                # work done when the well jumps by 2
    print("mean work %.2f,  <exp(-W)> = %.3f,  fraction with W < 0: %.1f%%" % (W.mean(), np.exp(-W).mean(), 100 * (W < 0).mean()))


if __name__ == "__main__":
    print("\n--- evidence from prior samples ---")
    example_1()
    print("\n--- how many rungs? ---")
    example_2()
    print("\n--- a proposal that is too narrow is fatal ---")
    example_3()
    print("\n--- Jarzynski's equality for a shifted spring ---")
    example_4()
