"""Worked examples of chapter 19: Flows: reshaping a Gaussian into anything

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 19_flows.py
"""


def example_1():
    """shift and stretch"""
    import numpy as np
    from scipy import stats
    mu, sigma = 1.0, 0.5
    rng = np.random.default_rng(0)
    theta = mu + sigma * rng.standard_normal(1_000_000)          # push standard Gaussian numbers through the map
    t = 1.3
    rule = stats.norm.pdf((t - mu) / sigma) / sigma               # base density at u, times the Jacobian 1/sigma
    hist = ((theta > t - 0.01) & (theta < t + 0.01)).mean() / 0.02
    print("samples: mean %.3f, sd %.3f" % (theta.mean(), theta.std()))
    print("density at %.1f: rule %.4f, N(mu, sigma) %.4f, histogram %.3f" % (t, rule, stats.norm(mu, sigma).pdf(t), hist))


def example_2():
    """a lognormal from a normal"""
    import numpy as np
    from scipy import stats
    mu, sigma = 0.0, 0.5
    rng = np.random.default_rng(0)
    theta = np.exp(mu + sigma * rng.standard_normal(2_000_000))
    grid = np.linspace(0.01, 5, 50000)
    density = stats.norm.pdf((np.log(grid) - mu) / sigma) / (sigma * grid)      # base density times Jacobian 1/(sigma theta)
    print("median: samples %.3f, formula %.3f" % (np.median(theta), np.exp(mu)))
    print("mean:   samples %.3f, formula %.3f" % (theta.mean(), np.exp(mu + sigma**2 / 2)))
    print("peak:   of the density %.3f, formula %.3f" % (grid[density.argmax()], np.exp(mu - sigma**2)))


def example_3():
    """training the simplest flow"""
    import numpy as np
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    data = rng.normal(1.5, 0.7, 5000)                # examples to fit
    def loss(params):
        mu, log_sigma = params
        return log_sigma + ((data - mu)**2).mean() / (2 * np.exp(2 * log_sigma))
    fit = minimize(loss, [0.0, 0.0])
    print("trained flow:  mu = %.4f, sigma = %.4f" % (fit.x[0], np.exp(fit.x[1])))
    print("sample values: mean %.4f, sd %.4f" % (data.mean(), data.std()))


def example_4():
    """a map that is not invertible"""
    import numpy as np
    from scipy import stats
    theta = 1.0
    density = 2 * stats.norm.pdf(np.sqrt(theta)) / (2 * np.sqrt(theta))      # two branches, each with its Jacobian
    print("density at 1: %.3f   chi-squared(1): %.3f" % (density, stats.chi2(1).pdf(theta)))


def example_5():
    """turning a uniform number into an exponential one"""
    import numpy as np
    rng = np.random.default_rng(0)
    theta = -np.log(1 - rng.random(1_000_000))       # the map applied to uniform numbers
    print("mean %.3f, sd %.3f (exponential: 1, 1);  P(theta > 2) = %.4f (exp(-2) = %.4f)" % (theta.mean(), theta.std(), (theta > 2).mean(), np.exp(-2)))


if __name__ == "__main__":
    print("\n--- shift and stretch ---")
    example_1()
    print("\n--- a lognormal from a normal ---")
    example_2()
    print("\n--- training the simplest flow ---")
    example_3()
    print("\n--- a map that is not invertible ---")
    example_4()
    print("\n--- turning a uniform number into an exponential one ---")
    example_5()
