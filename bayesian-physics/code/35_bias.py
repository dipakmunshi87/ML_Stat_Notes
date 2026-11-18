"""Worked examples of chapter 35: Bias: three different things with one name

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 35_bias.py
"""


def example_1():
    """the best tolerance for a fixed budget"""
    import numpy as np
    from scipy.optimize import minimize_scalar
    tau2, s2, xbar = 4.0, 0.25, 1.2
    mean = lambda e: xbar * tau2 / (tau2 + s2 + e**2)
    var = lambda e: 1 / (1 / tau2 + 1 / (s2 + e**2))
    kept = lambda e: np.sqrt(e**2 / (e**2 + tau2 + s2)) * np.exp(-xbar**2 / (2 * (e**2 + tau2 + s2)))
    for n in (1e3, 1e4, 1e5, 1e6):
        err2 = lambda e: (mean(e) - mean(0))**2 + var(e) / (n * kept(e))         # bias squared + Monte Carlo variance
        e = minimize_scalar(err2, bounds=(1e-3, 2), method="bounded").x
        print("n = %7d: best tolerance %.2f, bias %.3f, total error %.3f" % (n, e, mean(e) - mean(0), np.sqrt(err2(e))))


def example_2():
    """the pull of a prior, and how it fades"""
    import numpy as np
    prior_var = 9.0
    for err in (1.0, 0.1):                           # one measurement, then the mean of a hundred
        w = prior_var / (prior_var + err**2)
        est, sd = 5.0 * w, np.sqrt(w) * err
        print("error %.1f: estimate %.3f +/- %.3f, pulled by %.2f sigma" % (err, est, sd, (5.0 - est) / sd))


def example_3():
    """a biased estimate that is more accurate"""
    for c in (1.0, 0.5):
        variance, bias2 = c**2 * 1.0, ((1 - c) * 1.0)**2
        print("c = %.1f: variance %.2f + bias^2 %.2f = mean squared error %.2f" % (c, variance, bias2, variance + bias2))


if __name__ == "__main__":
    print("\n--- the best tolerance for a fixed budget ---")
    example_1()
    print("\n--- the pull of a prior, and how it fades ---")
    example_2()
    print("\n--- a biased estimate that is more accurate ---")
    example_3()
