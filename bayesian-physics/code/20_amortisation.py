"""Worked examples of chapter 20: Amortisation: train once, infer many times

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 20_amortisation.py
"""


def example_1():
    """when does amortisation pay?"""
    training, per_run = 100_000, 5_000
    print("break-even at %d data sets" % (training // per_run))
    for M in (1, 20, 1000):
        print("%4d data sets: sampler %9d calls, network %d calls" % (M, M * per_run, training))


def example_2():
    """the gap for the linear-Gaussian NPE"""
    import numpy as np
    tau2, s2 = 4.0, 0.25
    v = tau2 * s2 / (tau2 + s2)                      # posterior variance
    for N in (100, 10_000):
        sigma_a = np.sqrt(v / (N * (tau2 + s2)))     # statistical error of the fitted slope
        print("N = %5d: error at x = 1.2: %.3f (%.1f%% of the error bar), at x = 6: %.3f (%.0f%%)" % (N, 1.2 * sigma_a, 120 * sigma_a / np.sqrt(v), 6 * sigma_a, 600 * sigma_a / np.sqrt(v)))
    rng = np.random.default_rng(0)                   # check N = 100 by refitting many times
    slopes = []
    for _ in range(20000):
        th = rng.normal(0, 2, 100); x = th + rng.normal(0, 0.5, 100)
        slopes.append((th * x).sum() / (x * x).sum())
    print("scatter of the slope over 20,000 training sets of 100: %.4f (formula %.4f)" % (np.std(slopes), np.sqrt(v / (100 * (tau2 + s2)))))


def example_3():
    """how many prior simulations land near the answer"""
    for d in (2, 6):
        frac = (0.6 / 10) ** d                       # box of +/- 3 posterior sd in a prior of width 10
        print("%d parameters: %.2e of prior draws, %.3g out of 10,000" % (d, frac, 1e4 * frac))


def example_4():
    """every posterior moment from one function of the data"""
    import numpy as np
    s2, x = 0.25, 1.2
    theta = np.linspace(-10, 10, 400001)
    prior = 0.5 * np.exp(-(theta + 1.5)**2 / 2) + 0.5 * np.exp(-(theta - 2)**2 / 0.5)    # any prior: here two bumps
    def evidence(xx): return np.trapezoid(prior * np.exp(-(xx - theta)**2 / (2 * s2)), theta)
    h = 1e-3
    d1 = (np.log(evidence(x + h)) - np.log(evidence(x - h))) / (2 * h)
    d2 = (np.log(evidence(x + h)) - 2 * np.log(evidence(x)) + np.log(evidence(x - h))) / h**2
    post = prior * np.exp(-(x - theta)**2 / (2 * s2)); post /= np.trapezoid(post, theta)
    mean = np.trapezoid(theta * post, theta); var = np.trapezoid((theta - mean)**2 * post, theta)
    print("posterior mean:     direct %.4f, from the evidence %.4f" % (mean, x + s2 * d1))
    print("posterior variance: direct %.4f, from the evidence %.4f" % (var, s2 + s2**2 * d2))


if __name__ == "__main__":
    print("\n--- when does amortisation pay? ---")
    example_1()
    print("\n--- the gap for the linear-Gaussian NPE ---")
    example_2()
    print("\n--- how many prior simulations land near the answer ---")
    example_3()
    print("\n--- every posterior moment from one function of the data ---")
    example_4()
