"""Worked examples of chapter 27: Nuisance parameters: integrating out what you do not want

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 27_marginalisation.py
"""


def example_1():
    """the cost of one nuisance parameter"""
    import numpy as np
    F = np.array([[10., 8.], [8., 10.]])
    F_eff = F[0, 0] - F[0, 1]**2 / F[1, 1]           # Schur complement
    print("F_eff = %.1f: marginal error %.3f, from the full inverse %.3f, with the nuisance known %.3f" % (F_eff, F_eff**-0.5, np.sqrt(np.linalg.inv(F)[0, 0]), F[0, 0]**-0.5))


def example_2():
    """a calibration offset sets a floor"""
    import numpy as np
    for n in (4, 100, 10**8):
        print("n = %9d: error without offset %.2f, with offset %.2f" % (n, 1 / np.sqrt(n), np.sqrt(1 / n + 0.3**2)))


def example_3():
    """an unknown noise level"""
    from scipy import stats
    for n in (5, 20):
        t = stats.t(n - 1)
        print("%2d readings: 68%% interval x %.2f, 95%% interval %.2f standard errors (Gaussian 1.96)" % (n, t.ppf(0.8413), t.ppf(0.975)))


if __name__ == "__main__":
    print("\n--- the cost of one nuisance parameter ---")
    example_1()
    print("\n--- a calibration offset sets a floor ---")
    example_2()
    print("\n--- an unknown noise level ---")
    example_3()
