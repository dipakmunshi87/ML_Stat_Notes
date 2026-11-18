"""Worked examples of chapter 37: Checks: how to tell when the answer is wrong

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 37_checks.py
"""


def example_1():
    """what overconfidence looks like"""
    from scipy import stats
    for factor, label in ((0.5, "too small by 2"), (1 / 3, "too small by 3"), (2.0, "too wide by 2")):
        print("%-15s: a nominal 68%% interval covers %.0f%%" % (label, 100 * (2 * stats.norm.cdf(factor) - 1)))


def example_2():
    """what a rank test shows for an overconfident posterior"""
    import numpy as np
    from scipy import stats
    p_end = stats.norm.cdf(0.8 * stats.norm.ppf(0.1))            # truth in the lowest tenth of a posterior 0.8 as wide
    expected, sd = 200 * 0.1, np.sqrt(200 * 0.1 * 0.9)
    print("P(end bin) = %.3f: %.1f cases where %.0f +/- %.1f are expected, an excess of %.1f sd" % (p_end, 200 * p_end, expected, sd, (200 * p_end - expected) / sd))


def example_3():
    """two chains that disagree"""
    import numpy as np
    means, within_var = np.array([0.0, 0.6]), 1.0
    between = means.var(ddof=1)
    print("variance between chain means %.2f, R-hat = %.3f" % (between, np.sqrt((within_var + between) / within_var)))


if __name__ == "__main__":
    print("\n--- what overconfidence looks like ---")
    example_1()
    print("\n--- what a rank test shows for an overconfident posterior ---")
    example_2()
    print("\n--- two chains that disagree ---")
    example_3()
