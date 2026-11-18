"""Worked examples of chapter 36: Misspecification: when the simulator is wrong

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 36_misspecification.py
"""


def example_1():
    """the wrong noise level"""
    from scipy import stats
    for ratio in (0.5, 1.0, 2.0):                    # assumed noise / true noise
        print("model noise = %.1f x true: the '68%%' interval covers %.0f%%" % (ratio, 100 * (2 * stats.norm.cdf(ratio * stats.norm.ppf(0.84134)) - 1)))


def example_2():
    """a missing component"""
    import numpy as np
    c = 0.2                                          # the missing offset, in units of the noise s
    for n in (1, 25, 100, 400):
        print("n = %3d: error bar %.2f s, bias %.1f s, bias = %.1f error bars" % (n, 1 / np.sqrt(n), c, c * np.sqrt(n)))


if __name__ == "__main__":
    print("\n--- the wrong noise level ---")
    example_1()
    print("\n--- a missing component ---")
    example_2()
