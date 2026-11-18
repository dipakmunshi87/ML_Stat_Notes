"""Worked examples of chapter 10: The curse of dimensionality

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 10_curse.py
"""


def example_1():
    """kept fraction against number of summaries"""
    import numpy as np
    V, eps = 4.25, 0.5
    f = np.sqrt(eps**2 / (eps**2 + V))
    print("one summary: %.3f" % f)
    for d in (1, 2, 5, 10, 20):
        print("d = %2d: kept fraction %.2g, simulations per kept sample %.3g" % (d, f**d, f**-d))


def example_2():
    """where the volume of a ball is"""
    for d in (2, 10, 100):
        print("d = %3d: outer tenth of the radius holds %.3f%% of the volume" % (d, 100 * (1 - 0.9**d)))


def example_3():
    """a Gaussian in 100 dimensions is a shell"""
    import numpy as np
    from scipy import stats
    r2 = stats.chi2(100)                             # squared distance from the centre
    print("distance: mean %.1f, sd %.2f" % (np.sqrt(100), np.sqrt(0.5)))
    print("P(9 < r < 11) = %.2f,  P(r < 5) = %.1e" % (r2.cdf(121) - r2.cdf(81), r2.cdf(25)))


if __name__ == "__main__":
    print("\n--- kept fraction against number of summaries ---")
    example_1()
    print("\n--- where the volume of a ball is ---")
    example_2()
    print("\n--- a Gaussian in 100 dimensions is a shell ---")
    example_3()
