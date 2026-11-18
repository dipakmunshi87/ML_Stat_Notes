"""Worked examples of chapter 8: Information geometry: the Fisher matrix is a metric

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 08_geometry.py
"""


def example_1():
    """the geometry of the Gaussian family"""
    import numpy as np
    def distance(mu1, s1, mu2, s2):                  # Fisher-Rao distance between two Gaussians
        return np.sqrt(2) * np.arccosh(1 + ((mu1 - mu2)**2 / 2 + (s1 - s2)**2) / (2 * s1 * s2))
    print("shift the mean by 1, width 1:   %.2f" % distance(0, 1, 1, 1))
    print("double the width:               %.2f" % distance(0, 1, 0, 2))
    print("shift the mean by 1, width 0.1: %.1f" % distance(0, 0.1, 1, 0.1))


def example_2():
    """a prior that does not depend on the parametrisation"""
    from scipy import stats
    heads, tails = 7, 3
    jeffreys = stats.beta(heads + 0.5, tails + 0.5)
    flat = stats.beta(heads + 1, tails + 1)
    print("Jeffreys prior: %.3f +/- %.3f" % (jeffreys.mean(), jeffreys.std()))
    print("flat prior:     %.3f +/- %.3f" % (flat.mean(), flat.std()))


def example_3():
    """how many coins can be told apart?"""
    import numpy as np
    from scipy import integrate
    length = integrate.quad(lambda p: 1 / np.sqrt(p * (1 - p)), 0, 1)[0]     # equals pi
    for n in (100, 10000):
        print("%d tosses: %.0f distinguishable coins" % (n, length * np.sqrt(n)))


if __name__ == "__main__":
    print("\n--- the geometry of the Gaussian family ---")
    example_1()
    print("\n--- a prior that does not depend on the parametrisation ---")
    example_2()
    print("\n--- how many coins can be told apart? ---")
    example_3()
