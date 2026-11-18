"""Worked examples of chapter 3: Path integrals: the likelihood as a sum over histories

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 03_path_integral.py
"""


def example_1():
    """doing the path integral"""
    import numpy as np
    from scipy import stats, integrate
    D, t1, t2, x2 = 0.5, 1.0, 2.0, 1.3
    integrand = lambda x1: stats.norm(0, np.sqrt(2*D*t1)).pdf(x1) * stats.norm(0, np.sqrt(2*D*t2)).pdf(x2 - x1)
    print("integral over the midpoint: %.6f" % integrate.quad(integrand, -20, 20)[0])
    print("N(x2; 0, 2D(t1 + t2)):      %.6f" % stats.norm(0, np.sqrt(2*D*(t1 + t2))).pdf(x2))
    rng = np.random.default_rng(0)                   # the same thing by simulating many walks of 300 steps
    steps = rng.normal(0, np.sqrt(2 * D * 0.01), (20000, 300))
    print("variance of the end point: %.2f (2Dt = %.2f)" % (steps.sum(1).var(), 2 * D * 3.0))


def example_2():
    """Wick's theorem, and the first correction to an error bar"""
    import numpy as np
    from scipy import integrate, stats
    print("Gaussian moments <x^2>, <x^4>, <x^6>:", [round(stats.norm.moment(n)) for n in (2, 4, 6)])
    C = 1.0
    for lam in (0.2, 1.0, 5.0):
        w = lambda t, n: t**n * np.exp(-t**2 / (2 * C) - lam * t**4 / 24)
        exact = integrate.quad(w, -10, 10, args=(2,))[0] / integrate.quad(w, -10, 10, args=(0,))[0]
        print("lambda = %.1f: first order %.3f, exact %.3f" % (lam, C - lam * C**3 / 2, exact))


def example_3():
    """a sum over two histories"""
    from scipy import stats
    theta = 0.3                              # fraction of events that are signal
    signal = stats.norm(1.0, 0.2)            # signal energies
    background = 0.5                         # flat density between 0 and 2
    for x in (1.2, 1.0, 1.6):
        a, b = theta * signal.pdf(x), (1 - theta) * background
        print("energy %.1f: likelihood %.3f = %.3f + %.3f,  P(signal) = %.2f" % (x, a + b, a, b, a / (a + b)))


if __name__ == "__main__":
    print("\n--- doing the path integral ---")
    example_1()
    print("\n--- Wick's theorem, and the first correction to an error bar ---")
    example_2()
    print("\n--- a sum over two histories ---")
    example_3()
