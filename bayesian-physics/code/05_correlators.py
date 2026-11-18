"""Worked examples of chapter 5: Cumulants and correlation functions: describing what is not Gaussian

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 05_correlators.py
"""


def example_1():
    """counting statistics"""
    from scipy import stats
    for lam in (4, 100):
        m, v, s, k = stats.poisson(lam).stats(moments="mvsk")
        print("mean %g: variance %g, skewness %.2f, excess kurtosis %.2f" % (m, v, s, k))


def example_2():
    """an exponential waiting time"""
    import numpy as np, math
    from scipy import stats
    print("cumulants (n-1)!:", [math.factorial(n - 1) for n in (1, 2, 3, 4)])
    m, v, s, k = stats.expon().stats(moments="mvsk")
    print("skewness %.0f, excess kurtosis %.0f; sum of ten: skewness %.2f" % (s, k, s / np.sqrt(10)))


def example_3():
    """correcting a tail probability"""
    import numpy as np
    from scipy import stats
    n, x = 10, 18.0
    z = (x - n) / np.sqrt(n)
    skew = 2 / np.sqrt(n)
    gauss = stats.norm.sf(z)
    edgeworth = gauss + stats.norm.pdf(z) * skew / 6 * (z**2 - 1)
    exact = stats.gamma(n).sf(x)
    print("z = %.2f   Gaussian %.4f   Edgeworth %.4f   exact %.4f   exact/Gaussian %.1f" % (z, gauss, edgeworth, exact, exact / gauss))


def example_4():
    """a weak quartic interaction"""
    import numpy as np
    from scipy import integrate
    for g in (0.001, 0.01):
        mom = lambda n: integrate.quad(lambda x: x**n * np.exp(-x**2/2 - g*x**4), -14, 14)[0]
        var = mom(2) / mom(0)
        k4 = mom(4) / mom(0) - 3 * var**2            # connected four-point function
        print("g = %g: variance %.4f (first order %.4f), fourth cumulant %.4f (first order %.4f)" % (g, var, 1 - 12*g, k4, -24*g))


if __name__ == "__main__":
    print("\n--- counting statistics ---")
    example_1()
    print("\n--- an exponential waiting time ---")
    example_2()
    print("\n--- correcting a tail probability ---")
    example_3()
    print("\n--- a weak quartic interaction ---")
    example_4()
