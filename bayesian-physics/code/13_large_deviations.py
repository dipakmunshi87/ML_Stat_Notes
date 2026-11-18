"""Worked examples of chapter 13: Large deviations: the probability of the unlikely

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 13_large_deviations.py
"""


def example_1():
    """100 tosses"""
    import numpy as np
    from scipy import stats
    n = 100
    for heads in (60, 70, 90):
        x = heads / n
        I = x * np.log(2 * x) + (1 - x) * np.log(2 * (1 - x))        # exact rate function for a fair coin
        exact = stats.binom(n, 0.5).sf(heads - 1)
        gauss = stats.norm.sf((heads - 0.5 - 50) / 5)                # Gaussian with continuity correction
        print("%d heads: I = %.4f, quadratic %.4f, exact %.1e, Gaussian %.1e, ratio %.1f" % (heads, I, 2 * (x - 0.5)**2, exact, gauss, gauss / exact))


def example_2():
    """the Gaussian case"""
    import numpy as np
    from scipy import stats
    n, a = 50, 0.5
    print("large-deviation estimate exp(-n a^2/2) = %.1e,  exact = %.1e" % (np.exp(-n * a**2 / 2), stats.norm.sf(a * np.sqrt(n))))


def example_3():
    """a rare count"""
    import numpy as np
    from scipy import stats
    lam, k = 10, 25
    I = k * np.log(k / lam) - k + lam                # rate function
    print("I = %.2f,  estimate %.2e,  exact P(25) = %.2e" % (I, np.exp(-I) / np.sqrt(2 * np.pi * k), stats.poisson(lam).pmf(k)))
    z = (k - lam) / np.sqrt(lam)
    print("P(25 or more): exact %.1e, Gaussian (%.1f sigma) %.1e" % (stats.poisson(lam).sf(k - 1), z, stats.norm.sf(z)))


if __name__ == "__main__":
    print("\n--- 100 tosses ---")
    example_1()
    print("\n--- the Gaussian case ---")
    example_2()
    print("\n--- a rare count ---")
    example_3()
