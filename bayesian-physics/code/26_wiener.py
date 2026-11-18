"""Worked examples of chapter 26: The Wiener filter: a field answering a source

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 26_wiener.py
"""


def example_1():
    """one number"""
    import numpy as np
    S, N, d = 4.0, 1.0, 3.0
    print("weight %.1f, estimate %.1f, variance %.1f, sd %.2f" % (S / (S + N), S / (S + N) * d, S * N / (S + N), np.sqrt(S * N / (S + N))))


def example_2():
    """filling a gap"""
    import numpy as np
    S = np.array([[1.0, 0.8], [0.8, 1.0]])           # signal covariance of the two pixels
    N, d1 = 0.5, 1.2                                 # noise variance and reading of pixel 1
    est = S[1, 0] / (S[0, 0] + N) * d1
    var = S[1, 1] - S[1, 0]**2 / (S[0, 0] + N)
    print("unobserved pixel: %.2f +/- %.2f (variance %.2f)" % (est, np.sqrt(var), var))


def example_3():
    """filtering by wavelength"""
    for k in (0, 1, 3, 10):
        Ps, Pn = 1 / (k**2 + 1), 0.1
        print("k = %2d: signal power %.3f, filter weight %.2f" % (k, Ps, Ps / (Ps + Pn)))


if __name__ == "__main__":
    print("\n--- one number ---")
    example_1()
    print("\n--- filling a gap ---")
    example_2()
    print("\n--- filtering by wavelength ---")
    example_3()
