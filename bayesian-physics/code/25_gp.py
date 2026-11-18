"""Worked examples of chapter 25: Gaussian processes: when the unknown is a function

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 25_gp.py
"""


def example_1():
    """two measurements"""
    import numpy as np
    k = lambda a, b: np.exp(-0.5 * (a - b)**2)       # length 1, unit prior variance
    xo, yo, noise = np.array([0.0, 1.0]), np.array([1.0, 0.5]), 0.1
    K = k(xo[:, None], xo[None, :]) + noise**2 * np.eye(2)
    for xs in (0.5, 2.0, 4.0):
        ks = k(xs, xo)
        print("x = %.1f: predicted %.2f, error bar %.2f" % (xs, abs(ks @ np.linalg.solve(K, yo)), np.sqrt(1 - ks @ np.linalg.solve(K, ks))))


def example_2():
    """interpolating, and forgetting"""
    import numpy as np
    k = lambda a, b: np.exp(-0.5 * (a - b)**2)       # squared-exponential kernel, length 1
    xo, fo = np.array([0.0, 2.0]), np.array([1.0, 0.0])
    K = k(xo[:, None], xo[None, :])
    for xs in (1.0, 5.0):
        ks = k(xs, xo)
        w = np.linalg.solve(K, ks)
        print("x = %.0f: weights %s, mean %.2f, sd %.2f" % (xs, w.round(3), w @ fo, np.sqrt(1 - ks @ w)))


if __name__ == "__main__":
    print("\n--- two measurements ---")
    example_1()
    print("\n--- interpolating, and forgetting ---")
    example_2()
