"""Worked examples of chapter 31: Phase transitions in inference

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 31_phase_transitions.py
"""


def example_1():
    """the posterior probability that the coin is biased"""
    import numpy as np
    c = np.log(7 / 3); hc = np.log(5 / 3) / c
    print("c = %.3f, h_c = %.3f" % (c, hc))
    for n in (10, 100, 1000, 10000):
        P = lambda h: 1 / (1 + np.exp(-n * c * (h - hc)))
        print("n = %5d: P(biased) at h = 0.58, 0.60, 0.62: %.3f %.3f %.3f   width %.5f" % (n, P(0.58), P(0.60), P(0.62), 2 * np.log(9) / (c * n)))


def example_2():
    """a hidden direction in noise"""
    import numpy as np
    rng = np.random.default_rng(1)
    n = 1500
    for lam in (0.5, 1.5, 3.0):
        W = rng.standard_normal((n, n)); W = (W + W.T) / np.sqrt(2 * n)       # noise with eigenvalues in [-2, 2]
        v = rng.standard_normal(n); v /= np.linalg.norm(v)
        vals, vecs = np.linalg.eigh(W + lam * np.outer(v, v))
        theory = (lam + 1 / lam, 1 - 1 / lam**2) if lam > 1 else (2.0, 0.0)
        print("strength %.1f: top eigenvalue %.2f (theory %.2f), overlap^2 %.2f (theory %.2f)" % (lam, vals[-1], theory[0], (vecs[:, -1] @ v)**2, theory[1]))


def example_3():
    """two parameters, two models"""
    import numpy as np
    from scipy import integrate, special
    for n in (1e2, 1e4, 1e6):
        regular = (np.sqrt(np.pi / n) * special.erf(np.sqrt(n)) / 2)**2                 # K = a^2 + b^2
        inner = lambda a: np.sqrt(np.pi / (n * a * a)) * special.erf(np.sqrt(n) * a) / 2 if a > 0 else 1.0
        singular = integrate.quad(inner, 0, 1, limit=400, points=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1])[0]   # K = a^2 b^2
        print("n = %7d: regular %.2f (ln n = %.2f)   singular %.2f (rule %.2f)" % (n, -np.log(regular), np.log(n), -np.log(singular), 0.5 * np.log(n) - np.log(np.log(n))))


if __name__ == "__main__":
    print("\n--- the posterior probability that the coin is biased ---")
    example_1()
    print("\n--- a hidden direction in noise ---")
    example_2()
    print("\n--- two parameters, two models ---")
    example_3()
