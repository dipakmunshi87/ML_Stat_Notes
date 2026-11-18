"""Worked examples of chapter 32: Symmetry and degeneracy: what the data cannot tell you

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 32_symmetry.py
"""


def example_1():
    """the Fisher matrix knows"""
    import numpy as np
    a, b, s = 2.0, 0.5, 0.1
    grad = np.array([b, a])                          # derivatives of y = a b with respect to (a, b)
    F = np.outer(grad, grad) / s**2
    lam, vec = np.linalg.eigh(F)
    print("determinant %.1f, eigenvalues %.0f and %.0f" % (np.linalg.det(F), lam[1], abs(lam[0])))
    print("measured direction", (vec[:, 1] / vec[0, 1] * b).round(2), " blind direction", (vec[:, 0] / vec[0, 0] * a).round(2), " error along the first: %.2f" % lam[1]**-0.5)


def example_2():
    """only the product is measured"""
    import numpy as np
    a, b = 2.0, 3.0
    grad = np.array([b, a])                          # derivatives of y = a b x with respect to (a, b)
    F = np.outer(grad, grad)
    lam, vec = np.linalg.eigh(F)
    print("Fisher matrix", F.tolist(), " eigenvalues", lam.round(6), " blind direction", (vec[:, 0] / vec[0, 0] * a).round(2))


def example_3():
    """which component is which?"""
    import numpy as np
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 1_000_000)            # the sampler lands in either labelling equally often
    mu1 = np.where(label == 0, -1.0, 1.0) + 0.05 * rng.standard_normal(label.size)
    print("posterior for the centre of component 1: mean %.2f, sd %.2f (each peak has sd 0.05)" % (mu1.mean(), mu1.std()))


if __name__ == "__main__":
    print("\n--- the Fisher matrix knows ---")
    example_1()
    print("\n--- only the product is measured ---")
    example_2()
    print("\n--- which component is which? ---")
    example_3()
