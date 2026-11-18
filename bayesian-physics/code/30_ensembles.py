"""Worked examples of chapter 30: Three ensembles, three kinds of inference

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 30_ensembles.py
"""


def example_1():
    """how many coefficients does the curve need?"""
    import numpy as np
    x = np.linspace(-1, 1, 20)
    y = np.array([-0.0578, -0.4708, 0.2324, 0.0531, 0.1977, 0.7224, 0.7013, 0.6845, 1.3723, 0.2087,
                  0.9230, 0.9663, 1.6518, 1.0614, 1.3489, 1.0267, 1.7997, 1.3545, 0.4366, 1.2630])
    sigma, tau = 0.3, 2.0            # noise per point, prior width of each coefficient
    n, lnZ = len(y), []
    for d in (1, 2, 3, 4, 5):
        X = np.vander(x, d, increasing=True)
        K = sigma**2 * np.eye(n) + tau**2 * X @ X.T
        lnZ.append(-0.5 * (y @ np.linalg.solve(K, y) + np.linalg.slogdet(K)[1] + n * np.log(2 * np.pi)))
    lnZ = np.array(lnZ)
    p = np.exp(lnZ - lnZ.max()); p /= p.sum()
    print("ln Z:", lnZ.round(2))
    print("probabilities (%):", (100 * p).round(1), " mean number of coefficients %.2f" % (p @ np.arange(1, 6)))


if __name__ == "__main__":
    print("\n--- how many coefficients does the curve need? ---")
    example_1()
