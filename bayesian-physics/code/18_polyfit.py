"""Worked examples of chapter 18: A worked example: fitting a line, a quadratic and a cubic

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 18_polyfit.py
"""


def example_1():
    """predicting beyond the data"""
    import numpy as np
    x = np.linspace(-1, 1, 20)
    y = np.array([-0.0578, -0.4708, 0.2324, 0.0531, 0.1977, 0.7224, 0.7013, 0.6845, 1.3723, 0.2087,
                  0.9230, 0.9663, 1.6518, 1.0614, 1.3489, 1.0267, 1.7997, 1.3545, 0.4366, 1.2630])
    sigma, tau = 0.3, 2.0            # noise per point, prior width of each coefficient
    print("truth at 1.5 and 2.0: %.2f, %.2f" % (1 + 0.8*1.5 - 0.6*1.5**2, 1 + 0.8*2 - 0.6*4))
    for d, name in ((2, "line"), (3, "quadratic"), (4, "cubic")):
        X = np.vander(x, d, increasing=True)
        C = np.linalg.inv(X.T @ X / sigma**2 + np.eye(d) / tau**2)
        mu = C @ X.T @ y / sigma**2
        for xs in (1.5, 2.0):
            v = xs ** np.arange(d)
            print("%-9s at x = %.1f: %5.2f +/- %.2f" % (name, xs, v @ mu, np.sqrt(v @ C @ v)))


if __name__ == "__main__":
    print("\n--- predicting beyond the data ---")
    example_1()
