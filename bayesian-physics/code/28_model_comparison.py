"""Worked examples of chapter 28: Model comparison: evidence, Bayes factors and Occam's razor

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 28_model_comparison.py
"""


def example_1():
    """is there a signal?"""
    import numpy as np
    tau, s = 2.0, 0.5
    penalty = 0.5 * np.log(1 + tau**2 / s**2)
    print("Occam penalty %.2f; break-even at %.2f = %.1f sigma" % (penalty, np.sqrt(2 * penalty / (1 / s**2 - 1 / (s**2 + tau**2))), np.sqrt(2 * penalty / (1 / s**2 - 1 / (s**2 + tau**2))) / s))
    for xbar in (0.5, 0.87, 1.2, 2.0):
        lnB = xbar**2 / 2 * (1 / s**2 - 1 / (s**2 + tau**2)) - penalty
        print("xbar = %.2f (%.1f sigma): ln B = %+.2f, Bayes factor %.3g" % (xbar, xbar / s, lnB, np.exp(lnB)))


def example_2():
    """three sigma is not always evidence"""
    from scipy import stats
    post = stats.norm(0.30, 0.10)
    for width in (2, 20):
        bf = post.pdf(0) / stats.norm(0, width).pdf(0)           # Savage-Dickey ratio, in favour of zero
        print("prior width %2d: posterior %.4f, prior %.4f, ratio %.2f -> odds %.1f to 1 %s" % (width, post.pdf(0), stats.norm(0, width).pdf(0), bf, max(bf, 1 / bf), "for zero" if bf > 1 else "against zero"))
    print("two-sided p-value at 3 sigma: 1 in %.0f" % (1 / (2 * stats.norm.sf(3))))


def example_3():
    """the BIC against the exact evidence"""
    import numpy as np
    x = np.linspace(-1, 1, 20)
    y = np.array([-0.0578, -0.4708, 0.2324, 0.0531, 0.1977, 0.7224, 0.7013, 0.6845, 1.3723, 0.2087,
                  0.9230, 0.9663, 1.6518, 1.0614, 1.3489, 1.0267, 1.7997, 1.3545, 0.4366, 1.2630])
    sigma, tau = 0.3, 2.0            # noise per point, prior width of each coefficient
    bic, lnZ = {}, {}
    for d, name in ((2, "line"), (3, "quadratic"), (4, "cubic")):
        X = np.vander(x, d, increasing=True)
        chi2 = ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0])**2).sum() / sigma**2
        bic[name] = chi2 / 2 + d / 2 * np.log(len(y))
        K = sigma**2 * np.eye(len(y)) + tau**2 * X @ X.T         # covariance of the data under the model
        lnZ[name] = -0.5 * (y @ np.linalg.solve(K, y) + np.linalg.slogdet(K)[1] + len(y) * np.log(2 * np.pi))
        print("%-9s BIC/2 = %.2f   -ln Z = %.2f" % (name, bic[name], -lnZ[name]))
    print("quadratic preferred over line, cubic:  BIC %.2f, %.2f   exact %.2f, %.2f" % (bic["line"] - bic["quadratic"], bic["cubic"] - bic["quadratic"], lnZ["quadratic"] - lnZ["line"], lnZ["quadratic"] - lnZ["cubic"]))


if __name__ == "__main__":
    print("\n--- is there a signal? ---")
    example_1()
    print("\n--- three sigma is not always evidence ---")
    example_2()
    print("\n--- the BIC against the exact evidence ---")
    example_3()
