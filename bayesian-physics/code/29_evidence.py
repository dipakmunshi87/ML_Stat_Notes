"""Worked examples of chapter 29: Computing the evidence: thermodynamic integration and information criteria

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 29_evidence.py
"""


def example_1():
    """thermodynamic integration done exactly"""
    import numpy as np
    from scipy import stats, integrate
    tau2, s2, d = 4.0, 1.0, 1.0                      # prior variance, noise variance, the measurement
    def mean_lnL(beta):                              # average log-likelihood over the tempered posterior
        prec = beta / s2 + 1 / tau2
        mu, var = (beta * d / s2) / prec, 1 / prec
        return -0.5 * np.log(2 * np.pi * s2) - 0.5 * ((d - mu)**2 + var) / s2
    print("<ln L> at beta = 0, 0.1, 0.5, 1:", ", ".join("%.2f" % mean_lnL(b) for b in (0, 0.1, 0.5, 1)))
    print("integral %.3f   direct ln Z %.3f" % (integrate.quad(mean_lnL, 0, 1)[0], stats.norm(0, np.sqrt(tau2 + s2)).logpdf(d)))


def example_2():
    """how many parameters did the data constrain?"""
    import numpy as np
    rng = np.random.default_rng(0)
    tau2, s2, d = 4.0, 1.0, 1.0
    var = 1 / (1 / s2 + 1 / tau2); mu = d / s2 * var
    theta = rng.normal(mu, np.sqrt(var), 2_000_000)  # posterior samples
    lnL = -0.5 * (d - theta)**2 / s2
    print("posterior mean %.1f, variance %.1f, Var(ln L) = %.3f, d = %.2f" % (mu, var, lnL.var(), 2 * lnL.var()))


def example_3():
    """four criteria on one data set"""
    import numpy as np
    x = np.linspace(-1, 1, 20)
    y = np.array([-0.0578, -0.4708, 0.2324, 0.0531, 0.1977, 0.7224, 0.7013, 0.6845, 1.3723, 0.2087,
                  0.9230, 0.9663, 1.6518, 1.0614, 1.3489, 1.0267, 1.7997, 1.3545, 0.4366, 1.2630])
    sigma, tau = 0.3, 2.0            # noise per point, prior width of each coefficient
    n = len(y)
    for d, name in ((2, "line"), (3, "quadratic"), (4, "cubic")):
        X = np.vander(x, d, increasing=True)
        H = X @ np.linalg.solve(X.T @ X, X.T)        # hat matrix of the least-squares fit
        r = y - H @ y
        chi2 = (r**2).sum() / sigma**2
        K = sigma**2 * np.eye(n) + tau**2 * X @ X.T
        lnZ = -0.5 * (y @ np.linalg.solve(K, y) + np.linalg.slogdet(K)[1] + n * np.log(2 * np.pi))
        loo = r / (1 - np.diag(H))                   # leave-one-out prediction errors
        print("%-9s chi2 %.1f  AIC %.1f  BIC %.1f  -2lnZ %.1f  cross-val %.3f  in-sample %.3f" % (name, chi2, chi2 + 2*d, chi2 + d*np.log(n), -2*lnZ, np.sqrt((loo**2).mean()), np.sqrt((r**2).mean())))
    aic = np.array([40.0, 32.9, 34.2]); w = np.exp(-0.5 * (aic - aic.min()))
    print("AIC weights (line, quadratic, cubic):", (w / w.sum()).round(2))


if __name__ == "__main__":
    print("\n--- thermodynamic integration done exactly ---")
    example_1()
    print("\n--- how many parameters did the data constrain? ---")
    example_2()
    print("\n--- four criteria on one data set ---")
    example_3()
