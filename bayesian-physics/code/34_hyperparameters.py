"""Worked examples of chapter 34: Hyperparameters: parameters about your assumptions

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 34_hyperparameters.py
"""


def example_1():
    """the best single value"""
    import numpy as np
    from scipy.optimize import minimize_scalar
    n, chi2 = 20, 45.0                               # twenty points that fit badly
    best = minimize_scalar(lambda a: -(n / 2 * np.log(a) - a * chi2 / 2), bounds=(1e-3, 10), method="bounded").x
    print("numerical maximum at alpha = %.4f, formula n/chi2 = %.4f, rescaled chi2 per point = %.2f" % (best, n / chi2, best * chi2 / n))


def example_2():
    """integrating the weight away"""
    import numpy as np
    from scipy import integrate
    n = 10
    flat_log = lambda chi2: integrate.quad(lambda a: a**(n/2 - 1) * np.exp(-a * chi2 / 2), 0, np.inf)[0]
    unit_exp = lambda chi2: integrate.quad(lambda a: a**(n/2) * np.exp(-a * (chi2 / 2 + 1)), 0, np.inf)[0]
    print("prior flat in ln(alpha): ratio of marginal likelihoods at chi2 = 10 and 30: %.2f, (30/10)^(n/2) = %.2f" % (flat_log(10) / flat_log(30), 3.0**(n / 2)))
    print("exponential prior:       ratio %.2f, (32/12)^(n/2+1) = %.2f" % (unit_exp(10) / unit_exp(30), (32 / 12)**(n / 2 + 1)))
    print("a Gaussian likelihood would give exp(10) = %.0f" % np.exp(10))


def example_3():
    """two experiments, one of them overconfident"""
    import numpy as np
    theta = np.linspace(-2, 6, 800001); n = 10
    chiA = 9 + 10 * (theta - 1)**2
    chiB = 81 + 10 * (theta - 3)**2
    def summary(logp):
        p = np.exp(logp - logp.max()); p /= np.trapezoid(p, theta)
        m = np.trapezoid(theta * p, theta); return m, np.sqrt(np.trapezoid((theta - m)**2 * p, theta)), theta[p.argmax()]
    m, sd, _ = summary(-(chiA + chiB) / 2)
    print("add the chi-squareds: %.2f +/- %.2f, chi2 at the best fit %.0f" % (m, sd, (chiA + chiB).min()))
    m, sd, peak = summary(-n / 2 * np.log(chiA) - n / 2 * np.log(chiB))
    i = np.abs(theta - peak).argmin()
    print("marginalised weights: %.2f +/- %.2f (peak %.2f); weights n/chi2: A %.2f, B %.2f" % (m, sd, peak, n / chiA[i], n / chiB[i]))
    m, sd, _ = summary(-chiA / 2)
    print("experiment A alone:   %.2f +/- %.2f" % (m, sd))


if __name__ == "__main__":
    print("\n--- the best single value ---")
    example_1()
    print("\n--- integrating the weight away ---")
    example_2()
    print("\n--- two experiments, one of them overconfident ---")
    example_3()
