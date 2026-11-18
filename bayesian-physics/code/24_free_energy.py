"""Worked examples of chapter 24: Free energy: what training minimises

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 24_free_energy.py
"""


def example_1():
    """Derivation in three lines"""
    import numpy as np
    theta = np.linspace(-8, 8, 400001)
    E = 0.25 * theta**4 - theta**2 + 0.3 * theta     # any energy will do: here a lopsided double well
    lnZ = np.log(np.trapezoid(np.exp(-E), theta))
    lnp = -E - lnZ                                   # log of the true posterior
    for m, w in ((1.3, 0.6), (-1.5, 0.5), (0.0, 1.5)):               # three Gaussian trial distributions
        q = np.exp(-(theta - m)**2 / (2 * w**2)) / np.sqrt(2 * np.pi * w**2)
        F = np.trapezoid(q * E, theta) + np.trapezoid(q * np.log(q + 1e-300), theta)     # <E> - S
        KL = np.trapezoid(q * (np.log(q + 1e-300) - lnp), theta)
        print("trial N(%.1f, %.1f): F[q] = %.4f, -ln Z = %.4f, difference %.4f, KL = %.4f" % (m, w, F, -lnZ, F + lnZ, KL))


def example_2():
    """a Gaussian trial for a quadratic energy"""
    import numpy as np
    from scipy.optimize import minimize
    kappa, theta0 = 4.0, 1.0
    F = lambda z: kappa * (z[0] - theta0)**2 / 2 + kappa * np.exp(2 * z[1]) / 2 - z[1]      # z = (m, ln w)
    fit = minimize(F, [0.0, 0.0])
    print("variational: m = %.4f, w = %.4f;  exact posterior: mean %.4f, sd %.4f" % (fit.x[0], np.exp(fit.x[1]), theta0, kappa**-0.5))


def example_3():
    """mean field on two correlated parameters"""
    import numpy as np
    for rho in (0.0, 0.5, 0.9, 0.99):
        K = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))        # stiffness matrix of the true posterior
        mean_field_sd = K[0, 0]**-0.5                # each factor sees the stiffness with the other held fixed
        print("rho = %.2f: true sd 1.00, mean-field sd %.2f, too small by %.2fx" % (rho, mean_field_sd, 1 / mean_field_sd))


def example_4():
    """Stirling's formula as a loop expansion"""
    import numpy as np, math
    n = 5
    classical = n**n * np.exp(-n)
    one_loop = classical * np.sqrt(2 * np.pi * n)
    two_loop = one_loop * (1 + 1 / (12 * n))
    print("classical %.2f, one loop %.2f, two loops %.2f, exact %d" % (classical, one_loop, two_loop, math.factorial(n)))


if __name__ == "__main__":
    print("\n--- Derivation in three lines ---")
    example_1()
    print("\n--- a Gaussian trial for a quadratic energy ---")
    example_2()
    print("\n--- mean field on two correlated parameters ---")
    example_3()
    print("\n--- Stirling's formula as a loop expansion ---")
    example_4()
