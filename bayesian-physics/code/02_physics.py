"""Worked examples of chapter 2: The dictionary: inference as statistical physics

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 02_physics.py
"""


def example_1():
    """one measurement of a Gaussian mean"""
    import numpy as np
    x, sigma, tau = 1.2, 0.5, 2.0                    # the measurement, its noise, the prior width
    theta = np.linspace(-6, 6, 200001)
    E = (x - theta)**2 / (2 * sigma**2) + theta**2 / (2 * tau**2)
    p = np.exp(-E); p /= np.trapezoid(p, theta)      # Boltzmann weight of the energy
    mean = np.trapezoid(theta * p, theta); sd = np.sqrt(np.trapezoid((theta - mean)**2 * p, theta))
    kappa = 1 / sigma**2 + 1 / tau**2
    print("numerical: mean %.4f, sd %.4f" % (mean, sd))
    print("formula:   mean %.4f, sd %.4f" % (x * tau**2 / (tau**2 + sigma**2), kappa**-0.5))


def example_2():
    """free energy = energy − entropy, for two hypotheses"""
    import numpy as np
    like  = np.array([0.02, 0.005])          # probability of the data under each hypothesis
    prior = np.array([0.5, 0.5])
    E = -np.log(like * prior)                # energies
    Z = np.exp(-E).sum()                     # partition function = evidence
    p = np.exp(-E) / Z                       # posterior
    F = -np.log(Z)
    meanE = (p * E).sum()
    S = -(p * np.log(p)).sum()
    print("energies", E.round(3), " Z = %.4f" % Z, " posterior", p.round(2))
    print("F = %.3f   <E> = %.3f   S = %.3f   <E> - S = %.3f" % (F, meanE, S, meanE - S))


def example_3():
    """counting photons"""
    import numpy as np
    from scipy import stats
    k, prior_mean = 7, 10.0                  # counts seen; mean of the exponential prior
    rate = 1 + 1/prior_mean                  # posterior is Gamma(shape k+1, rate 1.1)
    post = stats.gamma(k + 1, scale=1/rate)
    print("minimum of the energy: %.2f" % (k / rate))
    print("posterior mean %.2f, sd %.2f" % (post.mean(), post.std()))


if __name__ == "__main__":
    print("\n--- one measurement of a Gaussian mean ---")
    example_1()
    print("\n--- free energy = energy − entropy, for two hypotheses ---")
    example_2()
    print("\n--- counting photons ---")
    example_3()
