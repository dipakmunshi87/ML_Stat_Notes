"""Worked examples of chapter 14: Samplers: Metropolis, Hamiltonian and nested

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 14_samplers.py
"""


def example_1():
    """detailed balance on three states"""
    import numpy as np
    E = np.array([0.0, 1.0, 2.0])
    w = np.exp(-E); Z = w.sum(); p = w / Z
    print("Z = %.3f, probabilities" % Z, p.round(3))
    T = np.zeros((3, 3))                             # Metropolis: propose one of the other two with probability 1/2
    for i in range(3):
        for j in range(3):
            if i != j: T[i, j] = 0.5 * min(1, np.exp(-(E[j] - E[i])))
        T[i, i] = 1 - T[i].sum()
    print("flow 0->1 %.4f, flow 1->0 %.4f" % (p[0] * T[0, 1], p[1] * T[1, 0]))
    print("distribution after one step:", (p @ T).round(3))


def example_2():
    """one trajectory in a quadratic well"""
    import numpy as np
    rng = np.random.default_rng(0)
    kappa, n = 4.0, 200_000
    theta0 = rng.normal(3.0, 0.1, n)                 # start far from equilibrium
    mom = rng.standard_normal(n)                     # fresh random momenta
    steps = 100; dt = (np.pi / 2) / np.sqrt(kappa) / steps        # a quarter period
    theta, pm = theta0.copy(), mom.copy()
    for _ in range(steps):                           # leapfrog integration of Hamilton's equations
        pm -= 0.5 * dt * kappa * theta; theta += dt * pm; pm -= 0.5 * dt * kappa * theta
    print("after a quarter period: mean %.3f, variance %.4f (1/kappa = %.4f)" % (theta.mean(), theta.var(), 1 / kappa))
    print("correlation with the starting point: %.3f" % np.corrcoef(theta, theta0)[0, 1])


def example_3():
    """how long a run takes"""
    import numpy as np
    compression, live = 1e6, 400
    print("ln(1e6) = %.1f, iterations to reach the posterior = %.0f" % (np.log(compression), live * np.log(compression)))


def example_4():
    """the best step size for Metropolis"""
    import numpy as np
    rng = np.random.default_rng(3)
    for step in (0.3, 2.4, 10.0):
        x, chain, accepted = 0.0, [], 0
        for _ in range(200_000):
            y = x + step * rng.standard_normal()
            if np.log(rng.random()) < 0.5 * (x*x - y*y):     # Metropolis rule for a unit Gaussian
                x, accepted = y, accepted + 1
            chain.append(x)
        c = np.array(chain) - np.mean(chain)
        tau = 1.0                                            # integrated autocorrelation time
        for lag in range(1, 2000):
            rho = (c[:-lag] * c[lag:]).mean() / c.var()
            if rho < 0.02: break
            tau += 2 * rho
        print("step %4.1f: accepted %2.0f%%, %.1f steps per independent sample" % (step, 100 * accepted / len(chain), tau))


def example_5():
    """how long nested sampling takes"""
    import numpy as np
    Z = np.sqrt(2 * np.pi) / 10                      # unit Gaussian likelihood, flat prior of width 10
    H = np.log(10 / np.sqrt(2 * np.pi * np.e))       # information gained, in nats
    print("evidence %.3f,  H = %.2f nats,  steps with 100 live points = %.0f" % (Z, H, 100 * H))
    print("a posterior 1e6 times smaller than the prior: %.0f x live points" % np.log(1e6))


if __name__ == "__main__":
    print("\n--- detailed balance on three states ---")
    example_1()
    print("\n--- one trajectory in a quadratic well ---")
    example_2()
    print("\n--- how long a run takes ---")
    example_3()
    print("\n--- the best step size for Metropolis ---")
    example_4()
    print("\n--- how long nested sampling takes ---")
    example_5()
