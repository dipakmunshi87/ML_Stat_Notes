"""Worked examples of chapter 22: Energy-based models: learning the energy itself

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 22_energy_based.py
"""


def example_1():
    """two spins"""
    import numpy as np
    target = 0.6                                     # correlation in the data
    print("exact coupling atanh(0.6) = %.3f, ln 2 = %.3f" % (np.arctanh(target), np.log(2)))
    J = 0.0
    for step in range(1, 31):
        J += 0.25 * (target - np.tanh(J))            # data correlation minus model correlation
        if step in (1, 5, 10, 30): print("after %2d steps: J = %.2f" % (step, J))


def example_2():
    """one spin and a field"""
    import numpy as np
    mean_spin = 0.73 - 0.27
    print("mean spin %.2f, field h = %.3f" % (mean_spin, np.arctanh(mean_spin)))


def example_3():
    """direct and indirect correlations"""
    import numpy as np, itertools
    J = np.log(2)
    states = np.array(list(itertools.product([-1, 1], repeat=3)))
    w = np.exp(J * (states[:, 0] * states[:, 1] + states[:, 1] * states[:, 2]))    # chain 1-2-3
    p = w / w.sum()
    corr = lambda i, j: (p * states[:, i] * states[:, j]).sum()
    print("neighbours: %.2f   ends of the chain: %.2f" % (corr(0, 1), corr(0, 2)))


if __name__ == "__main__":
    print("\n--- two spins ---")
    example_1()
    print("\n--- one spin and a field ---")
    example_2()
    print("\n--- direct and indirect correlations ---")
    example_3()
