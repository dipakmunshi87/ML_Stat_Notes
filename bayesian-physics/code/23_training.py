"""Worked examples of chapter 23: Training as physics: gradient descent with noise

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 23_training.py
"""


def example_1():
    """one weight in a quadratic loss"""
    import numpy as np
    rng = np.random.default_rng(0)
    eta, B, kappa = 0.1, 10, 1.0
    noise = rng.standard_normal(400_000) / np.sqrt(B)
    w, ws = 0.0, []
    for n in noise:
        w -= eta * (kappa * w + n)                   # one SGD step with a noisy gradient
        ws.append(w)
    print("simulated variance %.5f, formula %.5f, Boltzmann T/kappa %.5f, sd %.3f" % (np.var(ws[1000:]), eta / (B * kappa * (2 - eta * kappa)), eta / (2 * B), np.std(ws[1000:])))


def example_2():
    """cooling by halving the learning rate"""
    import numpy as np
    B, kappa = 10, 1.0
    for eta in (0.1, 0.05):
        var = eta / (B * kappa * (2 - eta * kappa))
        print("learning rate %.2f: variance %.5f, spread %.3f" % (eta, var, np.sqrt(var)))


def example_3():
    """escaping a poor minimum"""
    import numpy as np
    barrier = 0.02
    for T in (0.005, 0.0025):
        print("T = %.4f: exp(-barrier/T) = %.5f" % (T, np.exp(-barrier / T)))
    print("ratio: %.0f" % np.exp(barrier / 0.0025 - barrier / 0.005))


if __name__ == "__main__":
    print("\n--- one weight in a quadratic loss ---")
    example_1()
    print("\n--- cooling by halving the learning rate ---")
    example_2()
    print("\n--- escaping a poor minimum ---")
    example_3()
