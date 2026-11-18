"""Worked examples of chapter 7: Information: how much a measurement tells you

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 07_information.py
"""


def example_1():
    """one measurement, then many"""
    import numpy as np
    tau = 2.0
    print("tau = 2, s = 0.5: %.2f nats = %.1f bits" % (0.5 * np.log(1 + tau**2 / 0.5**2), 0.5 * np.log2(1 + tau**2 / 0.5**2)))
    for n in (1, 4, 16, 64):                         # n measurements of noise 1: variance of the mean is 1/n
        nats = 0.5 * np.log(1 + tau**2 * n)
        print("n = %2d: %.2f nats, %.1f bits" % (n, nats, nats / np.log(2)))


def example_2():
    """how many bits is a measurement worth?"""
    import numpy as np
    prior_sd, noise_sd = 10.0, 1.0
    nats = 0.5 * np.log(1 + prior_sd**2 / noise_sd**2)
    print("%.2f nats = %.2f bits;  log2(10) = %.2f" % (nats, nats / np.log(2), np.log2(10)))


def example_3():
    """where to put the next data point"""
    import numpy as np
    prior_sd, noise_sd = 2.0, 0.3
    for x in (0.0, 0.5, 1.0):
        print("x = %.1f: expected information %.2f nats" % (x, 0.5 * np.log(1 + (x * prior_sd / noise_sd)**2)))


def example_4():
    """the heat of forgetting"""
    import numpy as np
    kB, T = 1.380649e-23, 300.0
    print("kT ln 2 = %.2e joules per bit" % (kB * T * np.log(2)))


if __name__ == "__main__":
    print("\n--- one measurement, then many ---")
    example_1()
    print("\n--- how many bits is a measurement worth? ---")
    example_2()
    print("\n--- where to put the next data point ---")
    example_3()
    print("\n--- the heat of forgetting ---")
    example_4()
