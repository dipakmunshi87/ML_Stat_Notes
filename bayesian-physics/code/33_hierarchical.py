"""Worked examples of chapter 33: Hierarchical models: many objects, one population

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 33_hierarchical.py
"""


def example_1():
    """eight noisy measurements"""
    import numpy as np
    J, sigma, tau, ybar, y = 8, 1.0, 0.5, 2.0, 3.5
    lam = sigma**2 / (sigma**2 + tau**2)
    print("lambda = %.1f" % lam)
    print("object alone:  %.2f +/- %.2f" % (y, sigma))
    print("hierarchical:  %.2f +/- %.2f" % (y - lam * (y - ybar), np.sqrt((1 - lam) * sigma**2 + lam * sigma**2 / J)))
    print("all identical: %.2f +/- %.2f" % (ybar, sigma / np.sqrt(J)))
    print("population mean known to %.2f" % np.sqrt((sigma**2 + tau**2) / J))


def example_2():
    """two stars, measured well and badly"""
    import numpy as np
    pop_mean, pop_sd, reading = 100.0, 10.0, 130.0
    for err in (20.0, 5.0):
        w = pop_sd**2 / (pop_sd**2 + err**2)
        print("error %2.0f: weight %.1f, estimate %.0f +/- %.1f" % (err, w, pop_mean + w * (reading - pop_mean), (1 / pop_sd**2 + 1 / err**2)**-0.5))


if __name__ == "__main__":
    print("\n--- eight noisy measurements ---")
    example_1()
    print("\n--- two stars, measured well and badly ---")
    example_2()
