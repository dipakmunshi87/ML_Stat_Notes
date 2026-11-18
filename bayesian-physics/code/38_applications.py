"""Worked examples of chapter 38: Applications: where SBI is used in physics

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 38_applications.py
"""


def example_1():
    """a counting experiment in particle physics"""
    import numpy as np
    from scipy import stats
    obs, bkg = 25, 15.0
    s = np.linspace(0, 60, 60001)
    post = np.exp(obs * np.log(s + bkg) - (s + bkg)); post /= np.trapezoid(post, s)   # flat prior on s >= 0
    cdf = np.cumsum(post) * (s[1] - s[0])
    mean = np.trapezoid(s * post, s); sd = np.sqrt(np.trapezoid(s**2 * post, s) - mean**2)
    q = lambda f: s[np.searchsorted(cdf, f)]
    print("mode %.0f, mean %.1f +/- %.1f, 68%% interval %.1f to %.1f, 95%% upper limit %.1f" % (s[post.argmax()], mean, sd, q(0.16), q(0.84), q(0.95)))
    p = stats.poisson(bkg).sf(obs - 1)
    print("P(25 or more from background) = %.1f%% = %.1f sigma" % (100 * p, stats.norm.isf(p)))


if __name__ == "__main__":
    print("\n--- a counting experiment in particle physics ---")
    example_1()
