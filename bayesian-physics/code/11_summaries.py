"""Worked examples of chapter 11: Summaries, sufficiency and the Fisher limit

From "The Physics of Bayesian Inference". Each function reproduces the numbers of one worked
example in the text. Run the file to run them all:  python 11_summaries.py
"""


def example_1():
    """for two-level atoms the count is sufficient"""
    import numpy as np
    N = 50
    pattern_a = np.zeros(N, int); pattern_a[:12] = 1               # the first 12 atoms excited
    pattern_b = np.zeros(N, int); pattern_b[4::4] = 1               # 12 different atoms excited
    print("excited atoms in each pattern:", pattern_a.sum(), pattern_b.sum())
    for T in (0.5, 0.87, 2.0):
        p = 1 / (1 + np.exp(1 / T))
        like = lambda pat: np.prod(np.where(pat == 1, p, 1 - p))
        print("T = %.2f: likelihood ratio of the two patterns = %.6f" % (T, like(pattern_a) / like(pattern_b)))


def example_2():
    """the best possible thermometer"""
    import numpy as np
    N, k = 50, 12
    p = k / N
    T = 1 / np.log((N - k) / k)                      # from p = 1 / (1 + exp(1/T))
    C = N * p * (1 - p) / T**2                       # heat capacity dU/dT, in units of k_B
    fisher = N * p * (1 - p) / T**4
    print("T = %.2f, heat capacity %.1f, Fisher information %.1f = C/T^2 = %.1f, error %.2f" % (T, C, fisher, C / T**2, fisher**-0.5))


def example_3():
    """five pixels, one number"""
    import numpy as np
    from scipy import stats
    counts = np.array([3, 5, 2, 6, 4])
    post = stats.gamma(counts.sum() + 1, scale=1 / counts.size)       # flat prior on the rate
    print("total %d: rate %.1f +/- %.2f per pixel" % (counts.sum(), post.mean(), post.std()))


def example_4():
    """a summary that loses information"""
    import numpy as np
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200_000, 100))       # 200,000 experiments of 100 readings
    sd_mean, sd_median = data.mean(1).std(), np.median(data, 1).std()
    print("error of the mean %.3f, of the median %.3f, ratio %.3f (sqrt(pi/2) = %.3f)" % (sd_mean, sd_median, sd_median / sd_mean, np.sqrt(np.pi / 2)))


if __name__ == "__main__":
    print("\n--- for two-level atoms the count is sufficient ---")
    example_1()
    print("\n--- the best possible thermometer ---")
    example_2()
    print("\n--- five pixels, one number ---")
    example_3()
    print("\n--- a summary that loses information ---")
    example_4()
