"""A small analysis pipeline with a hidden bottleneck, to practise profiling on."""
import numpy as np


def load(n, rng):
    return rng.normal(size=(n, 3))


def pair_counts(x, rmax):
    # O(n^2) Python loop: the bottleneck
    count = 0
    for i in range(len(x)):
        d = np.sqrt(((x[i + 1:] - x[i]) ** 2).sum(axis=1))
        count += int((d < rmax).sum())
    return count


def summary(x):
    return x.mean(axis=0), np.cov(x.T)


def main():
    rng = np.random.default_rng(1)
    x = load(4000, rng)
    mu, cov = summary(x)
    print("pairs closer than 0.5:", pair_counts(x, 0.5))


if __name__ == "__main__":
    main()
