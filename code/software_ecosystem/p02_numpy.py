import time
import numpy as np

rng = np.random.default_rng(42)          # modern Generator API (not np.random.seed)
x = rng.normal(size=(2000, 3))           # 2000 particles in 3D

def pairwise_loop(x):
    n = len(x); d = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.sqrt(((x[i] - x[j]) ** 2).sum())
    return d

def pairwise_broadcast(x):
    diff = x[:, None, :] - x[None, :, :]  # (n,1,3) - (1,n,3) -> (n,n,3)
    return np.sqrt((diff ** 2).sum(-1))

def pairwise_gram(x):                     # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b  (BLAS matmul)
    sq = (x ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0))

small = x[:300]
t = time.perf_counter(); a = pairwise_loop(small); t_loop = time.perf_counter() - t
t = time.perf_counter(); b = pairwise_broadcast(small); t_b = time.perf_counter() - t
print(f"300 particles: loops {t_loop*1e3:.0f} ms, broadcasting {t_b*1e3:.1f} ms, same: {np.allclose(a, b)}")
t = time.perf_counter(); c = pairwise_gram(x); t_g = time.perf_counter() - t
print(f"2000 particles via matrix product: {t_g*1e3:.0f} ms, max error {abs(c - pairwise_broadcast(x)).max():.1e}")
# einsum: the same contraction written as index notation
print(np.allclose(np.einsum("ik,jk->ij", x, x), x @ x.T))
