import time
import numpy as np
from numba import njit, prange

def energy_python(x, m, eps):
    n, e = len(m), 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.sqrt(((x[i] - x[j]) ** 2).sum() + eps * eps)
            e -= m[i] * m[j] / r
    return e

@njit(parallel=True, fastmath=False, cache=True)
def energy_numba(x, m, eps):
    n, e = len(m), 0.0
    for i in prange(n):                      # parallel over i (OpenMP-like threads)
        ei = 0.0
        for j in range(i + 1, n):
            dx = x[i, 0] - x[j, 0]; dy = x[i, 1] - x[j, 1]; dz = x[i, 2] - x[j, 2]
            ei -= m[i] * m[j] / np.sqrt(dx * dx + dy * dy + dz * dz + eps * eps)
        e += ei                              # numba turns this into a reduction
    return e

rng = np.random.default_rng(0)
x, m = rng.normal(size=(3000, 3)), rng.uniform(0.5, 1.5, 3000)
energy_numba(x[:10], m[:10], 0.01)           # first call compiles
t = time.perf_counter(); e_py = energy_python(x[:400], m[:400], 0.01); t_py = time.perf_counter() - t
t = time.perf_counter(); e_nb = energy_numba(x[:400], m[:400], 0.01); t_nb = time.perf_counter() - t
print(f"400 bodies: Python {t_py*1e3:.0f} ms, Numba {t_nb*1e3:.2f} ms, agree: {np.isclose(e_py, e_nb)}")
t = time.perf_counter(); energy_numba(x, m, 0.01); print(f"3000 bodies with Numba: {(time.perf_counter()-t)*1e3:.0f} ms")
