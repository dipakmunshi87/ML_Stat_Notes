import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def darts(seed, n=4_000_000):
    rng = np.random.default_rng(seed)
    x, y = rng.random(n), rng.random(n)
    return int(np.count_nonzero(x * x + y * y <= 1.0))

if __name__ == "__main__":                       # required for process pools on macOS/Windows
    seeds = range(8)
    t = time.perf_counter(); serial = sum(map(darts, seeds)); t1 = time.perf_counter() - t
    t = time.perf_counter()
    with ProcessPoolExecutor() as pool:          # one Python process per core: sidesteps the GIL
        parallel = sum(pool.map(darts, seeds))
    t2 = time.perf_counter() - t
    print(f"pi ~ {4 * parallel / (8 * 4_000_000):.5f}; serial {t1:.2f} s, process pool {t2:.2f} s, same: {serial == parallel}")
