"""After profiling: replace the O(n^2) loop with a KD-tree (O(n log n))."""
import time
import numpy as np
from scipy.spatial import cKDTree
from slow_pipeline import pair_counts

x = np.random.default_rng(1).normal(size=(4000, 3))
t = time.perf_counter(); slow = pair_counts(x, 0.5); t_slow = time.perf_counter() - t
t = time.perf_counter(); fast = len(cKDTree(x).query_pairs(0.5)); t_fast = time.perf_counter() - t
print(f"loop: {slow} pairs in {t_slow*1e3:.0f} ms;  KD-tree: {fast} pairs in {t_fast*1e3:.0f} ms;  same: {slow == fast}")
