"""GPU-style asynchronous execution: why naive timing lies (runs on CPU too)."""
import time
import jax
import jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (3000, 3000))
f = jax.jit(lambda a: (a @ a.T).sum())
f(x).block_until_ready()                      # warm-up: first call includes compilation

t = time.perf_counter(); y = f(x); t_naive = time.perf_counter() - t                 # returns before the work is done
t = time.perf_counter(); y = f(x).block_until_ready(); t_true = time.perf_counter() - t
print(f"naive timing {t_naive*1e3:.2f} ms  vs  with block_until_ready {t_true*1e3:.1f} ms")
# PyTorch on a GPU: call torch.cuda.synchronize() before reading the clock, or use torch.cuda.Event timers.
