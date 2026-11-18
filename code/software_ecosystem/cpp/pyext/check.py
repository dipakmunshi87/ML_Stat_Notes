import numpy as np
import fastkepler
M = np.linspace(0, 2 * np.pi, 1_000_000)
E = fastkepler.solve_kepler(M, 0.9)
print("max |E - e sin E - M| =", np.abs(E - 0.9 * np.sin(E) - M).max())
try:
    fastkepler.solve_kepler(M, 1.5)
except ValueError as err:                    # C++ std::invalid_argument arrives as Python ValueError
    print("ValueError:", err)
