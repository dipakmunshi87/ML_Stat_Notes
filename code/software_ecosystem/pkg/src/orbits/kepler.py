import numpy as np
from numpy.typing import ArrayLike, NDArray


def solve_kepler(
    M: ArrayLike, e: float, tol: float = 1e-13, maxiter: int = 50
) -> NDArray[np.float64]:
    """Solve Kepler's equation E - e sin E = M for the eccentric anomaly E (Newton's method).

    Vectorised over M. Raises ValueError for e outside [0, 1).
    """
    if not 0 <= e < 1:
        raise ValueError(f"eccentricity must be in [0, 1), got {e}")
    M = np.asarray(M, dtype=float)
    E = np.where(e > 0.8, np.pi, M)              # robust starting guess for high e
    for _ in range(maxiter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            return E
    raise RuntimeError("Kepler solver did not converge")
