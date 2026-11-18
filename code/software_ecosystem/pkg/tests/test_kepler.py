import numpy as np
import pytest

from orbits import solve_kepler


@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_satisfies_keplers_equation(e):
    M = np.linspace(0, 2 * np.pi, 101)
    E = solve_kepler(M, e)
    np.testing.assert_allclose(E - e * np.sin(E), M, atol=1e-12)


def test_circular_orbit_is_identity():
    M = np.array([0.1, 1.0, 3.0])
    np.testing.assert_allclose(solve_kepler(M, 0.0), M)


def test_rejects_unbound_orbit():
    with pytest.raises(ValueError):
        solve_kepler(1.0, 1.2)
