#include "kepler.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

std::vector<double> solve_kepler(const std::vector<double>& M, double e) {
    if (e < 0.0 || e >= 1.0) throw std::invalid_argument("eccentricity must be in [0, 1)");
    std::vector<double> E(M.size());
    for (std::size_t i = 0; i < M.size(); ++i) {
        double x = e > 0.8 ? std::numbers::pi : M[i];
        for (int it = 0; it < 50; ++it) {
            const double dx = (x - e * std::sin(x) - M[i]) / (1.0 - e * std::cos(x));
            x -= dx;
            if (std::abs(dx) < 1e-13) break;
        }
        E[i] = x;
    }
    return E;
}
