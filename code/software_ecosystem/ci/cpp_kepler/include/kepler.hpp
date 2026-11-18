#pragma once
#include <vector>
// Solve Kepler's equation E - e sin E = M for each M; throws std::invalid_argument if e is outside [0, 1).
std::vector<double> solve_kepler(const std::vector<double>& M, double e);
