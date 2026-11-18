// Minimal test executable: returns non-zero on failure, so ctest (and CI) report it.
#include "kepler.hpp"
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    std::vector<double> M;
    for (int i = 0; i <= 100; ++i) M.push_back(2 * std::numbers::pi * i / 100);
    for (double e : {0.0, 0.1, 0.5, 0.9, 0.99}) {
        const auto E = solve_kepler(M, e);
        for (std::size_t i = 0; i < M.size(); ++i) CHECK(std::abs(E[i] - e * std::sin(E[i]) - M[i]) < 1e-12);
    }
    bool threw = false;
    try { solve_kepler(M, 1.2); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all tests passed", failures);
    return failures ? 1 : 0;
}
