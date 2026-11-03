// omp_sum.cpp -- OpenMP: parallel reduction for the gravitational potential energy of N bodies.
// Build: g++ -std=c++17 -O3 -fopenmp omp_sum.cpp -o omp_sum && OMP_NUM_THREADS=4 ./omp_sum
// macOS (Apple clang has no OpenMP): brew install libomp, or use g++-14 from Homebrew.
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

int main() {
    const int n = 20000;
    std::mt19937_64 rng(42);
    std::normal_distribution<double> N(0.0, 1.0);
    std::vector<double> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i) { x[i] = N(rng); y[i] = N(rng); z[i] = N(rng); }

    auto t0 = std::chrono::steady_clock::now();
    double energy = 0.0;
    #pragma omp parallel for reduction(+ : energy) schedule(dynamic, 64)
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + 1e-4);
        }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "threads " << omp_get_max_threads() << ", E = " << energy << ", "
              << secs << " s for " << (double)n * (n - 1) / 2 << " pairs\n";
}
