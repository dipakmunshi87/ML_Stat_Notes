// hotspot.cpp -- two functions; which one dominates? Profile, don't guess.
#include <cmath>
#include <iostream>
#include <vector>

double potential(const std::vector<double>& x) {          // O(n^2)
    double e = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = i + 1; j < x.size(); ++j)
            e -= 1.0 / std::sqrt((x[i] - x[j]) * (x[i] - x[j]) + 1e-3);
    return e;
}

double kinetic(const std::vector<double>& v) {            // O(n)
    double e = 0.0;
    for (double vi : v) e += 0.5 * vi * vi;
    return e;
}

int main() {
    const std::size_t n = 6000;
    std::vector<double> x(n), v(n);
    for (std::size_t i = 0; i < n; ++i) { x[i] = std::sin(0.37 * i); v[i] = std::cos(0.11 * i); }
    double total = 0.0;
    for (int step = 0; step < 5; ++step) total += kinetic(v) + potential(x);
    std::cout << "total energy " << total << '\n';
}
