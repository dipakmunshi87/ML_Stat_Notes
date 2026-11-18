// oob.cpp -- a classic off-by-one bug that often "works" silently.
#include <iostream>
#include <vector>

double smooth_sum(const std::vector<double>& u) {
    double s = 0.0;
    for (std::size_t i = 0; i <= u.size(); ++i)   // bug: <= reads one element past the end
        s += 0.5 * (u[i] + u[i - (i > 0)]);
    return s;
}

int main() {
    std::vector<double> u(8, 1.0);
    std::cout << "sum = " << smooth_sum(u) << '\n';
}
