// modern.cpp -- the C++11/14/17/20 features interviewers ask about, in one program.
// Build: g++ -std=c++20 -O2 -Wall -Wextra modern.cpp -o modern && ./modern
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// RAII + rule of zero: the vector owns the memory; no destructor, no new/delete needed.
class Field {
public:
    explicit Field(std::size_t n, double v = 0.0) : data_(n, v) {}
    double& operator[](std::size_t i) { return data_[i]; }
    std::span<const double> view() const { return data_; }   // C++20: non-owning view
    std::size_t size() const { return data_.size(); }
private:
    std::vector<double> data_;
};

// Runtime polymorphism: virtual interface + virtual destructor.
struct Potential {
    virtual ~Potential() = default;
    virtual double operator()(double r) const = 0;
};
struct Plummer final : Potential {
    double a;
    explicit Plummer(double a_) : a(a_) {}
    double operator()(double r) const override { return -1.0 / std::sqrt(r * r + a * a); }
};

// C++17 std::optional: a result that may be absent, instead of a magic value like -1.
std::optional<double> safe_sqrt(double x) {
    if (x < 0) return std::nullopt;
    return std::sqrt(x);
}

double mean(std::span<const double> xs) {                        // works for vector, array, C array
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

int main() {
    // smart pointers: unique ownership, freed automatically
    std::unique_ptr<Potential> phi = std::make_unique<Plummer>(0.1);
    std::cout << "phi(1) = " << (*phi)(1.0) << '\n';

    // move semantics: steal the buffer instead of copying a million doubles
    Field f(1'000'000, 1.0);
    Field g = std::move(f);
    std::cout << "moved field size " << g.size() << ", mean " << mean(g.view()) << '\n';

    // lambdas + algorithms
    std::vector<double> r{0.5, 2.0, 0.1, 1.5};
    std::sort(r.begin(), r.end(), [](double a, double b) { return a > b; });
    std::cout << "sorted descending: ";
    for (double x : r) std::cout << x << ' ';
    std::cout << '\n';

    // C++17 structured bindings over a map
    std::map<std::string, double> masses{{"Sun", 1.0}, {"Vega", 2.14}};
    for (const auto& [name, m] : masses) std::cout << name << " -> " << m << '\n';

    // C++20 ranges: lazy filter + transform pipeline
    auto squares_of_even = std::views::iota(1, 11)
                         | std::views::filter([](int i) { return i % 2 == 0; })
                         | std::views::transform([](int i) { return i * i; });
    std::cout << "squares of even numbers: ";
    for (int s : squares_of_even) std::cout << s << ' ';
    std::cout << '\n';

    if (auto v = safe_sqrt(-4.0); !v) std::cout << "sqrt(-4): no real value\n";
    std::cout << "sqrt(9) = " << safe_sqrt(9.0).value() << '\n';
}
