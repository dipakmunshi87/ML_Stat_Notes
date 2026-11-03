// crash.cpp -- a segmentation fault to practise on with gdb (Linux) or lldb (macOS).
#include <iostream>
#include <memory>
#include <vector>

struct Particle { double x, v; };

double kinetic_energy(const std::vector<Particle*>& ps) {
    double e = 0.0;
    for (const Particle* p : ps) e += 0.5 * p->v * p->v;   // crashes on a null pointer
    return e;
}

int main() {
    auto a = std::make_unique<Particle>(Particle{0.0, 1.0});
    auto b = std::make_unique<Particle>(Particle{1.0, 2.0});
    std::vector<Particle*> ps{a.get(), nullptr, b.get()};  // bug: a missing particle
    std::cout << "E = " << kinetic_energy(ps) << '\n';
}
