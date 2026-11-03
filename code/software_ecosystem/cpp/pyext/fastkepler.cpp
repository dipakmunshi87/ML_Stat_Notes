// fastkepler.cpp -- expose a C++ function to Python with pybind11 (zero-copy NumPy arrays).
#include <cmath>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

py::array_t<double> solve_kepler(py::array_t<double, py::array::c_style | py::array::forcecast> M, double e) {
    if (e < 0.0 || e >= 1.0) throw std::invalid_argument("eccentricity must be in [0, 1)");
    auto in = M.unchecked<1>();
    py::array_t<double> E(in.shape(0));
    auto out = E.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;            // pure C++ below: let other Python threads run
        for (py::ssize_t i = 0; i < in.shape(0); ++i) {
            double x = e > 0.8 ? M_PI : in(i);
            for (int it = 0; it < 50; ++it) {
                const double dx = (x - e * std::sin(x) - in(i)) / (1.0 - e * std::cos(x));
                x -= dx;
                if (std::abs(dx) < 1e-13) break;
            }
            out(i) = x;
        }
    }
    return E;
}

PYBIND11_MODULE(fastkepler, m) {
    m.doc() = "Kepler-equation solver in C++";
    m.def("solve_kepler", &solve_kepler, py::arg("M"), py::arg("e"),
          "Eccentric anomaly E from mean anomaly M (vectorised)");
}
