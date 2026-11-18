#include "pic/pic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#ifdef PIC_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef PIC_HAVE_MPI
#include <mpi.h>
#endif

namespace pic {

static constexpr double kTwoPi = 6.283185307179586;

Simulation::Simulation(const Config& cfg) : cfg_(cfg), dx_(cfg.length / cfg.n_grid) {
#ifdef PIC_HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks_);
#endif
  if (cfg.n_particles % (2 * nranks_) != 0)
    throw std::runtime_error("n_particles must be divisible by 2 x number of MPI ranks");
  const std::int64_t n_local = cfg.n_particles / nranks_;
  const std::int64_t first = rank_ * n_local;          // this rank owns global indices [first, first + n_local)
  x_.resize(n_local); v_.resize(n_local);
  rho_.assign(cfg.n_grid, 0.0); e_.assign(cfg.n_grid, 0.0);

  std::mt19937_64 rng(cfg.seed + 1000003ULL * rank_);  // independent stream per rank
  std::normal_distribution<double> gauss(0.0, 1.0);
  const std::int64_t n_pairs = cfg.n_particles / 2;
  for (std::int64_t j = 0; j < n_local; ++j) {
    const std::int64_t i = first + j;                  // global index: quiet start, beams interleaved
    const int beam = static_cast<int>(i % 2);
    double x = (static_cast<double>(i / 2) + 0.5) * cfg.length / static_cast<double>(n_pairs);
    x += cfg.perturbation * cfg.length * std::cos(kTwoPi * x / cfg.length);
    x = std::fmod(std::fmod(x, cfg.length) + cfg.length, cfg.length);
    x_[j] = x;
    const double sign = cfg.single_beam ? 1.0 : (beam ? 1.0 : -1.0);
    v_[j] = sign * cfg.v0 + cfg.vth * gauss(rng);
  }
  deposit_and_solve();
  // stagger velocities back half a step for leapfrog
  for (std::size_t j = 0; j < x_.size(); ++j) v_[j] += 0.5 * cfg_.dt * e_at(x_[j]);
}

void Simulation::deposit_and_solve() {
  const int ng = cfg_.n_grid;
  std::fill(rho_.begin(), rho_.end(), 0.0);
  const std::int64_t n = static_cast<std::int64_t>(x_.size());
#ifdef PIC_HAVE_OPENMP
#pragma omp parallel
  {
    std::vector<double> local(ng, 0.0);                // thread-private grid avoids write races
#pragma omp for schedule(static) nowait
    for (std::int64_t j = 0; j < n; ++j) {
      const double g = x_[j] / dx_ - 0.5; const double fl = std::floor(g); const double f = g - fl;
      const int i = static_cast<int>(fl);
      local[(i + ng) % ng] += 1.0 - f; local[(i + 1) % ng] += f;
    }
#pragma omp critical
    for (int i = 0; i < ng; ++i) rho_[i] += local[i];
  }
#else
  for (std::int64_t j = 0; j < n; ++j) {
    const double g = x_[j] / dx_ - 0.5; const double fl = std::floor(g); const double f = g - fl;
    const int i = static_cast<int>(fl);
    rho_[(i + ng) % ng] += 1.0 - f; rho_[(i + 1) % ng] += f;
  }
#endif
#ifdef PIC_HAVE_MPI
  MPI_Allreduce(MPI_IN_PLACE, rho_.data(), ng, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  // Gauss's law, dE/dx = 1 - n_e/n0, integrated exactly on the staggered grid: the cumulative sum gives E at the
  // cell faces i+1/2 (equivalently, the periodic discrete Poisson solution for phi); subtracting the mean removes
  // the net field of a periodic box. Averaging the two neighbouring faces gives E at the nodes, i.e. the central
  // difference -(phi_{i+1} - phi_{i-1})/2dx. With the same CIC kernel for deposit and interpolation this has no
  // self-force and conserves total momentum to round-off.
  const double n0 = static_cast<double>(cfg_.n_particles) / ng;
  std::vector<double> face(ng);
  double acc = 0, mean = 0;
  for (int i = 0; i < ng; ++i) { acc += (1.0 - rho_[i] / n0) * dx_; face[i] = acc; mean += acc; }
  mean /= ng; field_energy_ = 0;
  for (int i = 0; i < ng; ++i) { face[i] -= mean; field_energy_ += 0.5 * face[i] * face[i] * dx_; }
  for (int i = 0; i < ng; ++i) e_[i] = 0.5 * (face[(i + ng - 1) % ng] + face[i]);
}

double Simulation::e_at(double x) const {
  const int ng = cfg_.n_grid;
  const double g = x / dx_ - 0.5; const double fl = std::floor(g); const double f = g - fl;
  const int i = static_cast<int>(fl);
  return (1.0 - f) * e_[(i + ng) % ng] + f * e_[(i + 1) % ng];
}

void Simulation::step() {
  const std::int64_t n = static_cast<std::int64_t>(x_.size());
  const double dt = cfg_.dt, L = cfg_.length;
#ifdef PIC_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t j = 0; j < n; ++j) {
    v_[j] -= dt * e_at(x_[j]);                          // electrons: dv/dt = -E
    double x = x_[j] + dt * v_[j];
    x = std::fmod(x, L); if (x < 0) x += L;
    x_[j] = x;
  }
  deposit_and_solve();
  t_ += dt;
}

double Simulation::total_momentum() const {
  double p = 0;
  for (double v : v_) p += v;
#ifdef PIC_HAVE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return p / static_cast<double>(cfg_.n_particles);
}

double Simulation::kinetic_energy() const {
  double k = 0;
  for (double v : v_) k += 0.5 * v * v;
#ifdef PIC_HAVE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  // normalise so that field and kinetic energies share units (n0 * L particles per box)
  return k * cfg_.length / static_cast<double>(cfg_.n_particles);
}

void Simulation::run(Diagnostics& d, bool verbose) {
  const int nsteps = static_cast<int>(std::lround(cfg_.t_end / cfg_.dt));
  for (int s = 0; s < nsteps; ++s) {
    step();
    d.time.push_back(t_); d.field_energy.push_back(field_energy_);
    d.kinetic_energy.push_back(kinetic_energy()); d.momentum.push_back(total_momentum());
    if (verbose && rank_ == 0 && (s % 50 == 0 || s == nsteps - 1))
      std::printf("  t = %6.2f   field energy = %.3e\n", t_, field_energy_);
    if (cfg_.snapshot_every > 0 && (s + 1) % cfg_.snapshot_every == 0) {
      char name[256];
      std::snprintf(name, sizeof name, "%s/phase_%06d.csv", cfg_.out_dir.c_str(), s + 1);
      write_phase_space(*this, name, 20000);
    }
  }
}

double two_stream_growth_rate(double k, double v0) {
  // 1 = (1/2)[1/(w - k v0)^2 + 1/(w + k v0)^2]  =>  s = w^2 = [(2a^2+1) - sqrt(8a^2+1)]/2,  a = k v0
  const double a2 = (k * v0) * (k * v0);
  const double s = 0.5 * ((2 * a2 + 1) - std::sqrt(8 * a2 + 1));
  return s < 0 ? std::sqrt(-s) : 0.0;
}

double measure_growth_rate(const Diagnostics& d, double t0, double t1) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
  for (std::size_t i = 0; i < d.time.size(); ++i) {
    if (d.time[i] < t0 || d.time[i] > t1 || d.field_energy[i] <= 0) continue;
    const double x = d.time[i], y = 0.5 * std::log(d.field_energy[i]);  // W_E ~ exp(2 gamma t)
    sx += x; sy += y; sxx += x * x; sxy += x * y; ++n;
  }
  if (n < 3) return 0;
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

}  // namespace pic
