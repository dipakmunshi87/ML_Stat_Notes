// 1D-1V electrostatic particle-in-cell code: electrons on a fixed neutralising ion background.
// Plasma units: omega_p = 1, Debye-scaled lengths. Companion to Section 9 of
// "Kinetic Hierarchies: from Liouville to Euler" (the same algorithm as the two-stream animation).
//
//   cloud-in-cell deposit -> Gauss's law dE/dx = 1 - n_e/n0 -> CIC interpolation -> leapfrog push
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pic {

struct Config {
  std::int64_t n_particles = 400000;  // global number of macro-particles
  int n_grid = 64;                    // grid cells
  double length = 12.566370614359172; // box length L (4 pi: fundamental k = 0.5)
  double v0 = 1.0;                    // beam speeds +-v0
  double vth = 0.02;                  // thermal spread of each beam
  double dt = 0.1;                    // time step (1/omega_p)
  double t_end = 40.0;                // end time
  double perturbation = 1e-3;         // initial displacement amplitude (fraction of L), mode 1
  int single_beam = 0;                // 1: one beam only (stable test case)
  std::uint64_t seed = 7;
  int snapshot_every = 0;             // write phase-space subsample every n steps (0: never)
  std::string out_dir = "output";
};

struct Diagnostics {
  std::vector<double> time, field_energy, kinetic_energy, momentum;
};

class Simulation {
 public:
  explicit Simulation(const Config& cfg);
  void step();                         // one leapfrog step (velocities live at half steps)
  void run(Diagnostics& d, bool verbose);
  double time() const { return t_; }
  double field_energy() const { return field_energy_; }
  double total_momentum() const;       // global, summed over ranks
  double kinetic_energy() const;       // global, at the half step
  const std::vector<double>& x() const { return x_; }
  const std::vector<double>& v() const { return v_; }
  int rank() const { return rank_; }
  int nranks() const { return nranks_; }

 private:
  void deposit_and_solve();           // charge deposit (all ranks) + Gauss's law
  double e_at(double x) const;
  Config cfg_;
  std::vector<double> x_, v_, rho_, e_;
  double t_ = 0, field_energy_ = 0, dx_;
  int rank_ = 0, nranks_ = 1;
};

// cold two-stream dispersion relation, symmetric beams: growth rate of mode k (0 if stable)
double two_stream_growth_rate(double k, double v0);
// least-squares slope of 0.5 ln W_E over [t0, t1]: the measured growth rate
double measure_growth_rate(const Diagnostics& d, double t0, double t1);

void write_diagnostics(const Diagnostics& d, const std::string& path);
void write_phase_space(const Simulation& s, const std::string& path, std::int64_t max_points);

}  // namespace pic
