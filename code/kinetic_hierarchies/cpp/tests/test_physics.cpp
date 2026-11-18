// Physics regression tests, run by ctest. Each returns 0 on pass.
//   growth    : linear growth rate of the two-stream instability within 10% of cold-beam theory
//   momentum  : total momentum conserved to round-off (CIC deposit + same-kernel interpolation, no self-force)
//   stable    : a single beam (no counter-stream) does not grow
#include <cmath>
#include <cstdio>
#include <cstring>

#include "pic/pic.hpp"
#ifdef PIC_HAVE_MPI
#include <mpi.h>
#endif

static int report(bool ok, const char* what, int rank) {
  if (rank == 0) std::printf("%s: %s\n", what, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
#ifdef PIC_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  const char* which = argc > 1 ? argv[1] : "growth";
  pic::Config cfg; cfg.n_particles = 200000; cfg.t_end = 20.0; cfg.out_dir = ".";
  int rc = 0, rank = 0;
#ifdef PIC_HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (!std::strcmp(which, "growth")) {
    pic::Simulation s(cfg); pic::Diagnostics d; s.run(d, false);
    const double g = pic::measure_growth_rate(d, 8.0, 16.0);
    const double g0 = pic::two_stream_growth_rate(6.283185307179586 / cfg.length, cfg.v0);
    if (rank == 0) std::printf("measured %.4f  theory %.4f\n", g, g0);
    rc = report(std::fabs(g / g0 - 1) < 0.10, "growth rate within 10% of theory", rank);
  } else if (!std::strcmp(which, "momentum")) {
    pic::Simulation s(cfg); const double p0 = s.total_momentum(); pic::Diagnostics d; s.run(d, false);
    const double dp = std::fabs(s.total_momentum() - p0);
    if (rank == 0) std::printf("momentum change %.3e\n", dp);
    rc = report(dp < 1e-9, "momentum conserved", rank);
  } else if (!std::strcmp(which, "stable")) {
    cfg.single_beam = 1; pic::Simulation s(cfg); pic::Diagnostics d; s.run(d, false);
    const double grow = d.field_energy.back() / (d.field_energy.front() + 1e-300);
    if (rank == 0) std::printf("field-energy ratio end/start %.3e\n", grow);
    rc = report(grow < 10.0, "single beam stays stable", rank);
  } else {
    if (rank == 0) std::printf("unknown test %s\n", which);
    rc = 2;
  }
#ifdef PIC_HAVE_MPI
  MPI_Finalize();
#endif
  return rc;
}
