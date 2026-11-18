// pic_two_stream: run the two-stream instability and report the growth rate against theory.
//
//   pic_two_stream [--particles N] [--grid NG] [--v0 V] [--vth S] [--dt DT] [--tend T]
//                  [--snapshots EVERY] [--out DIR]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "pic/pic.hpp"
#ifdef PIC_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef PIC_HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#ifdef PIC_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  pic::Config cfg;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i]; const char* val = argv[i + 1];
    if (a == "--particles") cfg.n_particles = std::atoll(val);
    else if (a == "--grid") cfg.n_grid = std::atoi(val);
    else if (a == "--v0") cfg.v0 = std::atof(val);
    else if (a == "--vth") cfg.vth = std::atof(val);
    else if (a == "--dt") cfg.dt = std::atof(val);
    else if (a == "--tend") cfg.t_end = std::atof(val);
    else if (a == "--snapshots") cfg.snapshot_every = std::atoi(val);
    else if (a == "--out") cfg.out_dir = val;
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
  }
  int rank = 0, nranks = 1, nthreads = 1;
#ifdef PIC_HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
#ifdef PIC_HAVE_OPENMP
  nthreads = omp_get_max_threads();
#endif
  if (rank == 0) {
    std::filesystem::create_directories(cfg.out_dir);
    std::printf("two-stream PIC: %lld particles, %d cells, %d MPI rank(s) x %d OpenMP thread(s)\n",
                static_cast<long long>(cfg.n_particles), cfg.n_grid, nranks, nthreads);
  }
#ifdef PIC_HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  const auto t0 = std::chrono::steady_clock::now();
  pic::Simulation sim(cfg);
  pic::Diagnostics d;
  sim.run(d, true);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  if (rank == 0) {
    pic::write_diagnostics(d, cfg.out_dir + "/diagnostics.csv");
    const double k = 6.283185307179586 / cfg.length;
    const double g_theory = pic::two_stream_growth_rate(k, cfg.v0);
    const double g_meas = pic::measure_growth_rate(d, 8.0, 16.0);
    const double steps = cfg.t_end / cfg.dt;
    std::printf("growth rate: measured %.4f, cold-beam theory %.4f (ratio %.3f)\n", g_meas, g_theory, g_meas / g_theory);
    std::printf("wall time %.2f s, %.3e particle-steps per second\n", secs, cfg.n_particles * steps / secs);
    std::printf("wrote %s/diagnostics.csv\n", cfg.out_dir.c_str());
  }
#ifdef PIC_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
