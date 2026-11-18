#include <cstdio>
#include <stdexcept>
#include <vector>

#include "pic/pic.hpp"
#ifdef PIC_HAVE_MPI
#include <mpi.h>
#endif

namespace pic {

void write_diagnostics(const Diagnostics& d, const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::fprintf(f, "time,field_energy,kinetic_energy,momentum\n");
  for (std::size_t i = 0; i < d.time.size(); ++i)
    std::fprintf(f, "%.6f,%.10e,%.10e,%.10e\n", d.time[i], d.field_energy[i], d.kinetic_energy[i], d.momentum[i]);
  std::fclose(f);
}

// Each rank contributes an evenly strided subsample; rank 0 gathers and writes.
void write_phase_space(const Simulation& s, const std::string& path, std::int64_t max_points) {
  const auto& x = s.x(); const auto& v = s.v();
  const std::int64_t n_local = static_cast<std::int64_t>(x.size());
  const std::int64_t per_rank = std::max<std::int64_t>(1, max_points / s.nranks());
  const std::int64_t stride = std::max<std::int64_t>(1, n_local / per_rank);
  std::vector<double> buf;
  for (std::int64_t j = 0; j < n_local; j += stride) { buf.push_back(x[j]); buf.push_back(v[j]); }
  std::vector<double> all = buf;
#ifdef PIC_HAVE_MPI
  int nr = s.nranks(); int mine = static_cast<int>(buf.size());
  std::vector<int> counts(nr), displs(nr, 0);
  MPI_Gather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (s.rank() == 0) { for (int r = 1; r < nr; ++r) displs[r] = displs[r - 1] + counts[r - 1];
                       all.resize(displs[nr - 1] + counts[nr - 1]); }
  MPI_Gatherv(buf.data(), mine, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  if (s.rank() != 0) return;
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::fprintf(f, "x,v\n");
  for (std::size_t i = 0; i + 1 < all.size(); i += 2) std::fprintf(f, "%.6f,%.6f\n", all[i], all[i + 1]);
  std::fclose(f);
}

}  // namespace pic
