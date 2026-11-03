// mpi_heat.cpp -- domain decomposition with halo exchange: 1D diffusion split across MPI ranks.
// Build: mpicxx -std=c++17 -O2 mpi_heat.cpp -o mpi_heat      Run: mpirun -np 4 ./mpi_heat
#include <cmath>
#include <cstdio>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4000;                              // global grid (periodic)
    const int n = N / size;                          // cells owned by this rank (assume divisible)
    const double dx = 1.0 / N, kappa = 1e-3, dt = 0.4 * dx * dx / kappa;
    std::vector<double> u(n + 2), unew(n + 2);       // u[0] and u[n+1] are ghost (halo) cells
    for (int i = 1; i <= n; ++i) {
        const double x = (rank * n + i - 0.5) * dx - 0.5;
        u[i] = std::exp(-0.5 * (x / 0.05) * (x / 0.05));
    }
    const int left = (rank - 1 + size) % size, right = (rank + 1) % size;

    auto total_heat = [&] {
        double local = 0.0, global = 0.0;
        for (int i = 1; i <= n; ++i) local += u[i] * dx;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   // every rank gets the sum
        return global;
    };
    const double q0 = total_heat();
    const double t0 = MPI_Wtime();
    for (int step = 0; step < 20000; ++step) {
        // halo exchange: send my edge cells to neighbours, receive theirs into my ghost cells
        MPI_Sendrecv(&u[n], 1, MPI_DOUBLE, right, 0, &u[0],     1, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left,  1, &u[n + 1], 1, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 1; i <= n; ++i) unew[i] = u[i] + kappa * dt / (dx * dx) * (u[i - 1] - 2 * u[i] + u[i + 1]);
        std::swap(u, unew);
    }
    const double elapsed = MPI_Wtime() - t0, q1 = total_heat();
    double peak_local = 0.0, peak = 0.0;
    for (int i = 1; i <= n; ++i) peak_local = std::max(peak_local, u[i]);
    MPI_Reduce(&peak_local, &peak, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);         // only rank 0 gets it
    if (rank == 0)
        std::printf("ranks %d: heat conserved to %.1e, peak %.5f, %.2f s\n", size, std::abs(q1 - q0) / q0, peak, elapsed);
    MPI_Finalize();
}
