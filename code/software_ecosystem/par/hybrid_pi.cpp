// hybrid_pi.cpp -- MPI between processes + OpenMP threads inside each: the standard HPC pattern.
// Build: mpicxx -std=c++17 -O2 -fopenmp hybrid_pi.cpp -o hybrid_pi
// Run:   OMP_NUM_THREADS=2 mpirun -np 2 ./hybrid_pi      (on a cluster: ranks per node x threads per rank = cores)
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);   // only the main thread calls MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long n = 200'000'000;                  // midpoint rule for pi = integral of 4/(1+x^2) on [0,1]
    const long lo = rank * n / size, hi = (rank + 1) * n / size;   // this rank's share of the intervals
    const double h = 1.0 / n;
    double local = 0.0;
    #pragma omp parallel for reduction(+ : local)                   // threads split this rank's share
    for (long i = lo; i < hi; ++i) {
        const double x = (i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }
    double pi = 0.0;
    MPI_Reduce(&local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("%d ranks x %d threads: pi = %.12f (error %.1e)\n",
                    size, omp_get_max_threads(), pi * h, std::abs(pi * h - M_PI));
    MPI_Finalize();
}
