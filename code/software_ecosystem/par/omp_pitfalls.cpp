// omp_pitfalls.cpp -- the classic OpenMP bug (a data race) and three correct fixes.
// Build: g++ -std=c++17 -O2 -fopenmp omp_pitfalls.cpp -o omp_pitfalls && OMP_NUM_THREADS=4 ./omp_pitfalls
#include <cstdio>
#include <vector>
#include <omp.h>

constexpr int NBINS = 16;

void racy(const std::vector<int>& x, long* h) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < x.size(); ++i) h[x[i]] += 1;            // WRONG: unsynchronised update
}
void atomic(const std::vector<int>& x, long* h) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < x.size(); ++i) {
        #pragma omp atomic
        h[x[i]] += 1;                                                    // correct; threads contend
    }
}
void array_reduction(const std::vector<int>& x, long* h) {
    #pragma omp parallel for reduction(+ : h[:NBINS])                    // private copies, summed at the end
    for (std::size_t i = 0; i < x.size(); ++i) h[x[i]] += 1;
}
void private_merge(const std::vector<int>& x, long* h) {
    #pragma omp parallel
    {
        long mine[NBINS] = {};                                           // thread-private histogram
        #pragma omp for nowait
        for (std::size_t i = 0; i < x.size(); ++i) mine[x[i]] += 1;
        #pragma omp critical                                             // one thread at a time, NBINS adds
        for (int b = 0; b < NBINS; ++b) h[b] += mine[b];
    }
}

int main() {
    const int n = 20'000'000;
    std::vector<int> x(n);
    for (int i = 0; i < n; ++i) x[i] = (i * 2654435761u) % NBINS;        // e.g. particle -> cell index
    struct { const char* name; void (*f)(const std::vector<int>&, long*); } variants[] = {
        {"racy (WRONG)", racy}, {"atomic", atomic}, {"array reduction", array_reduction}, {"private + merge", private_merge}};
    std::printf("threads: %d\n", omp_get_max_threads());
    for (auto& v : variants) {
        long h[NBINS] = {};
        const double t0 = omp_get_wtime();
        v.f(x, h);
        const double dt = omp_get_wtime() - t0;
        long total = 0;
        for (long c : h) total += c;
        std::printf("%-17s counted %9ld of %d  (%.3f s)\n", v.name, total, n, dt);
    }
}
