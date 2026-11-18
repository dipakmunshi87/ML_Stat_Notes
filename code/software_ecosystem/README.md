# Code for "From Laptop to Supercomputer: The Scientific Coder's Field Guide to Python, C++ and Fortran"

Every example in the note [software_ecosystem.html](https://dipakmunshi87.github.io/ML_Stat_Notes/software_ecosystem.html),
exactly as run (Python 3.13, GCC 13).

| File | Chapter | Run |
|---|---|---|
| `p01_core.py` | 1 Python core | `python p01_core.py` |
| `p02_numpy.py` | 3 NumPy broadcasting | `python p02_numpy.py` |
| `p03_frames.py` | 3 pandas / Polars / DuckDB | `pip install polars pyarrow duckdb` |
| `p04_sklearn.py` | 4 scikit-learn pipeline | `python p04_sklearn.py` |
| `p05_jax.py` | 4 JAX grad / jit / vmap | `pip install jax` |
| `p06_torch.py` | 4 PyTorch training loop | `pip install torch` |
| `p07_numba.py` | 5 Numba | `pip install numba` |
| `p08_scipy.py` | 3 SciPy ODE + fit | `python p08_scipy.py` |
| `p09_parallel.py` | 5 process pool | `python p09_parallel.py` |
| `pkg/` | 2, 6 packaging, pytest, CI | `cd pkg && pip install -e ".[test]" && pytest` |
| `cpp/modern.cpp` | 9 modern C++ | `g++ -std=c++20 -O2 modern.cpp -o modern && ./modern` |
| `cpp/omp_sum.cpp` | 11 OpenMP | `g++ -std=c++17 -O3 -fopenmp omp_sum.cpp -o omp_sum && ./omp_sum` |
| `cpp/pyext/` | 11 pybind11 | `pip install pybind11 && cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir) && cmake --build build && PYTHONPATH=build python check.py` |
| `f90/heat.f90` | 13 modern Fortran | `gfortran -O3 -std=f2018 heat.f90 -o heat && ./heat` |
| `f90/cfun.f90`, `call_fortran.py` | 13 Fortran from Python (ctypes) | `gfortran -O3 -shared -fPIC cfun.f90 -o libcfun.so` (`.dylib` on macOS), then `python call_fortran.py` |
| `f90/stats.f90` | 13 f2py | `pip install meson ninja && python -m numpy.f2py -c stats.f90 -m fstats` |

On macOS: `brew install gcc cmake libomp` (Apple clang has no OpenMP; use `g++-14` from Homebrew for OpenMP).

## Debugging and profiling (Chapters 15–16)

| File | What it shows | Run |
|---|---|---|
| `debug/nan_bug.py` | silent NaN made loud with `np.seterr`; post-mortem pdb | `python -m pdb -c continue nan_bug.py` |
| `debug/oob.cpp` | off-by-one found by AddressSanitizer | `g++ -g -O1 -fsanitize=address,undefined oob.cpp -o oob && ./oob` |
| `debug/crash.cpp` | segfault traced with gdb / lldb | `g++ -g -O0 crash.cpp -o crash && gdb ./crash` (macOS: `lldb ./crash`) |
| `debug/bounds.f90` | Fortran bounds checking | `gfortran -g -O0 -fcheck=all -fbacktrace bounds.f90 -o b && ./b` |
| `profile/slow_pipeline.py`, `fix_kdtree.py` | cProfile finds an O(n²) loop; KD-tree fix | `python -m cProfile -s cumulative slow_pipeline.py` |
| `profile/async_timing.py` | why GPU/JAX timing needs `block_until_ready` | `python async_timing.py` |
| `profile/torch_profile.py` | PyTorch profiler table and Perfetto trace | `python torch_profile.py` |
| `profile/hotspot.cpp` | callgrind (Linux) / Instruments (macOS) | `g++ -O2 -g hotspot.cpp -o hotspot && valgrind --tool=callgrind ./hotspot` |

## Appendix A: GitHub, CI/CD and automated testing

| File | What it is |
|---|---|
| `ci/cpp_kepler/` | small C++20 CMake project with ctest and a sanitizer option: `cmake -S . -B build -DENABLE_SANITIZERS=ON && cmake --build build && ctest --test-dir build` |
| `ci/workflows/cpp.yml` | GitHub Actions: Release + sanitizer builds on Linux and macOS |
| `ci/workflows/fortran.yml` | gfortran with run-time checks |
| `ci/workflows/pipeline-smoke.yml` | end-to-end smoke test of a research pipeline from a fresh install |
| `ci/workflows/docs.yml` | MkDocs documentation to GitHub Pages |
| `ci/workflows/release.yml` | tagged releases to PyPI with trusted publishing |

To use a workflow, copy it into your repository's `.github/workflows/` folder. All pass `actionlint`.

## Appendix B: OpenMP, MPI and their relatives

| File | What it shows | Run |
|---|---|---|
| `par/omp_pitfalls.cpp` | a data race (lost counts) and three correct fixes | `g++ -std=c++17 -O2 -fopenmp omp_pitfalls.cpp -o p && OMP_NUM_THREADS=4 ./p` |
| `par/mpi_heat.cpp` | domain decomposition with halo exchange | `mpicxx -std=c++17 -O2 mpi_heat.cpp -o h && mpirun -np 4 ./h` |
| `par/hybrid_pi.cpp` | hybrid MPI + OpenMP | `mpicxx -std=c++17 -O2 -fopenmp hybrid_pi.cpp -o hp && OMP_NUM_THREADS=2 mpirun -np 2 ./hp` |
| `par/mpi_omp.f90` | Fortran `mpi_f08` + OpenMP | `mpif90 -O2 -fopenmp mpi_omp.f90 -o f && OMP_NUM_THREADS=2 mpirun -np 2 ./f` |
| `par/mpi4py_halo.py` | the halo exchange in Python | `mpiexec -n 4 python mpi4py_halo.py` |

On macOS: `brew install open-mpi libomp` (or `conda install -c conda-forge openmpi mpi4py`).

## Appendix D: Docker

`docker_demo/`: one image with C++, Fortran and Python. `cd docker_demo && docker build -t science-demo . && docker run --rm science-demo`
(on Apple silicon add `--platform linux/amd64` to `docker build` if the image is for an x86-64 cluster).
