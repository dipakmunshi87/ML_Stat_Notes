! mpi_omp.f90 -- hybrid MPI (mpi_f08, the modern Fortran binding) + OpenMP: a Monte Carlo estimate of pi.
! Build: mpif90 -O2 -fopenmp mpi_omp.f90 -o mpi_omp      Run: OMP_NUM_THREADS=2 mpirun -np 2 ./mpi_omp
program mpi_omp
  use mpi_f08
  use omp_lib
  use, intrinsic :: iso_fortran_env, only: dp => real64, i8 => int64
  implicit none
  integer :: rank, nranks, provided
  integer(i8) :: i, hits_local, hits, n_local
  integer(i8), parameter :: n_total = 80000000_i8
  real(dp) :: x, y
  integer :: seed_size, tid
  integer, allocatable :: seed(:)

  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks)
  n_local = n_total / nranks
  hits_local = 0

  !$omp parallel private(x, y, i, seed, seed_size, tid) reduction(+:hits_local)
  tid = omp_get_thread_num()
  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 12345 + 1000 * rank + 17 * tid          ! a different random stream per rank and thread
  call random_seed(put=seed)
  !$omp do
  do i = 1, n_local
     call random_number(x); call random_number(y)
     if (x*x + y*y <= 1.0_dp) hits_local = hits_local + 1
  end do
  !$omp end do
  !$omp end parallel

  call MPI_Reduce(hits_local, hits, 1, MPI_INTEGER8, MPI_SUM, 0, MPI_COMM_WORLD)
  if (rank == 0) print '(i0, a, i0, a, f10.6)', nranks, ' ranks x ', omp_get_max_threads(), &
       ' threads: pi ~ ', 4.0_dp * real(hits, dp) / real(n_local * nranks, dp)
  call MPI_Finalize()
end program mpi_omp
