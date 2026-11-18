! bounds.f90 -- an out-of-bounds write that gfortran's run-time checks catch.
! Debug build: gfortran -g -O0 -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow bounds.f90 -o bounds
program bounds
  implicit none
  integer, parameter :: n = 5
  real :: u(n)
  integer :: i
  u = 1.0
  do i = 1, n + 1            ! bug: one past the end
     u(i) = 2.0 * u(max(i - 1, 1))
  end do
  print *, sum(u)
end program bounds
