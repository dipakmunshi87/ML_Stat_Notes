! cfun.f90 -- Fortran callable from C, C++ and Python through iso_c_binding (Fortran 2003).
! Build a shared library: gfortran -O3 -shared -fPIC cfun.f90 -o libcfun.so   (.dylib on macOS)
module cfun
  use, intrinsic :: iso_c_binding, only: c_double, c_int
  implicit none
contains
  subroutine trapz(n, x, y, result) bind(C, name="trapz")
    integer(c_int), value, intent(in) :: n
    real(c_double), intent(in) :: x(n), y(n)
    real(c_double), intent(out) :: result
    result = 0.5_c_double * sum((x(2:n) - x(1:n-1)) * (y(2:n) + y(1:n-1)))
  end subroutine trapz
end module cfun
