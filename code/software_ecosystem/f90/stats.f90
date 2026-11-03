! stats.f90 -- wrapped automatically for Python by f2py (ships with NumPy).
! Build: python -m numpy.f2py -c stats.f90 -m fstats      (needs meson + ninja: pip install meson ninja)
subroutine moments(x, n, mean, var)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  double precision, intent(out) :: mean, var
  mean = sum(x) / n
  var = sum((x - mean)**2) / (n - 1)
end subroutine moments
