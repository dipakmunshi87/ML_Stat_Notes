! heat.f90 -- modern Fortran (2008+): modules, derived types, array syntax, pure/elemental, do concurrent.
! Build: gfortran -O3 -std=f2018 -Wall heat.f90 -o heat && ./heat
module grid_mod
  use, intrinsic :: iso_fortran_env, only: dp => real64
  implicit none
  private
  public :: dp, grid_t, init_gaussian, step, total_heat

  type :: grid_t
     integer :: n
     real(dp) :: dx, kappa
     real(dp), allocatable :: u(:)            ! allocatable: freed automatically (RAII-like)
  end type grid_t

contains

  pure elemental real(dp) function gaussian(x, sigma)
    real(dp), intent(in) :: x, sigma
    gaussian = exp(-0.5_dp * (x / sigma)**2)
  end function gaussian

  subroutine init_gaussian(g, n, length, kappa)
    type(grid_t), intent(out) :: g
    integer, intent(in) :: n
    real(dp), intent(in) :: length, kappa
    integer :: i
    g%n = n; g%dx = length / n; g%kappa = kappa
    allocate(g%u(n))
    g%u = gaussian([(real(i, dp) * g%dx - 0.5_dp * length, i = 1, n)], 0.05_dp * length)  ! array constructor + elemental
  end subroutine init_gaussian

  subroutine step(g, dt)                     ! explicit diffusion step, periodic boundaries
    type(grid_t), intent(inout) :: g
    real(dp), intent(in) :: dt
    real(dp) :: lap(g%n)
    integer :: i
    do concurrent (i = 1:g%n)                ! iterations independent: compiler may vectorise/parallelise
       lap(i) = g%u(modulo(i - 2, g%n) + 1) - 2.0_dp * g%u(i) + g%u(modulo(i, g%n) + 1)
    end do
    g%u = g%u + g%kappa * dt / g%dx**2 * lap  ! whole-array arithmetic
  end subroutine step

  pure real(dp) function total_heat(g)
    type(grid_t), intent(in) :: g
    total_heat = sum(g%u) * g%dx
  end function total_heat

end module grid_mod

program heat
  use grid_mod
  implicit none
  type(grid_t) :: g
  real(dp) :: dt, q0
  integer :: it
  call init_gaussian(g, 400, 1.0_dp, 1.0e-3_dp)
  dt = 0.4_dp * g%dx**2 / g%kappa             ! stability: dt <= dx^2 / (2 kappa)
  q0 = total_heat(g)
  do it = 1, 5000
     call step(g, dt)
  end do
  print '(a, es10.3, a, f8.5)', 'relative heat change ', abs(total_heat(g) - q0) / q0, &
        ';  peak now ', maxval(g%u)
end program heat
