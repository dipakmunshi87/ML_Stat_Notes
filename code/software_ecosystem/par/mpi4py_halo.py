"""mpi4py: the same halo exchange in Python, sending NumPy arrays without copying.
Run: mpiexec -n 4 python mpi4py_halo.py"""
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
cart = comm.Create_cart(dims=[size], periods=[True])     # a periodic 1D process topology
left, right = cart.Shift(0, 1)                            # neighbour ranks

N = 4000; n = N // size
dx, kappa = 1.0 / N, 1e-3
dt = 0.4 * dx * dx / kappa
x = (rank * n + np.arange(1, n + 1) - 0.5) * dx - 0.5
u = np.zeros(n + 2); u[1:-1] = np.exp(-0.5 * (x / 0.05) ** 2)

q0 = cart.allreduce(u[1:-1].sum() * dx, op=MPI.SUM)      # lower-case methods: any Python object
for step in range(20000):
    cart.Sendrecv(u[n:n + 1], dest=right, recvbuf=u[0:1], source=left)     # upper-case: buffers, fast
    cart.Sendrecv(u[1:2], dest=left, recvbuf=u[n + 1:n + 2], source=right)
    u[1:-1] += kappa * dt / dx**2 * (u[:-2] - 2 * u[1:-1] + u[2:])
q1 = cart.allreduce(u[1:-1].sum() * dx, op=MPI.SUM)
peak = cart.reduce(u[1:-1].max(), op=MPI.MAX, root=0)
if rank == 0:
    print(f"ranks {size}: heat conserved to {abs(q1 - q0) / q0:.1e}, peak {peak:.5f}")
