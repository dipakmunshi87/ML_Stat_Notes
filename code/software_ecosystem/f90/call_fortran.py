import ctypes, sys
import numpy as np

lib = ctypes.CDLL("./libcfun.dylib" if sys.platform == "darwin" else "./libcfun.so")
dptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
lib.trapz.argtypes = [ctypes.c_int, dptr, dptr, ctypes.POINTER(ctypes.c_double)]
lib.trapz.restype = None

x = np.linspace(0, np.pi, 100_001)
out = ctypes.c_double()
lib.trapz(x.size, x, np.sin(x), ctypes.byref(out))      # no copies: Fortran reads NumPy's memory
print(f"integral of sin from 0 to pi = {out.value:.10f}  (exact 2)")
