"""Python part of the demo: confirms NumPy works and prints its version."""
import platform
import numpy as np

x = np.linspace(0, np.pi, 100_001)
y = np.sin(x)
integral = np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2      # trapezium rule, works on any NumPy version
print(f"Python {platform.python_version()}, NumPy {np.__version__}: integral of sin = {integral:.8f}")
