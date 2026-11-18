#!/bin/sh
# Runs the three programs built in the image: C++, Fortran and Python.
set -e
echo "--- C++ (g++ $(g++ -dumpfullversion))";  ./modern | head -3
echo "--- Fortran (gfortran $(gfortran -dumpfullversion))"; ./heat
echo "--- Python"; python3 check_numpy.py
