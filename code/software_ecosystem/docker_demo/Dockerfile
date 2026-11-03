# Dockerfile -- one image containing C++, Fortran and Python, built from source.
# Build:  docker build -t science-demo .
# Run:    docker run --rm science-demo
FROM ubuntu:24.04

# 1. System packages: compilers and Python, installed once and cached as an image layer
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ gfortran python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/*

# 2. Copy the source code into the image
WORKDIR /app
COPY modern.cpp heat.f90 check_numpy.py run_all.sh ./

# 3. Compile the C++ and Fortran programs inside the image
RUN g++ -std=c++20 -O2 modern.cpp -o modern \
 && gfortran -O2 -std=f2018 heat.f90 -o heat

# 4. What runs when the container starts
CMD ["./run_all.sh"]
