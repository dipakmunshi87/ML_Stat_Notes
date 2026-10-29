cmake_minimum_required(VERSION 3.18)
project(kinetic_pic VERSION 1.0 LANGUAGES CXX
        DESCRIPTION "1D electrostatic particle-in-cell (two-stream instability) for laptops and HPC")

# ---------------------------------------------------------------- options
option(PIC_USE_OPENMP "Thread the particle loops with OpenMP (if available)" ON)
option(PIC_USE_MPI    "Distribute particles across MPI ranks"                OFF)
option(PIC_BUILD_TESTS "Build the ctest physics tests"                      ON)
option(PIC_NATIVE     "Optimise for the build machine (-march=native); turn OFF for heterogeneous clusters" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------- library
add_library(pic src/pic.cpp src/io.cpp)
target_include_directories(pic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(pic PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pic PRIVATE -Wall -Wextra -Wpedantic)
  if(PIC_NATIVE)
    target_compile_options(pic PUBLIC -march=native)
  endif()
endif()

# OpenMP is optional: Apple clang has none by default (brew install libomp), most HPC compilers do.
if(PIC_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(pic PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(pic PUBLIC PIC_HAVE_OPENMP)
    message(STATUS "OpenMP: enabled")
  else()
    message(WARNING "OpenMP requested but not found: building serial. On macOS: brew install libomp")
  endif()
endif()

if(PIC_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(pic PUBLIC MPI::MPI_CXX)
  target_compile_definitions(pic PUBLIC PIC_HAVE_MPI)
  message(STATUS "MPI: enabled (${MPI_CXX_COMPILER})")
endif()

# ---------------------------------------------------------------- executable
add_executable(pic_two_stream src/main.cpp)
target_link_libraries(pic_two_stream PRIVATE pic)
install(TARGETS pic_two_stream RUNTIME DESTINATION bin)

# ---------------------------------------------------------------- tests
if(PIC_BUILD_TESTS)
  enable_testing()
  add_executable(test_physics tests/test_physics.cpp)
  target_link_libraries(test_physics PRIVATE pic)
  add_test(NAME growth_rate      COMMAND test_physics growth)
  add_test(NAME momentum_conservation COMMAND test_physics momentum)
  add_test(NAME stable_single_beam   COMMAND test_physics stable)
  if(PIC_USE_MPI)
    add_test(NAME growth_rate_mpi4 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
             $<TARGET_FILE:test_physics> growth ${MPIEXEC_POSTFLAGS})
  endif()
  set_tests_properties(growth_rate PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
endif()
