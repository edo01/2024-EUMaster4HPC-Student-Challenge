# establishes policy settings and ensures that the following CMake
# functions are run with a compatible version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(TEST_CG VERSION 1.0)

# Set icpx compiler
#set(CMAKE_CXX_COMPILER "icpx")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# Define default build type if not provided
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Set build type options
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
  STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add an option for enabling fast math
option(USE_FAST_MATH "Enable fast math optimizations." OFF)

if(USE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funsafe-math-optimizations")
    message(STATUS "Enabled fast math optimizations (-Ofast -funsafe-math-optimizations)")
endif()

# find required packages
# find MPI
find_package(MPI REQUIRED)
# find OpenMP
#cmake_policy(SET CMP0074 NEW)
find_package(OpenMP REQUIRED)

# Check if CUDA is available and add CUDA source files
find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA) #enable NVCC compiler

    # Add CUDA macro for conditional compilation
    add_definitions(-DUSE_CUDA)

    # Set policy to use CUDA_ARCHITECTURES
    #cmake_policy(SET CMP0104 NEW)

    # Set CUDA_ARCHITECTURES to compute capability 6.1
    #set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

#add_subdirectory(LAM)
add_subdirectory(test)