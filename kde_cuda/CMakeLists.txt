# Minimum version of CMake required.
cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
# Set the name of your project
project(KDEPROJECT)

# Include macros from the ParallelUtils library
include(ParallelUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to #
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################


## Example MPI program
#enable_mpi_support()
#add_executable(integrate_mpi integrate_mpi.cpp)
#target_link_libraries(integrate_mpi ${MPI_CXX_LIBRARIES})

## Example OpenMP program
#enable_openmp_support()
#add_executable(kde_omp kde_omp.cpp)

## Example CUDA program
enable_cuda_support()
#1 without any optimization
cuda_add_executable(kde_cuda_old kde_cuda_old.cu)
#2 partial optimization (avoiding re-computing edge correction factors)
cuda_add_executable(kde_cuda kde_cuda.cu)
#3 full optimization (avoiding re-computing edge correction factors and kd-tree indexing)
cuda_add_executable(kde_cuda_kdtr kde_cuda_kdtr.cu KDtree.cu CUDA_KDtree.cu)

set_target_properties(kde_cuda_old PROPERTIES COMPILE_FLAGS "-O4")
set_target_properties(kde_cuda PROPERTIES COMPILE_FLAGS "-O4")
set_target_properties(kde_cuda_kdtr PROPERTIES COMPILE_FLAGS "-O4")
